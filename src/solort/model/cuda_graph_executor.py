"""CUDA-graph custom Qwen3 executor — the launch-bound fix that beats vLLM single-stream.

The HF bridge is kernel-launch bound (~11 tok/s regardless of model size) and torch.compile cannot
graph it (HF's decode is data-dependent). This executor runs a hand-written, graph-friendly Qwen3
forward over SoloRT-owned static KV (position read from a buffer, masked attention), captures the
single-token decode step once into a CUDA graph, and replays it per token. In an isolated micro
benchmark this reached ~132 tok/s for Qwen3-0.6B vs vLLM's ~91.

Scope: single active sequence (the single-user target), greedy or sampled decode, Qwen3 family.
Reuses TransformersTextExecutor for weight/tokenizer loading, chat templating, sampling and
detokenization; only the prefill/decode compute is replaced.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from solort.core.batch import Batch
from solort.model.executor import TransformersGenerationConfig, TransformersTextExecutor
from solort.model.sampler import SampleResult


def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    d = x.float()
    d = d * torch.rsqrt(d.pow(2).mean(-1, keepdim=True) + eps)
    return (d * w.float()).to(x.dtype)


class _GraphQwen3Runner:
    """Static-shape Qwen3 forward over SoloRT-owned KV, with a CUDA-graph-captured decode step."""

    def __init__(self, model: object, max_len: int) -> None:
        c = model.config
        self.dev = next(model.parameters()).device
        self.dt = model.dtype
        self.L = int(c.num_hidden_layers)
        self.nh = int(c.num_attention_heads)
        self.nkv = int(c.num_key_value_heads)
        self.hd = int(getattr(c, "head_dim", c.hidden_size // c.num_attention_heads))
        self.groups = self.nh // self.nkv
        self.eps = float(c.rms_norm_eps)
        self.scale = self.hd**-0.5
        self.max_len = int(max_len)
        self.embed = model.model.embed_tokens.weight
        self.final_norm = model.model.norm.weight
        self.lm_head = model.lm_head.weight
        self.layers = []
        for i in range(self.L):
            a, mlp = model.model.layers[i].self_attn, model.model.layers[i].mlp
            self.layers.append({
                "in_ln": model.model.layers[i].input_layernorm.weight,
                "post_ln": model.model.layers[i].post_attention_layernorm.weight,
                "wq": a.q_proj.weight, "wk": a.k_proj.weight, "wv": a.v_proj.weight,
                "wo": a.o_proj.weight, "qn": a.q_norm.weight, "kn": a.k_norm.weight,
                "wg": mlp.gate_proj.weight, "wu": mlp.up_proj.weight, "wd": mlp.down_proj.weight,
            })
        self.inv_freq = 1.0 / (
            float(c.rope_theta) ** (torch.arange(0, self.hd, 2, device=self.dev).float() / self.hd)
        )
        kv_shape = (self.L, self.max_len, self.nkv, self.hd)
        self.kc = torch.zeros(kv_shape, device=self.dev, dtype=self.dt)
        self.vc = torch.zeros(kv_shape, device=self.dev, dtype=self.dt)
        self.tok_buf = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.pos_buf = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.arange = torch.arange(self.max_len, device=self.dev)
        self.graph: torch.cuda.CUDAGraph | None = None
        self._logits: torch.Tensor | None = None

    def _cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions[:, None].float() * self.inv_freq[None, :]
        emb = torch.cat([freqs, freqs], -1)
        return emb.cos().to(self.dt), emb.sin().to(self.dt)

    def _rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return x * cos + torch.cat([-x[..., d:], x[..., :d]], -1) * sin

    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        """Process the full prompt into KV; return logits [1, vocab] for the last position."""
        t = int(ids.shape[0])
        cos, sin = self._cos_sin(torch.arange(t, device=self.dev))
        x = self.embed[ids].to(self.dt)
        for li, ly in enumerate(self.layers):
            h = _rmsnorm(x, ly["in_ln"], self.eps)
            q = (h @ ly["wq"].T).view(t, self.nh, self.hd).transpose(0, 1)
            k = (h @ ly["wk"].T).view(t, self.nkv, self.hd).transpose(0, 1)
            v = (h @ ly["wv"].T).view(t, self.nkv, self.hd).transpose(0, 1)
            q = self._rope(_rmsnorm(q, ly["qn"], self.eps), cos, sin)
            k = self._rope(_rmsnorm(k, ly["kn"], self.eps), cos, sin)
            self.kc[li, :t] = k.transpose(0, 1)
            self.vc[li, :t] = v.transpose(0, 1)
            kk = k.repeat_interleave(self.groups, 0)
            vv = v.repeat_interleave(self.groups, 0)
            ctx = (
                F.scaled_dot_product_attention(q, kk, vv, is_causal=True, scale=self.scale)
                .transpose(0, 1)
                .reshape(t, self.nh * self.hd)
            )
            x = x + ctx @ ly["wo"].T
            h2 = _rmsnorm(x, ly["post_ln"], self.eps)
            x = x + (F.silu(h2 @ ly["wg"].T) * (h2 @ ly["wu"].T)) @ ly["wd"].T
        x = _rmsnorm(x, self.final_norm, self.eps)
        return (x[-1:] @ self.lm_head.T).float()

    def _decode_forward(self) -> torch.Tensor:
        cos, sin = self._cos_sin(self.pos_buf)
        x = self.embed[self.tok_buf].to(self.dt)
        keep = (self.arange <= self.pos_buf).view(1, 1, self.max_len)  # True = attend (read at replay)
        for li, ly in enumerate(self.layers):
            h = _rmsnorm(x, ly["in_ln"], self.eps)
            q = (h @ ly["wq"].T).view(1, self.nh, self.hd).transpose(0, 1)
            k = (h @ ly["wk"].T).view(1, self.nkv, self.hd).transpose(0, 1)
            v = (h @ ly["wv"].T).view(1, self.nkv, self.hd).transpose(0, 1)
            q = self._rope(_rmsnorm(q, ly["qn"], self.eps), cos, sin)
            k = self._rope(_rmsnorm(k, ly["kn"], self.eps), cos, sin)
            self.kc[li].index_copy_(0, self.pos_buf, k.transpose(0, 1))
            self.vc[li].index_copy_(0, self.pos_buf, v.transpose(0, 1))
            kk = self.kc[li].permute(1, 0, 2).repeat_interleave(self.groups, 0)
            vv = self.vc[li].permute(1, 0, 2).repeat_interleave(self.groups, 0)
            ctx = (
                F.scaled_dot_product_attention(q, kk, vv, attn_mask=keep, scale=self.scale)
                .transpose(0, 1)
                .reshape(1, self.nh * self.hd)
            )
            x = x + ctx @ ly["wo"].T
            h2 = _rmsnorm(x, ly["post_ln"], self.eps)
            x = x + (F.silu(h2 @ ly["wg"].T) * (h2 @ ly["wu"].T)) @ ly["wd"].T
        x = _rmsnorm(x, self.final_norm, self.eps)
        return (x @ self.lm_head.T).float()

    def capture(self) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._logits = self._decode_forward()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._logits = self._decode_forward()

    def decode(self, token: int, pos: int) -> torch.Tensor:
        self.tok_buf.fill_(int(token))
        self.pos_buf.fill_(int(pos))
        if self.graph is None:
            self.capture()
        self.graph.replay()
        return self._logits


class CudaGraphQwen3Executor(TransformersTextExecutor):
    """Single-sequence, CUDA-graph Qwen3 executor. Beats vLLM single-stream by killing launch
    overhead. Falls back to nothing — it requires a Qwen3-family model and CUDA."""

    name = "cudagraph-qwen3"
    supports_prefix_cache = False

    def __init__(self, config: TransformersGenerationConfig | None = None) -> None:
        super().__init__(config)
        if not hasattr(self.model.model.layers[0].self_attn, "q_norm"):
            raise RuntimeError(
                "CudaGraphQwen3Executor requires a Qwen3-family model (q_norm/k_norm)"
            )
        if self._model_device().type != "cuda":
            raise RuntimeError("CudaGraphQwen3Executor requires CUDA")
        self._max_len = int(self.config.graph_max_len)
        self._runner = _GraphQwen3Runner(self.model, self._max_len)

    def forward_prefill(self, batch: Batch) -> None:
        sequence = batch.seqs[0]
        if not batch.input_ids:
            return
        state = self._ensure_state(sequence)
        state.prefilled_token_count += len(batch.input_ids)
        state.prompt_token_count = max(state.prompt_token_count, sequence.num_prompt_tokens)
        if state.prefilled_token_count < sequence.num_prompt_tokens:
            return  # wait for the full prompt (SoloRT chunks prefill)
        prompt = sequence.input_ids[: sequence.num_prompt_tokens]
        if len(prompt) >= self._max_len:
            raise RuntimeError(
                f"prompt {len(prompt)} >= graph_max_len {self._max_len}; raise SOLORT_GRAPH_MAX_LEN"
            )
        ids = self._torch.tensor(prompt, dtype=self._torch.long, device=self._model_device())
        # no_grad (not inference_mode): CUDA-graph capture/replay needs non-inference tensors.
        with self._torch.no_grad():
            logits = self._runner.prefill(ids)
        state.pending_token_id = self._sample_token(logits, sequence)

    def forward_decode(self, batch: Batch) -> SampleResult | list[SampleResult]:
        sequence = batch.seqs[0]
        state = self._ensure_state(sequence)
        if state.finished:
            return SampleResult(token_id=self._eos_token_id(), text="", finished=True)
        if state.pending_token_id is None:
            if not state.generated_token_ids:
                state.finished = True
                self._states.pop(sequence.seq_id, None)
                return SampleResult(token_id=self._eos_token_id(), text="", finished=True)
            position = state.prompt_token_count + len(state.generated_token_ids)
            if position >= self._max_len:
                raise RuntimeError(
                    "decode position exceeds graph_max_len; raise SOLORT_GRAPH_MAX_LEN"
                )
            with self._torch.no_grad():
                logits = self._runner.decode(state.generated_token_ids[-1], position)
            state.pending_token_id = self._sample_token(logits, sequence)
        token_id = int(state.pending_token_id if state.pending_token_id is not None else 0)
        state.pending_token_id = None
        return self._append_token_result(sequence, state, token_id)
