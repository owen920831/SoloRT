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
from solort.core.sequence import Sequence
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
        self.q_dim = self.nh * self.hd
        self.kv_dim = self.nkv * self.hd
        self.layers = []
        for i in range(self.L):
            a, mlp = model.model.layers[i].self_attn, model.model.layers[i].mlp
            # Fuse QKV (3 GEMMs -> 1) and gate/up (2 -> 1): bigger, more efficient GEMMs.
            wqkv = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0).contiguous()
            wgu = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0).contiguous()
            self.inter = mlp.gate_proj.weight.shape[0]
            self.layers.append({
                "in_ln": model.model.layers[i].input_layernorm.weight,
                "post_ln": model.model.layers[i].post_attention_layernorm.weight,
                "wqkv": wqkv, "wo": a.o_proj.weight, "qn": a.q_norm.weight, "kn": a.k_norm.weight,
                "wgu": wgu, "wd": mlp.down_proj.weight,
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
        # Bucketed decode graphs: attention scans only `bound` keys (not the full max_len), so short
        # sequences are much cheaper. One graph per bucket, captured lazily, routed by position.
        self.buckets = sorted(
            {b for b in (128, 256, 512, 1024, 2048, 4096) if b < self.max_len} | {self.max_len}
        )
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._logits: dict[int, torch.Tensor] = {}
        self._tok: dict[int, torch.Tensor] = {}  # in-graph greedy argmax per bucket
        # Bucketed prefill graphs (kill the eager-prefill launch overhead that dominates TTFT).
        # Finer low buckets so short prompts pad little.
        self.pre_buckets = sorted(
            {b for b in (16, 32, 64, 128, 256, 512, 1024) if b <= self.max_len}
        )
        self.pre_tok = torch.zeros(self.max_len, dtype=torch.long, device=self.dev)
        self._pre_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._pre_hidden: dict[int, torch.Tensor] = {}
        # Verify graph (fixed K+1 tokens) for speculative decoding; lazily captured.
        self.verify_graph: torch.cuda.CUDAGraph | None = None
        self.vtok: torch.Tensor | None = None
        self.vpos0: torch.Tensor | None = None
        self._vlogits: torch.Tensor | None = None

    def _cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions[:, None].float() * self.inv_freq[None, :]
        emb = torch.cat([freqs, freqs], -1)
        return emb.cos().to(self.dt), emb.sin().to(self.dt)

    def _rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return x * cos + torch.cat([-x[..., d:], x[..., :d]], -1) * sin

    def _prefill_forward(self, bound: int) -> torch.Tensor:
        """Causal forward over the padded prompt buffer pre_tok[:bound]; returns hidden [bound,hid].

        Real tokens never attend padding (padding sits at later causal positions), so the hidden at
        the real positions is exact; padding KV is garbage but the decode mask never reads it."""
        t = bound
        ids = self.pre_tok[:bound]
        cos, sin = self._cos_sin(torch.arange(t, device=self.dev))
        x = self.embed[ids].to(self.dt)
        for li, ly in enumerate(self.layers):
            h = _rmsnorm(x, ly["in_ln"], self.eps)
            qkv = h @ ly["wqkv"].T
            q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
            q = (q.view(t, self.nh, self.hd)).transpose(0, 1)
            k = (k.view(t, self.nkv, self.hd)).transpose(0, 1)
            v = (v.view(t, self.nkv, self.hd)).transpose(0, 1)
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
            gate, up = (h2 @ ly["wgu"].T).split([self.inter, self.inter], dim=-1)
            x = x + (F.silu(gate) * up) @ ly["wd"].T
        return _rmsnorm(x, self.final_norm, self.eps)  # [bound, hidden] (lm_head applied later)

    def _capture_prefill(self, bound: int) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                hid = self._prefill_forward(bound)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            hid = self._prefill_forward(bound)
        self._pre_graphs[bound], self._pre_hidden[bound] = graph, hid

    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        """Graphed bucketed prefill: pad the prompt to a bucket, replay; logits [1,vocab] at the
        last real position. Kills the eager-prefill launch overhead that dominated TTFT. The graph
        outputs hidden states ([bound,hidden], ~MBs) and the lm_head runs eagerly for one position
        (avoids materializing a [bound,vocab] logits buffer per bucket)."""
        plen = int(ids.shape[0])
        bound = next(b for b in self.pre_buckets if b >= plen)
        self.pre_tok[:plen].copy_(ids)
        self.pre_tok[plen:bound].zero_()
        if bound not in self._pre_graphs:
            self._capture_prefill(bound)
        self._pre_graphs[bound].replay()
        return (self._pre_hidden[bound][plen - 1 : plen] @ self.lm_head.T).float()

    def _decode_forward(self, bound: int) -> torch.Tensor:
        cos, sin = self._cos_sin(self.pos_buf)
        x = self.embed[self.tok_buf].to(self.dt)
        # Mask (True = ignore) for the live length; grouped attention avoids materializing the
        # GQA-expanded KV (repeat_interleave), reading the cache as [nkv, bound, hd] directly.
        nkeep = (self.arange[:bound] > self.pos_buf).view(1, 1, bound)
        for li, ly in enumerate(self.layers):
            h = _rmsnorm(x, ly["in_ln"], self.eps)
            qkv = h @ ly["wqkv"].T
            q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
            q = q.view(1, self.nh, self.hd).transpose(0, 1)
            k = k.view(1, self.nkv, self.hd).transpose(0, 1)
            v = v.view(1, self.nkv, self.hd).transpose(0, 1)
            q = self._rope(_rmsnorm(q, ly["qn"], self.eps), cos, sin)
            k = self._rope(_rmsnorm(k, ly["kn"], self.eps), cos, sin)
            self.kc[li].index_copy_(0, self.pos_buf, k.transpose(0, 1))
            self.vc[li].index_copy_(0, self.pos_buf, v.transpose(0, 1))
            qg = q.reshape(self.nkv, self.groups, self.hd)        # [nkv, groups, hd]
            kt = self.kc[li, :bound].permute(1, 2, 0)             # [nkv, hd, bound]
            vb = self.vc[li, :bound].permute(1, 0, 2)             # [nkv, bound, hd]
            sc = (torch.matmul(qg, kt) * self.scale).masked_fill(nkeep, float("-inf"))
            ctx = torch.matmul(torch.softmax(sc, -1).to(self.dt), vb).reshape(1, self.nh * self.hd)
            x = x + ctx @ ly["wo"].T
            h2 = _rmsnorm(x, ly["post_ln"], self.eps)
            gate, up = (h2 @ ly["wgu"].T).split([self.inter, self.inter], dim=-1)
            x = x + (F.silu(gate) * up) @ ly["wd"].T
        x = _rmsnorm(x, self.final_norm, self.eps)
        return (x @ self.lm_head.T).float()

    def _bucket_for(self, pos: int) -> int:
        return next(b for b in self.buckets if b > pos)

    def _capture(self, bound: int) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                lg = self._decode_forward(bound)
                tk = lg.argmax(-1)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            lg = self._decode_forward(bound)
            tk = lg.argmax(-1)  # greedy argmax on-GPU, pipelined in the graph
        self._graphs[bound], self._logits[bound], self._tok[bound] = graph, lg, tk

    def _replay(self, pos: int) -> int:
        bound = self._bucket_for(pos)
        if bound not in self._graphs:
            self._capture(bound)
        self._graphs[bound].replay()
        return bound

    def decode(self, token: int, pos: int) -> torch.Tensor:
        self.tok_buf.fill_(int(token))
        self.pos_buf.fill_(int(pos))
        return self._logits[self._replay(pos)]

    def decode_argmax(self, token: int, pos: int) -> int:
        """Greedy decode reading the in-graph argmax (no eager argmax over the vocab)."""
        self.tok_buf.fill_(int(token))
        self.pos_buf.fill_(int(pos))
        return int(self._tok[self._replay(pos)].item())

    def decode_gpu(self, tok_t: torch.Tensor, pos: int) -> torch.Tensor:
        """Decode from a GPU-resident token (no CPU sync); returns next-token argmax [1] on GPU.

        Used by the speculative draft loop so the K draft replays pipeline back-to-back without a
        GPU->CPU flush between them."""
        self.tok_buf.copy_(tok_t.view(1))
        self.pos_buf.fill_(int(pos))
        return self._logits[self._replay(pos)].argmax(-1)

    def decode_gpu_argmax(self, tok_t: torch.Tensor, pos: int) -> torch.Tensor:
        """Greedy decode from a GPU-resident token, reading the in-graph argmax (no eager vocab
        argmax, no CPU sync). Returns the next token as a [1] GPU tensor, cloned so the next replay
        of this bucket does not clobber it. Lets the chunked decoder pipeline K replays back-to-back
        on the stream and sync once."""
        self.tok_buf.copy_(tok_t.view(1))
        self.pos_buf.fill_(int(pos))
        return self._tok[self._replay(pos)].clone()

    def _verify_forward(self) -> torch.Tensor:
        k1 = self.vtok.shape[0]
        pos = self.vpos0 + torch.arange(k1, device=self.dev)
        cos, sin = self._cos_sin(pos)
        x = self.embed[self.vtok].to(self.dt)
        keep = (self.arange[None, :] <= pos[:, None]).view(1, k1, self.max_len)
        for li, ly in enumerate(self.layers):
            h = _rmsnorm(x, ly["in_ln"], self.eps)
            q, k, v = (h @ ly["wqkv"].T).split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
            q = q.view(k1, self.nh, self.hd).transpose(0, 1)
            k = k.view(k1, self.nkv, self.hd).transpose(0, 1)
            v = v.view(k1, self.nkv, self.hd).transpose(0, 1)
            q = self._rope(_rmsnorm(q, ly["qn"], self.eps), cos, sin)
            k = self._rope(_rmsnorm(k, ly["kn"], self.eps), cos, sin)
            self.kc[li].index_copy_(0, pos, k.transpose(0, 1))
            self.vc[li].index_copy_(0, pos, v.transpose(0, 1))
            kk = self.kc[li].permute(1, 0, 2).repeat_interleave(self.groups, 0)
            vv = self.vc[li].permute(1, 0, 2).repeat_interleave(self.groups, 0)
            ctx = F.scaled_dot_product_attention(q, kk, vv, attn_mask=keep, scale=self.scale)
            x = x + ctx.transpose(0, 1).reshape(k1, self.nh * self.hd) @ ly["wo"].T
            h2 = _rmsnorm(x, ly["post_ln"], self.eps)
            gate, up = (h2 @ ly["wgu"].T).split([self.inter, self.inter], dim=-1)
            x = x + (F.silu(gate) * up) @ ly["wd"].T
        return (_rmsnorm(x, self.final_norm, self.eps) @ self.lm_head.T).float()

    def verify(self, tok_t: torch.Tensor, pos_start: int) -> torch.Tensor:
        """Score K+1 GPU tokens at [pos_start..pos_start+K]; logits [K+1, vocab]."""
        if self.verify_graph is None:
            self.vtok = torch.zeros(tok_t.shape[0], dtype=torch.long, device=self.dev)
            self.vpos0 = torch.zeros(1, dtype=torch.long, device=self.dev)
            self.vtok.copy_(tok_t)
            self.vpos0.fill_(int(pos_start))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._vlogits = self._verify_forward()
            torch.cuda.current_stream().wait_stream(stream)
            self.verify_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.verify_graph):
                self._vlogits = self._verify_forward()
        self.vtok.copy_(tok_t)
        self.vpos0.fill_(int(pos_start))
        self.verify_graph.replay()
        return self._vlogits


class CudaGraphQwen3Executor(TransformersTextExecutor):
    """Single-sequence, CUDA-graph Qwen3 executor. Beats vLLM single-stream by killing launch
    overhead. Falls back to nothing — it requires a Qwen3-family model and CUDA."""

    name = "cudagraph-qwen3"
    supports_prefix_cache = False
    uses_paged_kv = False  # owns its own static KV; skip RuntimeCore's paged-KV bookkeeping

    def __init__(self, config: TransformersGenerationConfig | None = None) -> None:
        super().__init__(config)
        if not hasattr(self.model.model.layers[0].self_attn, "q_norm"):
            raise RuntimeError(
                "CudaGraphQwen3Executor requires a Qwen3-family model (q_norm/k_norm)"
            )
        if self._model_device().type != "cuda":
            raise RuntimeError("CudaGraphQwen3Executor requires CUDA")
        self._max_len = int(self.config.graph_max_len)
        self._decode_chunk = max(1, int(getattr(self.config, "decode_chunk", 1)))
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
        # First token after prefill is already cached on the state; emit it directly.
        if state.pending_token_id is not None:
            token_id = int(state.pending_token_id)
            state.pending_token_id = None
            return self._append_token_result(sequence, state, token_id)
        if not state.generated_token_ids:
            state.finished = True
            self._states.pop(sequence.seq_id, None)
            return SampleResult(token_id=self._eos_token_id(), text="", finished=True)
        position = state.prompt_token_count + len(state.generated_token_ids)
        if position >= self._max_len:
            raise RuntimeError("decode position exceeds graph_max_len; raise SOLORT_GRAPH_MAX_LEN")

        if not self._is_greedy(sequence):
            with self._torch.no_grad():
                logits = self._runner.decode(state.generated_token_ids[-1], position)
                token_id = self._sample_token(logits, sequence)
            return self._append_token_result(sequence, state, token_id)

        # Greedy: read the in-graph argmax (1 elem, not an eager argmax over the whole vocab) and,
        # when decode_chunk>1, pipeline K replays back-to-back on the GPU stream with a single CPU
        # sync so the per-token RuntimeCore/executor Python amortizes over K tokens.
        remaining = sequence.max_new_tokens - len(state.generated_token_ids)
        k = max(1, min(self._decode_chunk, remaining, self._max_len - position))
        with self._torch.no_grad():
            if k == 1:
                ids = [self._runner.decode_argmax(state.generated_token_ids[-1], position)]
            else:
                cur = self._torch.tensor(
                    [state.generated_token_ids[-1]],
                    dtype=self._torch.long,
                    device=self._model_device(),
                )
                toks = []
                for i in range(k):
                    cur = self._runner.decode_gpu_argmax(cur, position + i)
                    toks.append(cur)
                ids = self._torch.cat(toks).tolist()  # one GPU->CPU sync for the whole chunk
        results: list[SampleResult] = []
        for token_id in ids:
            results.append(self._append_token_result(sequence, state, int(token_id)))
            if state.finished:  # stop tokens past the boundary were speculative; drop them
                break
        return results

    def _is_greedy(self, sequence: Sequence) -> bool:
        temp = float(sequence.metadata.get("temperature", self.config.default_temperature))
        rep = float(
            sequence.metadata.get("repetition_penalty", self.config.default_repetition_penalty)
        )
        return temp <= 0 and rep == 1.0


class SpecCudaGraphQwen3Executor(CudaGraphQwen3Executor):
    """CUDA-graph executor with exact greedy speculative decoding (0.6B draft -> 4B target).

    Both models run as CUDA graphs, so the draft is cheap and the target verifies K+1 tokens in one
    graphed forward. Unlike the HF bridge (where spec was a 2.3x loss), this beats the target-only
    cudagraph path and edges past vLLM on 4B. Greedy + repetition_penalty==1.0 only (exactness);
    other requests fall back to single-token decode.
    """

    name = "cudagraph-qwen3-spec"

    def __init__(self, config: TransformersGenerationConfig | None = None) -> None:
        super().__init__(config)
        self._spec_k = max(1, int(self.config.speculative_tokens))
        from transformers import AutoModelForCausalLM

        kwargs = self._model_kwargs()
        try:
            draft = AutoModelForCausalLM.from_pretrained(
                self.config.speculative_draft_model_id,
                trust_remote_code=self.config.trust_remote_code,
                **kwargs,
            )
        except TypeError:
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            draft = AutoModelForCausalLM.from_pretrained(
                self.config.speculative_draft_model_id,
                trust_remote_code=self.config.trust_remote_code,
                **kwargs,
            )
        draft.eval()
        if not hasattr(draft.model.layers[0].self_attn, "q_norm"):
            raise RuntimeError("speculative draft must be a Qwen3-family model")
        self._draft = _GraphQwen3Runner(draft, self._max_len)

    def forward_prefill(self, batch: Batch) -> None:
        super().forward_prefill(batch)
        sequence = batch.seqs[0]
        state = self._states.get(sequence.seq_id)
        if state is None or state.pending_token_id is None:
            return
        if getattr(state, "_draft_prefilled", False):
            return
        prompt = sequence.input_ids[: sequence.num_prompt_tokens]
        ids = self._torch.tensor(prompt, dtype=self._torch.long, device=self._model_device())
        with self._torch.no_grad():
            self._draft.prefill(ids)
        state._draft_prefilled = True

    def forward_decode(self, batch: Batch) -> SampleResult | list[SampleResult]:
        sequence = batch.seqs[0]
        state = self._ensure_state(sequence)
        if state.finished:
            return SampleResult(token_id=self._eos_token_id(), text="", finished=True)
        if state.pending_token_id is not None or not state.generated_token_ids:
            return super().forward_decode(batch)
        temperature = float(sequence.metadata.get("temperature", self.config.default_temperature))
        rep = float(
            sequence.metadata.get("repetition_penalty", self.config.default_repetition_penalty)
        )
        position = state.prompt_token_count + len(state.generated_token_ids)
        # Exact greedy spec only; otherwise single-token decode (handles sampling + rep penalty).
        if temperature > 0 or rep != 1.0 or position + self._spec_k + 2 >= self._max_len:
            return super().forward_decode(batch)
        new_tokens = self._spec_round(state.generated_token_ids[-1], position)
        results: list[SampleResult] = []
        for token_id in new_tokens:
            results.append(self._append_token_result(sequence, state, token_id))
            if state.finished:
                break
        return results

    def _spec_round(self, last_token: int, n: int) -> list[int]:
        torch = self._torch
        k = self._spec_k
        with torch.no_grad():
            t_gpu = torch.tensor([last_token], dtype=torch.long, device=self._model_device())
            drafts = []
            cur, pos = t_gpu, n
            for _ in range(k):
                cur = self._draft.decode_gpu(cur, pos)
                drafts.append(cur)
                pos += 1
            draft_vec = torch.cat(drafts)  # [k]
            tg = self._runner.verify(torch.cat([t_gpu, draft_vec]), n).argmax(-1)  # [k+1]
            a = int((tg[:k] == draft_vec).cumprod(0).sum().item())
            if a == k:
                out = torch.cat([draft_vec, tg[k : k + 1]]).tolist()
                self._draft.decode_gpu(draft_vec[k - 1 : k], n + k)  # dK draft KV continuity
            else:
                out = torch.cat([draft_vec[:a], tg[a : a + 1]]).tolist()
        return out
