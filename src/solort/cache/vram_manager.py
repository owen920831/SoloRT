"""Conservative VRAM budget policy for the mock runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VRAMBudgetManager:
    total_bytes: int = 16 * 1024**3
    weight_bytes: int = 0
    workspace_bytes: int = 512 * 1024**2
    graph_bytes: int = 256 * 1024**2
    safety_fraction: float = 0.90

    def reserved_bytes(self) -> int:
        return self.weight_bytes + self.workspace_bytes + self.graph_bytes

    def available_for_kv(self) -> int:
        safe_total = int(self.total_bytes * self.safety_fraction)
        return max(0, safe_total - self.reserved_bytes())

    def should_accept_background(self, estimated_extra_bytes: int) -> bool:
        return estimated_extra_bytes <= self.available_for_kv()

    def snapshot(self) -> dict[str, int | float | bool]:
        available = self.available_for_kv()
        return {
            "total_bytes": self.total_bytes,
            "reserved_bytes": self.reserved_bytes(),
            "available_for_kv_bytes": available,
            "safety_fraction": self.safety_fraction,
            "background_allowed": available > 0,
        }
