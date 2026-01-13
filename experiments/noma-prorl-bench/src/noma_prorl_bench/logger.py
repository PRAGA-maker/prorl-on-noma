from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class JSONLLogger:
    outdir: str

    def __post_init__(self) -> None:
        os.makedirs(self.outdir, exist_ok=True)
        self.metrics_path = os.path.join(self.outdir, "metrics.jsonl")
        self.events_path = os.path.join(self.outdir, "events.jsonl")

    def write_config(self, cfg: Dict[str, Any]) -> None:
        with open(os.path.join(self.outdir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)

    def log_metrics(self, d: Dict[str, Any]) -> None:
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(d) + "\n")

    def log_event(self, d: Dict[str, Any]) -> None:
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(d) + "\n")
