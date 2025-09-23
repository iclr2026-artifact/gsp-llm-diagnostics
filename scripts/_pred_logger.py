import json, os, time
from pathlib import Path

class PredLogger:
    def __init__(self, out_dir: str):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = Path(out_dir) / f"preds_{ts}.jsonl"
        self.f = open(self.path, "w", encoding="utf-8")
    def log(self, record: dict):
        self.f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.f.flush()
    def close(self):
        self.f.close()
