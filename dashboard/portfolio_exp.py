import json
from pathlib import Path


def load_exp_metrics(root: Path):
    reports_dir = root / "reports"
    out = {}
    exp_single = reports_dir / "ftmo_usdjpy_exp_v2_full_pipeline.json"
    exp_multi = reports_dir / "ftmo_multi_usdjpy_exp_v2_full_pipeline.json"
    for name, path in [("exp_v2", exp_single), ("exp_v2_multi", exp_multi)]:
        if path.exists():
            try:
                out[name] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                out[name] = None
        else:
            out[name] = None
    # optimizer results
    opt_path = reports_dir / "optimize_usdjpy_exp_v2_risk.json"
    if opt_path.exists():
        try:
            out["opt_top"] = json.loads(opt_path.read_text(encoding="utf-8"))
        except Exception:
            out["opt_top"] = None
    else:
        out["opt_top"] = None
    return out
