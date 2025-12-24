import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
DEFAULT_REPORT = ROOT / "reports" / "fastpass_research_report.md"
FASTPASS_CONFIG_JSON = ROOT / "configs" / "fastpass_symbol_configs.json"
SHADOW_LOG_DEFAULT = ROOT / "logs" / "shadow_fastpass_usdjpy_core.csv"


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_shadow_log(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["entry_time", "exit_time", "entry_date"])
        return df
    except Exception:
        return None


def summarize_shadow(df: pd.DataFrame) -> dict:
    n = len(df)
    wins = int((df["pnl"] > 0).sum()) if "pnl" in df else 0
    losses = int((df["pnl"] < 0).sum()) if "pnl" in df else 0
    flats = int((df["pnl"] == 0).sum()) if "pnl" in df else 0
    win_rate = wins / n if n else 0.0
    start_equity = df.get("equity_before", pd.Series([10_000.0])).iloc[0]
    end_equity = df.get("equity_after", pd.Series([start_equity])).iloc[-1]
    total_return = (end_equity / start_equity - 1.0) if start_equity else 0.0
    equity_curve = df["equity_after"] if "equity_after" in df else start_equity + df["pnl"].cumsum()
    max_equity = equity_curve.cummax()
    dd = ((max_equity - equity_curve) / max_equity.replace(0, pd.NA)).max(skipna=True)
    max_dd = float(dd or 0.0)
    trades_today = 0
    if "entry_date" in df:
        trades_today = int((df["entry_date"].dt.date == pd.Timestamp("now").date()).sum())
    trades_by_strategy = df["profile_name"].value_counts().to_dict() if "profile_name" in df else {}
    target_equity = start_equity * 1.07
    loss_equity = start_equity * 0.94
    hit_target = (equity_curve >= target_equity).any()
    hit_loss = (equity_curve <= loss_equity).any()
    if hit_target and hit_loss:
        first_target = equity_curve[equity_curve >= target_equity].index[0]
        first_loss = equity_curve[equity_curve <= loss_equity].index[0]
        challenge_status = "target_first" if first_target < first_loss else "loss_first"
    elif hit_target:
        challenge_status = "target_hit"
    elif hit_loss:
        challenge_status = "max_loss_breached"
    else:
        challenge_status = "in_progress"
    return {
        "trades": n,
        "wins": wins,
        "losses": losses,
        "flats": flats,
        "win_rate": win_rate,
        "start_equity": float(start_equity),
        "end_equity": float(end_equity),
        "total_return": float(total_return),
        "max_drawdown": max_dd,
        "trades_today": trades_today,
        "trades_by_strategy": trades_by_strategy,
        "challenge_status": challenge_status,
    }


def maybe_run(script: str) -> Optional[str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(SRC_DIR))
    try:
        out = subprocess.check_output([env.get("PYTHON", "python"), script], cwd=ROOT, env=env, stderr=subprocess.STDOUT, text=True, timeout=120)
        return out
    except Exception as exc:
        return f"Failed to run {script}: {exc}"


def main() -> None:
    lines = []
    lines.append("# FastPass Research Snapshot\n")

    # Config leaderboard
    cfg = load_json(FASTPASS_CONFIG_JSON)
    lines.append("## Per-symbol best configs (from configs/fastpass_symbol_configs.json)")
    if cfg:
        for sym, data in cfg.items():
            metrics = data.get("metrics", {})
            lines.append(f"- {sym}: London {data.get('london')} | Liquidity {data.get('liquidity')} | pass_rate={metrics.get('pass_rate'):.2%} | avg_ret={metrics.get('avg_return'):.2%} | max_dd={metrics.get('avg_max_dd'):.2%} | trades/eval={metrics.get('avg_trades')}")
    else:
        lines.append("- No config export found.")
    lines.append("")

    # Shadow log snapshot
    shadow_path = Path(os.getenv("OMEGAFX_SHADOW_LOG", SHADOW_LOG_DEFAULT))
    shadow_df = load_shadow_log(shadow_path)
    lines.append(f"## Shadow log snapshot ({shadow_path})")
    if shadow_df is not None and not shadow_df.empty:
        ss = summarize_shadow(shadow_df)
        lines.append(
            f"- Trades: {ss['trades']} (wins/losses/flats {ss['wins']}/{ss['losses']}/{ss['flats']}, win_rate {ss['win_rate']:.2%})"
        )
        lines.append(
            f"- Equity: start {ss['start_equity']:.2f} → end {ss['end_equity']:.2f} (return {ss['total_return']:.2%}), max DD {ss['max_drawdown']:.2%}"
        )
        lines.append(f"- Challenge status: {ss['challenge_status']}")
        lines.append(f"- Trades today: {ss['trades_today']}")
        lines.append(f"- Trades by strategy: {ss['trades_by_strategy']}")
    else:
        lines.append("- No shadow log found.")
    lines.append("")

    # Optional eval runs
    if os.getenv("FASTPASS_RUN_EVALS", "0") == "1":
        lines.append("## Fresh eval runs")
        core_script = ROOT / "scripts" / "demo_fastpass_usdjpy_v3_mt5_long.py"
        if core_script.exists():
            out = maybe_run(str(core_script))
            lines.append("### USDJPY FastPass V3 (30d random windows)")
            lines.append("```\n" + (out or "no output") + "\n```")
        core30 = ROOT / "scripts" / "run_profile_summary_momentum_m5_mt5.py"
        if core30.exists():
            out = maybe_run(str(core30))
            lines.append("### M5 Momentum summary")
            lines.append("```\n" + (out or "no output") + "\n```")
        v3_fast = ROOT / "scripts" / "demo_fastpass_usdjpy_v3_fastwindows_mt5.py"
        if v3_fast.exists():
            out = maybe_run(str(v3_fast))
            lines.append("### USDJPY FastPass V3 (30d/60d windows)")
            lines.append("```\n" + (out or "no output") + "\n```")
    else:
        lines.append("## Fresh eval runs")
        lines.append("- Skipped (set FASTPASS_RUN_EVALS=1 to run).")
    lines.append("")

    report_path = Path(os.getenv("FASTPASS_REPORT_OUT", DEFAULT_REPORT))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    report_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
