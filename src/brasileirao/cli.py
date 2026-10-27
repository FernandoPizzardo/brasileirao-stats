import argparse
from pathlib import Path

from brasileirao.backtest import backtest, score
from brasileirao.config import get_settings
from brasileirao.report import build_report
from brasileirao.service import build_context, load_matches, refresh_cache, simulate, zones_from
from brasileirao.simulate import summarize


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="brasileirao")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="refresh the current season from the API into the local cache")
    sub.add_parser("simulate", help="print final-position probabilities for each team")
    sub.add_parser("backtest", help="score the model against a frequency baseline on past matchdays")
    report = sub.add_parser("report", help="write a static HTML report")
    report.add_argument("--out", default="site", help="output directory")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.command == "fetch":
        refresh_cache(settings)
        print("Cache refreshed.")
        return
    if args.command == "report":
        build_report(settings, Path(args.out))
        print(f"Report written to {args.out}/index.html")
        return
    if args.command == "backtest":
        df = backtest(load_matches(settings), settings.half_life_days, settings.ridge)
        print(f"Matches scored: {len(df)}")
        for name, prefix in (("Model", "model"), ("Baseline", "base")):
            s = score(df, prefix)
            print(f"{name:<9} log loss {s['log_loss']:.4f}  brier {s['brier']:.4f}  accuracy {s['accuracy']:.3f}")
        return

    ctx = build_context(settings)
    result = simulate(ctx, settings)
    summary = summarize(result, ctx.table, zones_from(settings))
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
