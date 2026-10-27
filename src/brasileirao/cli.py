import argparse
from pathlib import Path

from brasileirao.backtest import backtest, score
from brasileirao.config import get_settings
from brasileirao.formatting import pt_num
from brasileirao.report import build_report
from brasileirao.service import build_context, load_matches, refresh_cache, simulate, zones_from
from brasileirao.simulate import summarize


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="brasileirao", description="Simulador do Brasileirão")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="atualiza a temporada atual no cache local, via API")
    sub.add_parser("simulate", help="mostra as probabilidades de posição final de cada time")
    sub.add_parser("backtest", help="avalia o modelo em rodadas passadas, contra uma referência simples")
    report = sub.add_parser("report", help="gera um relatório HTML estático")
    report.add_argument("--out", default="site", help="pasta de saída")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.command == "fetch":
        refresh_cache(settings)
        print("Cache atualizado.")
        return
    if args.command == "report":
        build_report(settings, Path(args.out))
        print(f"Relatório gerado em {args.out}/index.html")
        return
    if args.command == "backtest":
        df = backtest(load_matches(settings), settings.half_life_days, settings.ridge)
        print(f"Partidas avaliadas: {len(df)}")
        for name, prefix in (("Modelo", "model"), ("Referência", "base")):
            s = score(df, prefix)
            print(
                f"{name:<11} log loss {pt_num(s['log_loss'], 4)}  "
                f"brier {pt_num(s['brier'], 4)}  acurácia {pt_num(s['accuracy'] * 100, 1)}%"
            )
        return

    ctx = build_context(settings)
    result = simulate(ctx, settings)
    summary = summarize(result, ctx.table, zones_from(settings)).rename(
        columns={
            "team": "Time",
            "tla": "Sigla",
            "exp_points": "Pontos esperados",
            "exp_position": "Posição esperada",
            "title": "Título",
            "libertadores": "Libertadores",
            "sula": "Sul-Americana",
            "relegation": "Rebaixamento",
        }
    )
    print(summary.to_string(index=False, float_format=lambda v: pt_num(v, 3)))


if __name__ == "__main__":
    main()
