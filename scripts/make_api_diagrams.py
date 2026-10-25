from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "api_test_results.json"
OUT = ROOT / "docs" / "images" / "api"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})

C_BG = "#f7f9fc"
C_BLUE = "#2f6db3"
C_BLUE_L = "#e8f0fa"
C_GREEN = "#3a8f5a"
C_GREEN_L = "#e9f5ee"
C_ORANGE = "#d9822b"
C_ORANGE_L = "#fdf1e4"
C_RED = "#c44e52"
C_GREY = "#5b6573"


def box(ax, x, y, w, h, title, body="", fc="#ffffff", ec=C_BLUE, lw=1.4,
        title_size=10.5, body_size=9.0):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=lw, edgecolor=ec, facecolor=fc, transform=ax.transAxes,
    )
    ax.add_patch(patch)
    if body:
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=C_GREY,
                transform=ax.transAxes)
        ax.text(x + w / 2, y + h * 0.28, body, ha="center", va="center",
                fontsize=body_size, color=C_GREY, transform=ax.transAxes)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=C_GREY,
                transform=ax.transAxes)


def arrow(ax, x1, y1, x2, y2, color=C_BLUE, style="-|>", lw=1.5, ls="-", rad=0.0):
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
        linewidth=lw, color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}",
        transform=ax.transAxes,
    )
    ax.add_patch(patch)


def flow_diagram(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(C_BG)
    fig.patch.set_facecolor(C_BG)

    ax.text(0.5, 0.955, "Рикошет: сценарии тестирования API", ha="center",
            va="center", fontsize=20, fontweight="bold", color="#1f2a37",
            transform=ax.transAxes)
    ax.text(0.5, 0.915, "Онлайн-аварии → FastAPI → precheck → JSON-«дело» для ИИ-агента",
            ha="center", va="center", fontsize=11, color=C_GREY,
            transform=ax.transAxes)

    ax.text(0.025, 0.855, "Входные сценарии", fontsize=11, fontweight="bold",
            color=C_BLUE, transform=ax.transAxes)
    scenarios = [
        ("S1", "Известная platform-авария", "ожидаем кандидатов и риск MEDIUM"),
        ("S2", "10 аварий, 3 региона", "ожидаем BLAST_RADIUS и риск HIGH"),
        ("S3", "Неизвестная business-авария", "ожидаем пустых кандидатов и LOW"),
        ("S4", "Авария в окне обслуживания", "ожидаем MAINTENANCE_OVERLAP"),
        ("S5", "Ошибки 422 / 404", "проверяем контракт и коды ответов"),
    ]
    ys = [0.775, 0.655, 0.535, 0.415, 0.295]
    for (code, name, expected), y in zip(scenarios, ys):
        box(ax, 0.025, y, 0.235, 0.095, f"{code}  {name}", expected,
            fc="#ffffff", ec=C_BLUE, title_size=9.5, body_size=8.2)

    box(ax, 0.305, 0.700, 0.355, 0.145,
        "FastAPI / uvicorn   127.0.0.1:8077",
        "POST /api/v1/precheck/alarm\nPOST /api/v1/precheck/incident",
        fc=C_BLUE_L, ec=C_BLUE, title_size=11, body_size=9.2)
    box(ax, 0.305, 0.405, 0.355, 0.225,
        "assemble_case()",
        "последний граф пар  ·  окна обслуживания\n"
        "полнота обогащения  ·  похожие инциденты\n"
        "runbook  ·  blast radius  ·  risk score",
        fc="#ffffff", ec=C_BLUE, title_size=12, body_size=9.2)

    components = [
        ("258 пар", 0.305, 0.305, 0.080),
        ("81 отказ", 0.395, 0.305, 0.075),
        ("окна", 0.480, 0.305, 0.065),
        ("runbook", 0.555, 0.305, 0.085),
        ("risk", 0.650, 0.305, 0.060),
    ]
    for label, x, y, w in components:
        box(ax, x, y, w, 0.060, label, fc=C_GREEN_L, ec=C_GREEN, title_size=8.8)
        arrow(ax, x + w / 2, y + 0.060, x + w / 2, 0.405, color=C_GREEN, lw=1.0)

    box(ax, 0.720, 0.315, 0.255, 0.530,
        "Ответ 200 JSON",
        "alarms_total\n"
        "alarms_outside_maintenance\n"
        "maintenance\n"
        "enrichment\n"
        "root_cause_candidates\n"
        "similar_incidents\n"
        "runbook\n"
        "blast\n"
        "risk",
        fc=C_GREEN_L, ec=C_GREEN, title_size=13, body_size=9.3)
    box(ax, 0.720, 0.190, 0.255, 0.085,
        "Коды ошибок",
        "422 — невалидное тело\n404 — неизвестный job_id",
        fc="#fdecec", ec=C_RED, title_size=9.2, body_size=8.6)

    arrow(ax, 0.260, 0.640, 0.305, 0.720, color=C_BLUE, lw=1.8)
    ax.text(0.278, 0.705, "HTTPS/JSON", fontsize=8.2, color=C_BLUE,
            ha="center", va="center", transform=ax.transAxes, rotation=38)
    arrow(ax, 0.480, 0.700, 0.480, 0.630, color=C_BLUE, lw=1.8)
    arrow(ax, 0.660, 0.515, 0.720, 0.515, color=C_GREEN, lw=1.8)
    ax.text(0.690, 0.540, "case JSON", fontsize=8.5, color=C_GREEN,
            ha="center", va="center", transform=ax.transAxes)

    bench = results["benchmark"]
    seq = bench["sequential_100"]
    conc = bench["concurrent_8x100"]
    banner = (
        f"100 вызовов: p50 {seq['median_ms']:.1f} мс · p95 {seq['p95_ms']:.1f} мс · "
        f"{seq['rps']:.0f} req/s     |     8 потоков: {conc['rps']:.0f} req/s · "
        f"p95 {conc['p95_ms']:.1f} мс"
    )
    box(ax, 0.025, 0.035, 0.950, 0.095, "Замер скорости", banner,
        fc=C_ORANGE_L, ec=C_ORANGE, title_size=11, body_size=9.6)

    fig.savefig(OUT / "api_test_flow.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def sequence_diagram(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(C_BG)
    fig.patch.set_facecolor(C_BG)

    ax.text(0.5, 0.955, "Последовательность precheck-запроса", ha="center",
            va="center", fontsize=19, fontweight="bold", color="#1f2a37",
            transform=ax.transAxes)
    ax.text(0.5, 0.915, "Одна авария или инцидент → синхронный ответ; граф уже загружен в память",
            ha="center", va="center", fontsize=10.5, color=C_GREY,
            transform=ax.transAxes)

    lifelines = [
        ("Клиент / мониторинг", 0.10, C_BLUE),
        ("FastAPI route", 0.35, C_BLUE),
        ("assemble_case()", 0.62, C_GREEN),
        ("AppState (память)", 0.88, C_ORANGE),
    ]
    for title, x, color in lifelines:
        box(ax, x - 0.095, 0.835, 0.190, 0.055, title, fc="#ffffff", ec=color,
            title_size=10.2)
        ax.plot([x, x], [0.835, 0.110], color=color, lw=1.0, ls=(0, (4, 4)),
                transform=ax.transAxes, alpha=0.55)

    arrow(ax, 0.10, 0.755, 0.35, 0.755, color=C_BLUE, lw=1.6)
    ax.text(0.225, 0.785, "1. POST /api/v1/precheck/alarm\n{ts, integration, system, alert_group, node, severity}",
            ha="center", va="bottom", fontsize=8.4, color=C_GREY, transform=ax.transAxes)

    arrow(ax, 0.35, 0.675, 0.62, 0.675, color=C_BLUE, lw=1.6)
    ax.text(0.485, 0.705, "2. assemble_case(incident, alarms_history, pairs,\nmaintenance, incidents, runbooks)",
            ha="center", va="bottom", fontsize=8.4, color=C_GREY, transform=ax.transAxes)

    arrow(ax, 0.62, 0.595, 0.88, 0.595, color=C_ORANGE, lw=1.6)
    ax.text(0.75, 0.625, "3. читает граф, окна, историю, runbook-и\n(без пересчёта корреляций)",
            ha="center", va="bottom", fontsize=8.4, color=C_GREY, transform=ax.transAxes)

    arrow(ax, 0.88, 0.515, 0.62, 0.515, color=C_ORANGE, lw=1.4, ls="--")
    ax.text(0.75, 0.545, "4. срез данных", ha="center", va="bottom",
            fontsize=8.4, color=C_GREY, transform=ax.transAxes)

    box(ax, 0.440, 0.385, 0.360, 0.078,
        "5. фильтрация окон · обогащение\nкандидаты · blast · risk",
        fc=C_GREEN_L, ec=C_GREEN, title_size=9.2)

    arrow(ax, 0.62, 0.310, 0.35, 0.310, color=C_GREEN, lw=1.6, ls="--")
    ax.text(0.485, 0.338, "6. case JSON", ha="center", va="bottom",
            fontsize=8.4, color=C_GREY, transform=ax.transAxes)

    arrow(ax, 0.35, 0.235, 0.10, 0.235, color=C_GREEN, lw=1.6, ls="--")
    ax.text(0.225, 0.263, "7. 200 OK — case JSON", ha="center", va="bottom",
            fontsize=8.4, color=C_GREY, transform=ax.transAxes)

    seq = results["benchmark"]["sequential_100"]
    ax.text(0.5, 0.140,
            f"Факт: p50 {seq['median_ms']:.1f} мс · p95 {seq['p95_ms']:.1f} мс · "
            f"{seq['rps']:.0f} req/s на одном процессе.\n"
            "Ответ детерминирован: повторная отправка той же аварии даёт тот же case JSON.",
            ha="center", va="center", fontsize=10, color=C_GREY,
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.6", fc="#ffffff",
                                              ec=C_ORANGE, lw=1.2))

    fig.savefig(OUT / "precheck_sequence.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def latency_chart(results: dict) -> None:
    bench = results["benchmark"]
    modes = [
        ("keep-alive\n1 поток", bench["sequential_100"], C_BLUE),
        ("keep-alive\n8 потоков", bench["concurrent_8x100"], C_GREEN),
        ("новое\nсоединение", bench["cold_connection_100"], C_ORANGE),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor(C_BG)
    for ax in axes:
        ax.set_facecolor(C_BG)

    x = list(range(len(modes)))
    width = 0.26
    for idx, metric in enumerate(("median_ms", "p95_ms", "p99_ms")):
        vals = [m[1][metric] for m in modes]
        axes[0].bar([i + (idx - 1) * width for i in x], vals, width=width,
                    label={"median_ms": "p50", "p95_ms": "p95", "p99_ms": "p99"}[metric],
                    color=[C_BLUE, C_GREEN, C_ORANGE][idx], alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m[0] for m in modes])
    axes[0].set_ylabel("мс")
    axes[0].set_title("Латентность 100 вызовов precheck")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    vals = [m[1]["rps"] for m in modes]
    bars = axes[1].bar([m[0] for m in modes], vals,
                       color=[m[2] for m in modes], alpha=0.9)
    for bar, value in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}",
                     ha="center", va="bottom", fontsize=10, fontweight="bold", color=C_GREY)
    axes[1].set_ylabel("запросов/с")
    axes[1].set_title("Пропускная способность precheck")
    axes[1].set_ylim(0, max(vals) * 1.22)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Скорость онлайн-проверки аварий", fontsize=15, fontweight="bold",
                 color="#1f2a37")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "latency_benchmark.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def capacity_chart(results: dict) -> None:
    bench = results["benchmark"]
    rows = [
        ("1 млн/сутки, среднее", 1_000_000 / 86400, C_BLUE),
        ("1 млн/сутки, пик ×5", 5 * 1_000_000 / 86400, C_ORANGE),
        ("1 млн/сутки, пик ×10", 10 * 1_000_000 / 86400, C_RED),
        ("Замер: 1 поток", bench["sequential_100"]["rps"], C_GREEN),
        ("Замер: 8 потоков", bench["concurrent_8x100"]["rps"], C_GREEN),
        ("Замер: новое соединение", bench["cold_connection_100"]["rps"], C_ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(12, 5.6))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], alpha=0.9)
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 1.5, bar.get_y() + bar.get_height() / 2, f"{value:.1f}",
                va="center", fontsize=10, fontweight="bold", color=C_GREY)
    ax.axvline(115.7, color=C_RED, ls=":", lw=1.1)
    ax.set_xlabel("запросов в секунду")
    ax.set_title("Ёмкость precheck: 1 млн аварий в сутки", fontweight="bold", fontsize=15)
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0, max(values) * 1.18)
    ax.text(0.99, 0.02,
            "Только precheck без инференса модели.\n"
            "1 uvicorn worker · Ryzen 5 7535HS · 13 ГБ RAM.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.4,
            color=C_GREY)
    fig.tight_layout()
    fig.savefig(OUT / "capacity_model.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    results = json.loads(DATA.read_text(encoding="utf-8"))
    flow_diagram(results)
    sequence_diagram(results)
    latency_chart(results)
    capacity_chart(results)
    print("written", OUT)
    for path in sorted(OUT.glob("*.png")):
        print(path.name, path.stat().st_size)


if __name__ == "__main__":
    main()
