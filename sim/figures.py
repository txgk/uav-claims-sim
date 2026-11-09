"""Рисунки 1–4 статьи -> figures/*.png (300 dpi) и *.pdf.

    python3 figures.py
"""
import os

import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter

from model import Config, credibility, half_life
from stats import by, ghost_lat, load, mean, taken
from style import LABEL, LOSS_COLORS, RULES, STYLE, decimal_comma, plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
RS = [1000.0, 2000.0, 3000.0, 5000.0, 7000.0, 10000.0, 15000.0, 20000.0]
RTICKS = [1, 2, 3, 5, 7, 10, 15, 20]
NEAR = [1000.0, 2000.0, 3000.0]          # редкая связь (рис. 4, а)
FAR = [10000.0, 15000.0, 20000.0]        # частая связь (рис. 4, б)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    decimal_comma(fig)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


def r_axis(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(RTICKS))
    ax.set_xticklabels([str(t) for t in RTICKS])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.9, 22)
    ax.set_xlabel("радиус связи $R$, км")


def panel(ax, letter, x=0.02):
    ax.text(x, 0.98, letter, transform=ax.transAxes, va="top", ha="left", fontsize=12)


def fig1():
    """Апостериорная вероятность работоспособности молчащего агента, формула (1)."""
    lam = 1 / (24 * 60)                                   # средняя наработка на отказ 24 ч, 1/мин
    tau = np.linspace(0, 180, 721)
    fig, ax = plt.subplots(figsize=(5.6, 2.3))
    for c, m in zip(["#0d366b", "#1c5cab", "#3987e5", "#6da7ec"], [2, 7, 20, 60]):
        mu = 1 / m
        ax.plot(tau, credibility(tau, lam, mu), color=c, lw=2.0)
        th = half_life(lam, mu)
        if th < 180:
            ax.plot([th], [0.5], "o", color=c, ms=5, zorder=5)
            ax.annotate(f"{m} мин", (th, 0.5), xytext=(6, 3), textcoords="offset points",
                        ha="left", va="bottom", color="#222222")
        else:
            ax.text(158, float(credibility(158, lam, mu)) + 0.04, f"{m} мин", ha="center",
                    va="bottom", color="#222222")
    ax.plot(tau, np.exp(-lam * tau), color="#888888", lw=1.3, ls="--")
    ax.text(178, 0.955, "связи нет (μ = 0)", ha="right", va="bottom", color="#555555")
    ax.axhline(0.5, color="#bbbbbb", lw=0.8, zorder=0)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.07)
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax.set_xlabel("длительность молчания τ, мин")
    ax.set_ylabel("доверие π(τ)")
    fig.tight_layout()
    save(fig, "fig1")


def fig2():
    """а -- доля задач отказавших агентов, осмотренных другими за 30 мин; б -- повторные осмотры."""
    res = load("main")
    x = np.array(RS) / 1000
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6))
    for m in RULES:
        st = STYLE[m]
        kw = dict(color=st["color"], ls=st["ls"], marker=st["marker"], lw=st["lw"], ms=4.8,
                  zorder=st["zorder"])
        g = [100 * taken(by(res, m, R=R))[0] for R in RS]
        d = [100 * mean(by(res, m, R=R), "dup_share") for R in RS]
        axes[0].plot(x, g, label=LABEL[m], **kw)
        axes[1].plot(x, d, **kw)
    for ax in axes:
        r_axis(ax)
    axes[0].set_ylabel("осмотрено за 30 мин, %")
    axes[1].set_ylabel("повторные осмотры, %")
    axes[0].set_ylim(-3, 103)
    axes[1].set_ylim(-1.5, max(l.get_ydata().max() for l in axes[1].get_lines()) * 1.3)
    panel(axes[0], "а")
    panel(axes[1], "б")
    axes[1].legend(*axes[0].get_legend_handles_labels(), loc="upper right", handlelength=2.6,
                   borderaxespad=0.1, fontsize=10.5)       # справа вверху на панели б кривых нет
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig2")


def fig3():
    """а -- из-за чего ситуации упускаются к сроку (затухающее доверие);
    б -- доля ситуаций, заблокированных заявками-призраками, по правилам."""
    res = load("main")
    xi = np.arange(len(RS))
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    bottom = np.zeros(len(RS))
    parts = [("unknown", "о ситуации не знал\nни один агент"), ("other", "знали, но не успели"),
             ("ghost", "заблокирована заявкой")]
    for key, name in parts:
        v = np.array([100 * mean(by(res, "proposed", R=R), f"miss_{key}") for R in RS])
        ax.bar(xi, v, 0.7, bottom=bottom, color=LOSS_COLORS[key], edgecolor="white", linewidth=1.0,
               label=name)
        bottom += v
    ax.set_xticks(xi)
    ax.set_xticklabels([str(t) for t in RTICKS])
    ax.set_xlabel("радиус связи $R$, км")
    ax.set_ylabel("упущено, % ситуаций")
    ax.set_ylim(0, max(bottom) * 1.45)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", handlelength=1.2, borderaxespad=0.1, labelspacing=0.45, fontsize=10.5)
    panel(ax, "а")
    ax = axes[1]
    for m in RULES:
        st = STYLE[m]
        g = [100 * mean(by(res, m, R=R), "miss_ghost") for R in RS]
        ax.plot(np.array(RS) / 1000, g, color=st["color"], ls=st["ls"], marker=st["marker"],
                lw=st["lw"], ms=4.5, zorder=st["zorder"], label=LABEL[m])
    r_axis(ax)
    ax.set_ylabel("заблокировано, % ситуаций")
    top = max(max(l.get_ydata()) for l in ax.get_lines())
    ax.set_ylim(-0.05, top * 1.75)                     # место для легенды над кривыми
    ax.set_yticks([v for v in (0, 0.5, 1, 1.5, 2) if v <= top * 1.2])
    ax.legend(loc="upper right", handlelength=2.4, borderaxespad=0.1, fontsize=10.5)
    panel(ax, "б")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig3")


def fig4():
    """Доля задач отказавших агентов, осмотренных другими, по времени после отказа."""
    res = load("main")
    grid = np.linspace(0, 120, 481)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.65), sharey=True)
    titles = [f"а   $R$ ≤ {NEAR[-1] / 1000:g} км", f"б   $R$ ≥ {FAR[0] / 1000:g} км"]
    for ax, Rset, title in zip(axes, (NEAR, FAR), titles):
        for m in RULES:
            st = STYLE[m]
            lat = ghost_lat([r for R in Rset for r in by(res, m, R=R)])
            y = [100 * np.mean(lat <= t) for t in grid]
            ax.plot(grid, y, color=st["color"], ls=st["ls"], lw=st["lw"], zorder=st["zorder"],
                    label=LABEL[m])
        ax.set_xlim(0, 120)
        ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
        ax.set_ylim(0, 100)
        ax.set_xlabel("время после отказа, мин")
        ax.set_title(title, loc="left", fontsize=12)
    axes[0].set_ylabel("осмотрено, %")
    axes[0].legend(loc="upper right", handlelength=2.4, borderaxespad=0.2, fontsize=10.5)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig4")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("рисунки -> figures/")
