"""Стиль рисунков для статьи в Word: шрифт с засечками (метрически совпадает с Times New Roman),
кегль 12 pt при вставке в натуральную величину, латинские обозначения курсивом, греческие --
прямым шрифтом, десятичная запятая. Цвета правил одинаковы на всех рисунках; кроме цвета
правила различаются типом линии и маркером, поэтому рисунки читаются и в чёрно-белой печати."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402

plt.rcParams.update({
    "font.family": "Liberation Serif",
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.grid": True,
    "grid.color": "#e3e3e3",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

# правила: цвет (категориальная палитра, слоты 1–4 по порядку), линия, маркер, подпись
RULES = ["proposed", "hard", "ttl15", "ttl30"]
STYLE = {
    "proposed": dict(color="#2a78d6", ls="-", marker="D", lw=2.3, zorder=4),
    "hard": dict(color="#eb6834", ls="-", marker="s", lw=1.6, zorder=3),
    "ttl15": dict(color="#1baf7a", ls="--", marker="^", lw=1.6, zorder=3),
    "ttl30": dict(color="#eda100", ls="-.", marker="v", lw=1.6, zorder=3),
}
LABEL = {
    "proposed": "затухающее доверие",
    "hard": "жёсткие заявки",
    "ttl15": "таймаут 15 мин",
    "ttl30": "таймаут 30 мин",
    "indep": "независимые агенты",
}
# причины пропуска (рис. 3, а): свои цвета, не совпадающие с цветами правил
LOSS_COLORS = {"unknown": "#6250d6", "other": "#008300", "ghost": "#e87ba4"}


def decimal_comma(fig):
    """Десятичная запятая и знак «минус» на линейных осях."""
    def fmt(x, pos=None):
        s = f"{x:.10g}"
        return s if "e" in s else s.replace(".", ",").replace("-", "−")

    for ax in fig.get_axes():
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if scale == "linear" and isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(FuncFormatter(fmt))
