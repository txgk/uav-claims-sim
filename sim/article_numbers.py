"""Все числа, которые приводятся в тексте статьи, по разделам -> stdout.

    python3 article_numbers.py
"""
import os

import numpy as np

from model import Config, credibility
from stats import RES, by, ghost_lat, load, mean, never, paired, taken

RS = [1000.0, 2000.0, 3000.0, 5000.0, 7000.0, 10000.0, 15000.0, 20000.0]
REGIMES = [2000.0, 5000.0, 15000.0]
NEAR = [1000.0, 2000.0, 3000.0]
FAR = [10000.0, 15000.0, 20000.0]
CLAIM_RULES = ["hard", "ttl15", "ttl30", "proposed"]
LAM = Config().lam_true
BINS = np.array([0, 1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360])     # мин


def row(vals, fmt="{:6.1f}"):
    return " ".join(fmt.format(v) for v in vals)


def head(title):
    print("\n" + title)
    print("  R, км            " + row([R / 1000 for R in RS], "{:6.0f}"))


def calibration():
    print("\nПроверка формулы (1): заявки на первую задачу маршрута, доля исправных владельцев")
    if not os.path.exists(os.path.join(RES, "validity.json")):
        print("  нет results/validity.json")
        return
    data = load("validity")
    for R in REGIMES:
        rows = [d["validity"] for d in data if d["R"] == R]
        tau = np.concatenate([np.array(v["tau"]) for v in rows]) / 60.0
        alive = np.concatenate([np.array(v["alive"]) for v in rows])
        pos = np.concatenate([np.array(v["pos"]) for v in rows])
        mu = np.concatenate([np.array(v["mu"]) for v in rows])
        m0 = pos == 0
        tau, alive, mu = tau[m0], alive[m0], mu[m0]
        xs, ys, ns = [], [], []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (tau >= lo) & (tau < hi)
            if m.sum() >= 25:
                xs.append(np.sqrt(max(lo, 0.5) * hi))
                ys.append(alive[m].mean())
                ns.append(int(m.sum()))
        emp, bracket = float("nan"), None
        for (x0, y0), (x1, y1) in zip(zip(xs[:-1], ys[:-1]), zip(xs[1:], ys[1:])):
            if y0 >= 0.5 > y1:
                emp = float(np.exp(np.log(x0) + (0.5 - y0) / (y1 - y0) * (np.log(x1) - np.log(x0))))
                bracket = (x0, x1)
                break
        mu_med = float(np.median(mu))
        th = np.log(2 + mu_med / LAM) / (LAM + mu_med) / 60.0
        print(f"  R = {R / 1000:4.0f} км: интервал между сообщениями (медиана оценки) {1 / mu_med / 60:5.1f} мин;"
              f" прогноз τ½ {th:5.1f} мин; факт {emp:5.1f} мин"
              + (f" (между {bracket[0]:.1f} и {bracket[1]:.1f} мин)" if bracket else "")
              + f"; записей {len(tau)}")
        print("     доли по интервалам τ: " + ", ".join(f"{x:.1f}: {y:.2f} (n={n})"
                                                       for x, y, n in zip(xs, ys, ns)))


def main_series():
    mn = load("main")
    head("Индекс J")
    for m in ["indep"] + CLAIM_RULES:
        print(f"  {m:16s}", row([mean(by(mn, m, R=R), 'value_share') for R in RS], "{:6.3f}"))
    gaps = [max(mean(by(mn, m, R=R), "value_share") for m in CLAIM_RULES) -
            min(mean(by(mn, m, R=R), "value_share") for m in CLAIM_RULES) for R in RS]
    print("  разница hard/ttl/proposed", row(gaps, "{:6.3f}"), f"  максимум {max(gaps):.4f}")

    head("Повторные осмотры, %")
    for m in ["indep"] + CLAIM_RULES:
        print(f"  {m:16s}", row([100 * mean(by(mn, m, R=R), 'dup_share') for R in RS]))
    print("  proposed − hard, п. п. (парная разность ± 95 %):")
    print("                   " + "  ".join(
        "{:+.1f}±{:.1f}".format(*paired(by(mn, "hard", R=R), by(mn, "proposed", R=R), "dup_share", 100))
        for R in RS))

    head("Задачи отказавших агентов: осмотрены другими не позднее 30 мин после отказа, %")
    for m in CLAIM_RULES:
        print(f"  {m:16s}", row([100 * taken(by(mn, m, R=R))[0] for R in RS]))
    print("  число таких задач (за 30 прогонов):", {m: [taken(by(mn, m, R=R))[1] for R in RS]
                                                    for m in CLAIM_RULES})
    head("Задачи отказавших агентов, не осмотренные до срока актуальности: «n из N»")
    for m in CLAIM_RULES:
        print(f"  {m:16s}", "  ".join("{}/{}".format(*never(by(mn, m, R=R))) for R in RS))
    for lo in (5000.0, 7000.0, 10000.0):
        grp = [R for R in RS if R >= lo]
        print(f"  при R ≥ {lo / 1000:g} км:", ", ".join(
            "{}: {} из {} ({:.0f} %)".format(m, *never([r for R in grp for r in by(mn, m, R=R)]),
                                             100 * np.divide(*never([r for R in grp for r in by(mn, m, R=R)])))
            for m in CLAIM_RULES))
    print("  задач за отказавшими аппаратами за смену (среднее):")
    for m in CLAIM_RULES:
        print(f"  {m:16s}", row([mean(by(mn, m, R=R), 'ghost_n') for R in RS], "{:6.2f}"))

    head("Структура потерь при затухающем доверии, % всех ситуаций (рис. 3, а)")
    tot = np.zeros(len(RS))
    for key, name in (("unknown", "никто не знал"), ("other", "знали, не успели"), ("ghost", "заблокирована")):
        v = np.array([100 * mean(by(mn, "proposed", R=R), f"miss_{key}") for R in RS])
        tot += v
        print(f"  {name:16s}", row(v, "{:6.2f}"))
    print(f"  {'всего':16s}", row(tot))
    head("Заблокировано заявками-призраками, % всех ситуаций (рис. 3, б)")
    for m in CLAIM_RULES:
        print(f"  {m:16s}", row([100 * mean(by(mn, m, R=R), 'miss_ghost') for R in RS], "{:6.2f}"))

    grid = [5, 10, 15, 20, 30, 45, 60, 90, 120]
    print("\nОсмотрено задач отказавших агентов через t мин после отказа, % (рис. 4)")
    print("  t, мин                    ", row(grid, "{:6d}"), "  никогда")
    for Rset, name in ((NEAR, "R ≤ 3 км"), (FAR, "R ≥ 10 км")):
        for m in CLAIM_RULES:
            lat = ghost_lat([r for R in Rset for r in by(mn, m, R=R)])
            print(f"  {name:10s} {m:9s}      ", row([100 * np.mean(lat <= t) for t in grid]),
                  f"  {100 * np.mean(~np.isfinite(lat)):5.1f} %  (задач {len(lat)})")
        kn = [g["knowers"] > 0 for R in Rset for m in CLAIM_RULES for r in by(mn, m, R=R) for g in r["ghosts"]]
        print(f"  {name}: в момент отказа задача известна хотя бы одному исправному агенту — "
              f"{100 * np.mean(kn):.1f} %")


def ttl_sweep():
    if not os.path.exists(os.path.join(RES, "ttl.json")):
        print("\nнет results/ttl.json")
        return
    tt, mn = load("ttl"), load("main")
    ttls = sorted({int(round(r["ttl"] / 60)) for r in tt})
    print("\nПеребор таймаута: осмотрено за 30 мин, % / повторные осмотры, %")
    for R in REGIMES:
        p = by(mn, "proposed", R=R)
        tp, dp = 100 * taken(p)[0], 100 * mean(p, "dup_share")
        print(f"  R = {R / 1000:g} км; затухающее доверие: {tp:5.1f} / {dp:5.1f}")
        dom = []
        for T in ttls:
            g = by(tt, f"ttl{T}", R=R)
            tg, dg = 100 * taken(g)[0], 100 * mean(g, "dup_share")
            mark = " <- лучше по обоим" if (tg > tp and dg < dp) else ""
            if mark:
                dom.append(T)
            print(f"     таймаут {T:4d} мин: {tg:5.1f} / {dg:5.1f}{mark}")
        print(f"     таймауты, превзошедшие затухающее доверие по обоим показателям: {dom or 'нет'}")


def redundancy():
    if not os.path.exists(os.path.join(RES, "redundancy.json")):
        print("\nнет results/redundancy.json")
        return
    rd, mn = load("redundancy"), load("main")
    variants = [[0.9, 0.0, 0.0], [0.99, 0.9, 0.0]]
    W = np.array(Config().class_w) * np.array(Config().class_p)
    share = W / W.sum()
    print("\nТребуемая вероятность осмотра X (A, B, C), условие (5)")
    print("  отказ  R, км  X                   J      вовремя A    B      C    повт.%   пропуск A: незнание / прочее / заявка, %")
    for fail, rs in ((0.3, RS), (0.7, [2000.0, 5000.0])):
        for R in rs:
            base = by(mn, "proposed", R=R) if fail == 0.3 else by(rd, "proposed", R=R, fail=fail, x_req=None)
            for x in [None] + variants:
                g = base if x is None else by(rd, "proposed", R=R, fail=fail, x_req=x)
                f = lambda k: mean(g, k)
                print(f"  {fail:4.1f}  {R / 1000:5.0f}  {str(x):18s} {f('value_share'):.3f}    {100 * f('ontime_c0'):5.1f}"
                      f"  {100 * f('ontime_c1'):5.1f}  {100 * f('ontime_c2'):5.1f}   {100 * f('dup_share'):5.1f}"
                      f"     {100 * f('miss_unknown_c0'):4.1f} / {100 * f('miss_other_c0'):4.1f} / {100 * f('miss_ghost_c0'):4.2f}")
    print("\n  Парные разности с правилом без требования (± полуширина 95 % интервала) и вклад классов в ΔJ")
    for fail, rs in ((0.3, RS), (0.7, [2000.0, 5000.0])):
        for R in rs:
            base = by(mn, "proposed", R=R) if fail == 0.3 else by(rd, "proposed", R=R, fail=fail, x_req=None)
            for x in variants:
                g = by(rd, "proposed", R=R, fail=fail, x_req=x)
                dJ = paired(base, g, "value_share")
                dA = paired(base, g, "ontime_c0", 100)
                dC = paired(base, g, "ontime_c2", 100)
                contrib = [share[q] * (mean(g, f"value_c{q}") - mean(base, f"value_c{q}")) for q in range(3)]
                print(f"  {fail:4.1f}  {R / 1000:5.0f}  {str(x):18s} ΔJ {dJ[0]:+.3f}±{dJ[1]:.3f}  "
                      f"Δвовремя A {dA[0]:+.1f}±{dA[1]:.1f}  Δвовремя C {dC[0]:+.1f}±{dC[1]:.1f}  "
                      f"вклад A/B/C {contrib[0]:+.3f}/{contrib[1]:+.3f}/{contrib[2]:+.3f}")
    lam03, lam07 = Config().lam_true, Config(fail_prob_mission=0.7).lam_true
    print(f"\n  вероятность отказа за 10 мин полёта: {100 * (1 - np.exp(-lam03 * 600)):.2f} % (0,3 за смену),"
          f" {100 * (1 - np.exp(-lam07 * 600)):.2f} % (0,7 за смену)")


def analytic():
    lam = 1 / (24 * 60)
    print("\nФормула (2) при средней наработке на отказ 24 ч: период полураспада, мин")
    for m in (2, 7, 20, 60):
        print(f"  сообщения раз в {m:2d} мин: {np.log(2 + (1 / m) / lam) / (lam + 1 / m):6.1f}")
    print(f"  без связи: {np.log(2) / lam / 60:.1f} ч")


if __name__ == "__main__":
    analytic()
    calibration()
    main_series()
    ttl_sweep()
    redundancy()
