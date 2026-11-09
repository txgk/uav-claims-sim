"""Серии экспериментов статьи -> results/*.json.

    python3 experiments.py                 # все серии
    python3 experiments.py main ttl        # отдельные серии

Серии (во всех seed 5000–5029, кроме validity):
    main        радиус связи R × правило (independent, hard, ttl 15 и 30 мин, proposed);
    ttl         перебор таймаута T0 при трёх радиусах связи;
    validity    записи о каждой заявке молчащего агента для проверки формулы (1), seed 5000–5005;
    redundancy  правило proposed с требуемой вероятностью осмотра X, условие (5);
                при вероятности отказа 0,3 -- все R, при 0,7 -- R = 2 и 5 км.

Прогоны идут параллельно (по числу ядер, не больше 8); на результат это не влияет:
каждый прогон определяется конфигурацией и seed.
"""
import json
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool, cpu_count

from model import Config, run_one

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SEEDS = list(range(5000, 5030))
RS = [1000.0, 2000.0, 3000.0, 5000.0, 7000.0, 10000.0, 15000.0, 20000.0]   # радиусы связи, м
REGIMES = [2000.0, 5000.0, 15000.0]                       # редкая, средняя и частая связь
TTL_SWEEP = [5, 10, 15, 20, 30, 45, 60, 90, 120]          # мин
X_VARIANTS = [(0.9, 0.0, 0.0), (0.99, 0.9, 0.0)]          # требуемая вероятность осмотра (A, B, C)
PROCS = max(1, min(cpu_count(), 8))


def rules():
    base = Config()
    return [replace(base, method="indep"), replace(base, method="hard"),
            replace(base, method="ttl", ttl=15 * 60.0), replace(base, method="ttl", ttl=30 * 60.0),
            replace(base, method="proposed")]


def series(name):
    base = Config()
    if name == "main":
        return [(replace(c, R=R), s) for R in RS for c in rules() for s in SEEDS]
    if name == "ttl":
        return [(replace(base, method="ttl", ttl=T * 60.0, R=R), s)
                for R in REGIMES for T in TTL_SWEEP for s in SEEDS]
    if name == "validity":
        return [(replace(base, method="proposed", R=R), s, True) for R in REGIMES for s in SEEDS[:6]]
    if name == "redundancy":
        jobs = [(replace(base, method="proposed", R=R, x_req=x), s)
                for R in RS for x in X_VARIANTS for s in SEEDS]
        jobs += [(replace(base, method="proposed", R=R, fail_prob_mission=0.7, x_req=x), s)
                 for R in (2000.0, 5000.0) for x in [None] + X_VARIANTS for s in SEEDS]
        return jobs
    raise SystemExit(f"неизвестная серия: {name}")


def main(names):
    os.makedirs(RES, exist_ok=True)
    for name in names:
        jobs = series(name)
        t0 = time.time()
        with Pool(PROCS) as p:
            res = p.map(run_one, jobs, chunksize=1)
        with open(os.path.join(RES, f"{name}.json"), "w") as f:
            json.dump(res, f)
        print(f"[{name}] {len(res)} прогонов за {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["main", "ttl", "validity", "redundancy"])
