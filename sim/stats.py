"""Загрузка результатов и агрегирование: средние, доверительные интервалы, объединённые доли."""
import json
import math
import os

import numpy as np
from scipy import stats

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def load(name):
    with open(os.path.join(RES, f"{name}.json")) as f:
        return json.load(f)


def rule(r):
    """Имя правила прогона: indep, hard, ttl15, ttl30, ..., proposed."""
    return f"ttl{int(round(r['ttl'] / 60))}" if r["method"] == "ttl" else r["method"]


def by(res, rule_name=None, **kw):
    """Прогоны с заданным правилом и значениями полей (R, fail, x_req, ...)."""
    out = []
    for r in res:
        if rule_name is not None and rule(r) != rule_name:
            continue
        if all((r[k] == v) if not isinstance(v, float) else math.isclose(r[k], v) for k, v in kw.items()):
            out.append(r)
    return out


def mean(runs, key):
    return float(np.mean([r[key] for r in runs]))


def mean_ci(values, conf=0.95):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float(v.mean()) if len(v) else float("nan"), 0.0
    h = stats.t.ppf(0.5 + conf / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(h)


def paired(a_runs, b_runs, key, scale=1.0):
    """Парная разность b − a по совпадающим seed: среднее и полуширина 95 % интервала."""
    a = {r["seed"]: r[key] for r in a_runs}
    d = [scale * (r[key] - a[r["seed"]]) for r in b_runs if r["seed"] in a]
    return mean_ci(d)


def ghost_lat(runs):
    """Задержки (мин) осмотра задач отказавших агентов; inf -- не осмотрена до срока."""
    return np.array([np.inf if g["lat_min"] is None else g["lat_min"] for r in runs for g in r["ghosts"]])


def taken(runs, minutes=30):
    """Доля задач отказавших агентов, осмотренных другими не позднее minutes после отказа,
    и число таких задач (объединение по прогонам)."""
    lat = ghost_lat(runs)
    return (float(np.mean(lat <= minutes)) if len(lat) else float("nan")), len(lat)


def never(runs):
    lat = ghost_lat(runs)
    return int(np.sum(~np.isfinite(lat))), len(lat)
