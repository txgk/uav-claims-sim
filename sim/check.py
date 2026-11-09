"""Проверка воспроизводимости.

    python3 check.py quick     # 6 прогонов заново и сравнение с results/main.json (≈ 10 с)
    python3 check.py verify    # SHA-256 файлов results/*.json против results/SHA256SUMS

Код выхода 0 -- всё совпало, 1 -- есть расхождения.
"""
import hashlib
import json
import math
import os
import sys
from dataclasses import replace

from model import Config, run_one
from stats import RES, load, rule

CASES = [("hard", 2000.0, 5000), ("proposed", 2000.0, 5000), ("ttl15", 5000.0, 5011),
         ("ttl30", 15000.0, 5023), ("proposed", 20000.0, 5029), ("indep", 10000.0, 5017)]


def same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def quick():
    ref = {(rule(r), r["R"], r["seed"]): r for r in load("main")}
    bad = 0
    for name, R, seed in CASES:
        method, ttl = ("ttl", int(name[3:]) * 60.0) if name.startswith("ttl") else (name, Config().ttl)
        r = run_one((replace(Config(), method=method, ttl=ttl, R=R), seed))
        want = ref[(name, R, seed)]
        diff = [k for k in want if not same(r.get(k), want[k])]
        bad += bool(diff)
        print(f"  {name:9s} R = {R / 1000:4.0f} км, seed {seed}: " + ("совпадает" if not diff else f"РАЗЛИЧИЯ в {diff}"))
    print("Все прогоны совпали с results/main.json." if not bad else "Есть расхождения.")
    return bad == 0


def verify():
    path = os.path.join(RES, "SHA256SUMS")
    ok = True
    with open(path) as f:
        for line in f:
            digest, name = line.split()
            p = os.path.join(RES, name)
            if not os.path.exists(p):
                print(f"  {name:18s} нет файла")
                ok = False
                continue
            with open(p, "rb") as g:
                got = hashlib.sha256(g.read()).hexdigest()
            print(f"  {name:18s} " + ("совпадает байт в байт" if got == digest else "ОТЛИЧАЕТСЯ"))
            ok &= got == digest
    print("ИТОГ: " + ("данные совпали с эталоном." if ok else "есть расхождения."))
    return ok


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "quick"
    raise SystemExit(0 if (quick() if what == "quick" else verify()) else 1)
