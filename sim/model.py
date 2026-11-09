"""
Имитационная модель децентрализованного распределения задач ситуационного мониторинга
территории группой автономных агентов (БПЛА) при эпизодической связи.

Каждый агент держит короткий маршрут из не более чем B задач и объявляет заявки на все
задачи маршрута с плановым временем прибытия. Все сравниваемые правила используют один
и тот же выбор задач по ожидаемому вкладу (формула (4) статьи) и отличаются только
доверием ρ к чужим заявкам:

    indep    -- ρ = 0              независимые агенты (чужие заявки не учитываются);
    hard     -- ρ = 1              жёсткие заявки (действуют, пока их не отменят);
    ttl      -- ρ = 1[τ < T0]      фиксированный таймаут T0 (Config.ttl);
    proposed -- ρ = π(τ)·e^{−σδ}   затухающее доверие, формулы (1) и (3).

Дополнительно правило может требовать осмотра с вероятностью не ниже X по классам
важности (Config.x_req, условие (5) статьи).

Случайные потоки ситуаций, отказов и патрулирования порождаются генераторами
numpy.random.default_rng([seed, k]) и не зависят от правила, поэтому при одном seed все
правила сравниваются на одинаковых сценариях (общие случайные числа).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

METHODS = ("indep", "hard", "ttl", "proposed")


@dataclass(frozen=True)
class Config:
    # территория и группа
    L: float = 50_000.0             # сторона квадратной территории, м; ПУ в центре
    n_agents: int = 8
    speed: float = 20.0             # крейсерская скорость, м/с
    dt: float = 20.0                # шаг моделирования, с
    t_end: float = 8 * 3600.0       # длительность смены, с
    t_gen_end: float = 7 * 3600.0   # после этого новые ситуации не появляются
    # связь
    R: float = 5000.0               # радиус связи агент -- агент, м
    base_factor: float = 2.0        # радиус связи агент -> ПУ = base_factor * R
    bcast_period: float = 60.0      # период сводки ПУ всем исправным агентам, с
    # ситуации
    r_sense: float = 500.0          # радиус бортового обнаружения, м
    event_rate: float = 40.0 / 3600 # интенсивность появления ситуаций, 1/с
    class_w: tuple = (4.0, 2.0, 1.0)                    # важность классов
    class_T: tuple = (20 * 60.0, 45 * 60.0, 90 * 60.0)  # характерное время развития, с
    class_p: tuple = (0.25, 0.35, 0.40)                 # доли классов
    report_prob: float = 0.7        # доля ситуаций, о которых сообщают внешние источники
    report_delay: tuple = (120.0, 600.0)
    hotspot_share: float = 0.6      # доля ситуаций вокруг очагов
    n_hotspots: int = 4
    hotspot_sigma: float = 2000.0
    # отказы
    fail_prob_mission: float = 0.3  # вероятность «тихого» отказа агента за смену
    # поведение
    bundle: int = 3                 # длина маршрута (число заявленных задач)
    replan_period: float = 60.0
    idle_period: float = 40.0
    hysteresis: float = 0.15        # первая задача меняется, только если новая выгоднее на 15 %
    delta_min: float = 0.05         # минимальный вклад, при котором задача берётся в маршрут
    min_value_share: float = 0.02   # задача рассматривается, если её ценность > 2 % исходной
    # правило
    method: str = "proposed"
    ttl: float = 15 * 60.0          # таймаут T0 для method="ttl", с
    x_req: tuple | None = None      # требуемая вероятность осмотра для классов (A, B, C)
    # оценивание в правиле proposed
    lam0_factor: float = 1.0        # априорная интенсивность отказов = lam0_factor * истинная
    mu_prior_mean: float = 1 / 600.0    # априорное среднее интенсивности сообщений, 1/с
    mu_prior_n: float = 2.0             # вес априорного распределения (число наблюдений)
    sigma_prior_mean: float = 1 / 1800.0
    sigma_prior_n: float = 2.0
    gap_min: float = 25.0           # интервалы короче этого не считаются эпизодами молчания
    # запись данных для калибровки формулы (1)
    validity_period: float = 300.0

    @property
    def lam_true(self) -> float:
        p = min(max(self.fail_prob_mission, 0.0), 0.999)
        return -math.log(1.0 - p) / self.t_end if p > 0 else 0.0


def credibility(tau, lam, mu):
    """Формула (1): апостериорная вероятность работоспособности агента, молчащего tau."""
    tau = np.asarray(tau, dtype=float)
    theta = lam + mu
    x = np.minimum(theta * tau, 700.0)
    return 1.0 / (1.0 + (lam / theta) * np.expm1(x))


def half_life(lam, mu):
    """Формула (2): период полураспада доверия."""
    return math.log(2.0 + mu / lam) / (lam + mu)


class Simulation:
    def __init__(self, cfg: Config, seed: int, record_validity: bool = False):
        self.cfg = c = cfg
        self.seed = seed
        N, B = c.n_agents, c.bundle
        self.N, self.B = N, B
        self.base = N                                  # индекс узла ПУ
        self.base_pos = np.array([c.L / 2, c.L / 2])

        # --- ситуации: общие для всех правил при одном seed ---
        rg = np.random.default_rng([seed, 11])
        gaps = rg.exponential(1.0 / c.event_rate, size=int(c.event_rate * c.t_gen_end * 3) + 50)
        times = np.cumsum(gaps)
        times = times[times < c.t_gen_end]
        E = len(times)
        self.E = E
        cls = rg.choice(3, size=E, p=c.class_p)
        centers = rg.uniform(0.15 * c.L, 0.85 * c.L, size=(c.n_hotspots, 2))
        hot = rg.random(E) < c.hotspot_share
        which = rg.integers(c.n_hotspots, size=E)
        loc_hot = centers[which] + rg.normal(0.0, c.hotspot_sigma, size=(E, 2))
        loc_uni = rg.uniform(0.0, c.L, size=(E, 2))
        self.ev_xy = np.clip(np.where(hot[:, None], loc_hot, loc_uni), 0.0, c.L)
        self.ev_t0 = times
        self.ev_cls = cls
        self.ev_w = np.array(c.class_w)[cls]
        self.ev_T = np.array(c.class_T)[cls]
        reported = rg.random(E) < c.report_prob
        delay = rg.uniform(*c.report_delay, size=E)
        self.ev_t_report = np.where(reported, times + delay, np.inf)
        self.ev_deadline = times + self.ev_T * math.log(10.0)     # ценность падает до 10 %

        # --- отказы: общие для всех правил ---
        rf = np.random.default_rng([seed, 22])
        lam = c.lam_true
        self.fail_time = rf.exponential(1.0 / lam, size=N) if lam > 0 else np.full(N, np.inf)
        self.fail_time[self.fail_time > c.t_end] = np.inf

        self.rng_patrol = np.random.default_rng([seed, 33])
        # сектора патрулирования: сетка 4 x 2
        nx, ny = 4, 2
        self.sector = np.zeros((N, 4))
        for i in range(N):
            gx, gy = i % nx, (i // nx) % ny
            self.sector[i] = [gx * c.L / nx, (gx + 1) * c.L / nx, gy * c.L / ny, (gy + 1) * c.L / ny]

        # --- состояние агентов ---
        self.pos = np.array([self._new_waypoint(i) for i in range(N)], dtype=float)
        self.alive = np.ones(N, dtype=bool)
        self.route = [[] for _ in range(N)]
        self.waypoint = np.array([self._new_waypoint(i) for i in range(N)])
        self.next_decision = self.rng_patrol.uniform(0.0, c.replan_period, size=N)

        # --- знания узлов (агенты 0..N-1, ПУ -- узел N) ---
        M = N + 1
        self.kn_event = np.zeros((M, E), dtype=bool)        # знает о ситуации
        self.kn_served = np.full((M, E), np.inf)             # известный момент осмотра
        self.kn_info_t = np.zeros((M, N))                    # время последних сведений об агенте
        self.kn_claim = np.full((M, N, B), -1, dtype=int)    # известные заявки
        self.kn_eta = np.full((M, N, B), np.inf)             # их плановое время прибытия
        # оценивание интенсивностей сообщений μ и отмен заявок σ
        self.gap_n = np.zeros((M, N))
        self.gap_sum = np.zeros((M, N))
        self.ab_n = np.zeros(M)
        self.ab_exposure = np.zeros(M)

        # --- истина и показатели ---
        self.served_true = np.full(E, np.inf)
        self.server = np.full(E, -1, dtype=int)
        self.visits = 0
        self.dup_visits = 0
        self.ev_visits = np.zeros(E, dtype=int)
        self.ev_status_checked = np.zeros(E, dtype=bool)
        self.miss_log = []          # (класс, причина): 0 -- никто не знал, 1 -- заявка, 2 -- прочее
        self.ghost_tasks = []       # (ситуация, момент отказа, сколько исправных агентов знали)
        self.lam0 = c.lam_true * c.lam0_factor if c.lam_true > 0 else 1e-7 * c.lam0_factor
        self.record_validity = record_validity
        self.validity = []

    # ------------------------------------------------------------------
    def _new_waypoint(self, i):
        x0, x1, y0, y1 = self.sector[i]
        return np.array([self.rng_patrol.uniform(x0, x1), self.rng_patrol.uniform(y0, y1)])

    def value(self, j, t):
        return self.ev_w[j] * np.exp(-(t - self.ev_t0[j]) / self.ev_T[j])

    def learned_rates(self, i):
        """Оценки μ (по каждому агенту) и σ узла i: гамма-распределение, сопряжённое
        экспоненциальному."""
        c = self.cfg
        mu = (c.mu_prior_n + self.gap_n[i]) / (c.mu_prior_n / c.mu_prior_mean + self.gap_sum[i])
        sigma = (c.sigma_prior_n + self.ab_n[i]) / (c.sigma_prior_n / c.sigma_prior_mean + self.ab_exposure[i])
        return mu, sigma

    def rho_proposed(self, i, t):
        """Формула (3): доверие узла i ко всем известным ему заявкам, массив (N, B)."""
        tau = np.maximum(t - self.kn_info_t[i], 0.0)
        mu, sigma = self.learned_rates(i)
        pi = credibility(tau, self.lam0, mu)
        s = np.clip(self.kn_eta[i] - self.kn_info_t[i][:, None], 0.0, None)
        s = np.minimum(s, tau[:, None])                                # δ = min{τ, (t_kj − s_ik)^+}
        return np.exp(-sigma * s) * pi[:, None]

    def rho(self, i, t):
        m = self.cfg.method
        N, B = self.N, self.B
        if m == "indep":
            return np.zeros((N, B))
        if m == "hard":
            return np.ones((N, B))
        if m == "ttl":
            tau = t - self.kn_info_t[i]
            return np.repeat((tau < self.cfg.ttl).astype(float)[:, None], B, axis=1)
        return self.rho_proposed(i, t)

    def p_free(self, i, ta, claims_j, rho):
        """Произведение (1 − ρ) по претендентам с более ранним прибытием, чем у агента i."""
        p = 1.0
        eta = self.kn_eta[i]
        for (k, b) in claims_j:
            tk = eta[k, b]
            if tk < ta - 1.0 or (abs(tk - ta) <= 1.0 and k < i):
                p *= 1.0 - rho[k, b]
        return p

    def plan(self, i, t):
        """Жадное построение маршрута из не более чем B задач по вкладу (4)."""
        c = self.cfg
        open_mask = self.kn_event[i] & np.isinf(self.kn_served[i]) & (self.ev_t0 <= t)
        J = np.nonzero(open_mask)[0]
        if len(J) == 0:
            return [], [], []
        rho = self.rho(i, t)
        cl = self.kn_claim[i].copy()
        cl[i, :] = -1
        claims = {}
        ks, bs = np.nonzero(cl >= 0)
        for k, b in zip(ks, bs):
            claims.setdefault(int(cl[k, b]), []).append((int(k), int(b)))
        route, etas, gains = [], [], []
        p0 = self.pos[i].copy()
        t0 = t
        avail = np.ones(len(J), dtype=bool)
        cur_first = self.route[i][0] if self.route[i] else -1
        for step in range(c.bundle):
            Jc = J[avail]
            if len(Jc) == 0:
                break
            d = np.hypot(self.ev_xy[Jc, 0] - p0[0], self.ev_xy[Jc, 1] - p0[1])
            ta = t0 + d / c.speed
            vi = self.value(Jc, ta)
            g = vi.copy()
            for idx, j in enumerate(Jc):
                cj = claims.get(int(j))
                if cj:
                    pf = self.p_free(i, float(ta[idx]), cj, rho)
                    if c.x_req is not None and 1.0 - pf < c.x_req[int(self.ev_cls[j])]:
                        pf = 1.0         # осмотр другими менее вероятен, чем требуется (5)
                    g[idx] = pf * vi[idx]
            g = np.where(vi > c.min_value_share * self.ev_w[Jc], g, -np.inf)
            b = int(np.argmax(g))
            if step == 0 and cur_first >= 0:        # гистерезис для первой задачи маршрута
                hit = np.nonzero(Jc == cur_first)[0]
                if len(hit) and g[hit[0]] >= c.delta_min and g[b] <= (1 + c.hysteresis) * g[hit[0]]:
                    b = int(hit[0])
            if g[b] < c.delta_min:
                break
            j = int(Jc[b])
            route.append(j)
            etas.append(float(ta[b]))
            gains.append(float(g[b]))
            avail[np.nonzero(J == j)[0][0]] = False
            p0 = self.ev_xy[j].copy()
            t0 = ta[b]
        return route, etas, gains

    def decide(self, i, t):
        route, etas, _ = self.plan(i, t)
        if not route and self.route[i]:
            self.waypoint[i] = self._new_waypoint(i)
        self.route[i] = route
        self.kn_claim[i, i, :] = -1
        self.kn_eta[i, i, :] = np.inf
        for b, (j, e) in enumerate(zip(route, etas)):
            self.kn_claim[i, i, b] = j
            self.kn_eta[i, i, b] = e
        self.kn_info_t[i, i] = t

    # ------------------------------------------------------------------
    def components(self):
        """Связные группы узлов: агенты на расстоянии ≤ R, агент и ПУ -- ≤ base_factor·R."""
        c = self.cfg
        nodes = list(np.nonzero(self.alive)[0]) + [self.base]
        P = np.vstack([self.pos[self.alive], self.base_pos[None, :]])
        n = len(nodes)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
        lim = np.full((n, n), c.R)
        lim[-1, :] = lim[:, -1] = c.R * c.base_factor
        adj = D <= lim
        for a in range(n):
            for b in range(a + 1, n):
                if adj[a, b]:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[ra] = rb
        groups = {}
        for a in range(n):
            groups.setdefault(find(a), []).append(nodes[a])
        return [g for g in groups.values() if len(g) > 1]

    def _absorb(self, a, best_t, best_claim, best_eta, sv, t):
        """Узел a принимает более свежие сведения об агентах и обновляет оценки μ и σ."""
        upd = self.kn_info_t[a] < best_t
        if not upd.any():
            return
        ks = np.nonzero(upd)[0]
        gaps = best_t[ks] - self.kn_info_t[a, ks]
        silent = gaps > self.cfg.gap_min
        self.gap_n[a, ks[silent]] += 1
        self.gap_sum[a, ks[silent]] += gaps[silent]
        changed_target = False
        for k, g in zip(ks, gaps):
            old = set(self.kn_claim[a, k][self.kn_claim[a, k] >= 0].tolist())
            new = set(best_claim[k][best_claim[k] >= 0].tolist())
            if old:                         # отмена заявки: задача исчезла из маршрута, не осмотрена
                self.ab_exposure[a] += g * len(old)
                for j in old - new:
                    if not np.isfinite(sv[j]):
                        self.ab_n[a] += 1
            if a < self.N and self.route[a] and k != a and (new - old):
                if set(self.route[a]) & (new - old):
                    changed_target = True
        self.kn_info_t[a, ks] = best_t[ks]
        self.kn_claim[a, ks] = best_claim[ks]
        self.kn_eta[a, ks] = best_eta[ks]
        if changed_target:
            self.next_decision[a] = min(self.next_decision[a], t)

    def merge(self, C, t):
        """Обмен сведениями внутри связной группы (ретрансляция по цепочке за один шаг)."""
        C = np.array(C)
        ev = self.kn_event[C].any(axis=0)
        sv = self.kn_served[C].min(axis=0)
        self.kn_event[C] = ev
        self.kn_served[C] = sv
        it = self.kn_info_t[C]
        best = it.argmax(axis=0)
        ar = np.arange(self.N)
        best_t = it[best, ar]
        best_claim = self.kn_claim[C][best, ar]
        best_eta = self.kn_eta[C][best, ar]
        for a in C:
            self._absorb(a, best_t, best_claim, best_eta, sv, t)

    def broadcast(self, t):
        """Сводка ПУ всем исправным агентам (прямой канал покрывает всю территорию)."""
        src = self.base
        for a in np.nonzero(self.alive)[0]:
            if (self.kn_event[src] & ~self.kn_event[a]).any():
                self.kn_event[a] |= self.kn_event[src]
                self.next_decision[a] = min(self.next_decision[a], t)
            sv = np.minimum(self.kn_served[a], self.kn_served[src])
            self.kn_served[a] = sv
            self._absorb(a, self.kn_info_t[src], self.kn_claim[src], self.kn_eta[src], sv, t)

    # ------------------------------------------------------------------
    def claim_valid(self, k, j):
        return self.alive[k] and (j in self.route[k] or self.server[j] == k)

    def classify_missed(self, e, t):
        """Причина, по которой ситуация e не осмотрена к сроку актуальности: никто из
        исправных агентов о ней не знал (0); каждый знавший видел заявку-призрак с более
        ранним прибытием и доверием к ней больше 0,5 (1); прочее (2)."""
        cls = int(self.ev_cls[e])
        knowers = [a for a in np.nonzero(self.alive)[0] if self.kn_event[a, e]]
        if not knowers:
            self.miss_log.append((cls, 0))
            return
        if self.cfg.method == "indep":
            self.miss_log.append((cls, 2))
            return
        for a in knowers:
            rho = self.rho(a, t)
            ta = t + math.hypot(*(self.ev_xy[e] - self.pos[a])) / self.cfg.speed
            blocked = False
            for k in range(self.N):
                if k == a:
                    continue
                for b in range(self.B):
                    if self.kn_claim[a, k, b] != e:
                        continue
                    if (not self.claim_valid(k, e)) and self.kn_eta[a, k, b] < ta and rho[k, b] > 0.5:
                        blocked = True
            if not blocked:
                self.miss_log.append((cls, 2))
                return
        self.miss_log.append((cls, 1))

    def sample_validity(self, t):
        """Данные для проверки формулы (1): все известные исправным агентам заявки на ещё
        не осмотренные ситуации, длительность молчания владельца и его состояние."""
        for a in np.nonzero(self.alive)[0]:
            rp = self.rho_proposed(a, t)
            mu, _ = self.learned_rates(a)
            for k in range(self.N):
                if k == a:
                    continue
                for b in range(self.B):
                    j = self.kn_claim[a, k, b]
                    if j < 0:
                        continue
                    if np.isfinite(self.served_true[j]) and self.server[j] != k:
                        continue
                    self.validity.append((t - self.kn_info_t[a, k], bool(self.claim_valid(k, j)),
                                          float(rp[k, b]), b, float(mu[k]), float(self.alive[k])))

    def run(self):
        c = self.cfg
        N = self.N
        steps = int(round(c.t_end / c.dt))
        step_len = c.speed * c.dt
        every_bcast = max(int(round(c.bcast_period / c.dt)), 1)
        every_valid = max(int(round(c.validity_period / c.dt)), 1)
        for s in range(1, steps + 1):
            t = s * c.dt
            # 1. «тихие» отказы: заявки отказавшего агента остаются в сведениях остальных
            newly = self.alive & (self.fail_time <= t)
            if newly.any():
                others = self.alive & ~newly
                for k in np.nonzero(newly)[0]:
                    for j in self.route[k]:
                        if np.isinf(self.served_true[j]):
                            knowers = int(np.sum(self.kn_event[:N][others, j]))
                            self.ghost_tasks.append((j, t, knowers))
                self.alive[newly] = False
            # 2. сообщения внешних источников на ПУ
            rep = (self.ev_t_report <= t) & ~self.kn_event[self.base]
            if rep.any():
                self.kn_event[self.base, rep] = True
            # 3. движение и осмотр
            for i in np.nonzero(self.alive)[0]:
                if self.route[i] and np.isfinite(self.kn_served[i, self.route[i][0]]):
                    self.decide(i, t)
                    self.next_decision[i] = t + c.replan_period
                tgt = self.route[i][0] if self.route[i] else -1
                goal = self.ev_xy[tgt] if tgt >= 0 else self.waypoint[i]
                d = goal - self.pos[i]
                dist = math.hypot(d[0], d[1])
                if dist <= step_len:
                    self.pos[i] = goal
                    if tgt >= 0:
                        self.visits += 1
                        self.ev_visits[tgt] += 1
                        if np.isinf(self.served_true[tgt]):
                            self.served_true[tgt] = t
                            self.server[tgt] = i
                        else:
                            self.dup_visits += 1
                        self.kn_served[i, tgt] = min(self.kn_served[i, tgt], t)
                        self.kn_event[i, tgt] = True
                        self.route[i].pop(0)
                        self.next_decision[i] = t
                    else:
                        self.waypoint[i] = self._new_waypoint(i)
                else:
                    self.pos[i] += d / dist * step_len
            # 4. бортовое обнаружение
            act = np.nonzero((self.ev_t0 <= t) & (self.ev_deadline > t))[0]
            if len(act):
                al = np.nonzero(self.alive)[0]
                if len(al):
                    dx = self.ev_xy[act, 0][None, :] - self.pos[al, 0][:, None]
                    dy = self.ev_xy[act, 1][None, :] - self.pos[al, 1][:, None]
                    near = dx * dx + dy * dy <= c.r_sense ** 2
                    for r, i in enumerate(al):
                        hit = act[near[r]]
                        if len(hit):
                            new = hit[~self.kn_event[i, hit]]
                            if len(new):
                                self.kn_event[i, new] = True
                                self.next_decision[i] = min(self.next_decision[i], t)
            # 5. сведения агента о самом себе всегда свежие
            for i in np.nonzero(self.alive)[0]:
                self.kn_info_t[i, i] = t
            # 6. обмен внутри связных групп
            for C in self.components():
                before = self.kn_event[C].sum(axis=1)
                self.merge(C, t)
                after = self.kn_event[C].sum(axis=1)
                for a, b0, b1 in zip(C, before, after):
                    if a < N and b1 > b0:
                        self.next_decision[a] = min(self.next_decision[a], t)
            # 6а. сводка ПУ
            if math.isfinite(c.bcast_period) and s % every_bcast == 0:
                self.broadcast(t)
            # 7. решения
            for i in np.nonzero(self.alive)[0]:
                if t >= self.next_decision[i]:
                    self.decide(i, t)
                    self.next_decision[i] = t + (c.replan_period if self.route[i] else c.idle_period)
            # 8. ситуации, упущенные к сроку актуальности
            for e in np.nonzero((self.ev_deadline <= t) & ~self.ev_status_checked)[0]:
                self.ev_status_checked[e] = True
                if np.isinf(self.served_true[e]):
                    self.classify_missed(e, t)
            if self.record_validity and s % every_valid == 0:
                self.sample_validity(t)
        return self.metrics()

    def metrics(self):
        c = self.cfg
        served = np.isfinite(self.served_true)
        delay = self.served_true - self.ev_t0
        val = np.where(served, self.ev_w * np.exp(-np.where(served, delay, 0.0) / self.ev_T), 0.0)
        ontime = served & (self.served_true <= self.ev_deadline)
        n_checked = max(int(self.ev_status_checked.sum()), 1)
        miss = np.zeros((3, 3))
        for q, r in self.miss_log:
            miss[q, r] += 1
        out = dict(
            seed=self.seed, method=c.method, R=c.R, ttl=c.ttl, fail=c.fail_prob_mission,
            x_req=list(c.x_req) if c.x_req is not None else None,
            n_events=int(self.E),
            value_share=float(val.sum() / self.ev_w.sum()),             # индекс J
            ontime_share=float(ontime.mean()),
            dup_share=float(self.dup_visits / max(self.visits, 1)),
            visits=int(self.visits),
            miss_unknown=float(miss[:, 0].sum() / n_checked),
            miss_ghost=float(miss[:, 1].sum() / n_checked),
            miss_other=float(miss[:, 2].sum() / n_checked),
            failures=int(np.isfinite(self.fail_time).sum()),
        )
        for q in range(3):
            m = self.ev_cls == q
            v = self.ev_visits[m]
            n = int(m.sum())
            out[f"value_c{q}"] = float(val[m].sum() / max(self.ev_w[m].sum(), 1e-9))
            out[f"ontime_c{q}"] = float(ontime[m].mean()) if n else float("nan")
            out[f"dup_c{q}"] = float(np.maximum(v - 1, 0).sum() / max(v.sum(), 1))
            out[f"n_c{q}"] = n
            for r, name in enumerate(("unknown", "ghost", "other")):
                out[f"miss_{name}_c{q}"] = float(miss[q, r] / max(n, 1))
        ghosts = []
        for j, tf, kn in self.ghost_tasks:
            st = self.served_true[j]
            ok = np.isfinite(st) and st <= self.ev_deadline[j]
            ghosts.append(dict(lat_min=float((st - tf) / 60) if ok else None, knowers=kn,
                               cls=int(self.ev_cls[j])))
        out["ghost_n"] = len(ghosts)
        lat = np.array([np.inf if g["lat_min"] is None else g["lat_min"] for g in ghosts])
        out["ghost_taken30"] = float(np.mean(lat <= 30)) if len(lat) else float("nan")
        out["ghost_never"] = float(np.mean(~np.isfinite(lat))) if len(lat) else float("nan")
        out["ghosts"] = ghosts
        return out


def run_one(args):
    cfg, seed = args[0], args[1]
    record = len(args) > 2 and args[2]
    sim = Simulation(cfg, seed, record_validity=record)
    m = sim.run()
    if record:
        v = np.array(sim.validity, dtype=float).reshape(-1, 6)
        m["validity"] = dict(tau=v[:, 0].tolist(), valid=v[:, 1].astype(int).tolist(),
                             rho=v[:, 2].tolist(), pos=v[:, 3].astype(int).tolist(),
                             mu=v[:, 4].tolist(), alive=v[:, 5].astype(int).tolist())
    return m
