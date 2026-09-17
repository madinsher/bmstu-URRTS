"""ПР2. Роевое поведение и эмерджентность (пособие, гл. 5 и §1.1; лекции 1, 5).

Шаги бригады:
    шаг 1 — правила Рейнольдса (разделение, выравнивание, сплочение);
    шаг 2 — параметр порядка и модель Вичека: фазовый переход «беспорядок → стая»;
    шаг 3 — скорость PSO с локальной топологией (lbest);
    шаг 4 — отталкивание: переход от частиц PSO к роботам с телами.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Шаг 1. Правила Рейнольдса
# ---------------------------------------------------------------------------

def boids_acceleration(p, v, i, r_sep, r_view, w_sep=0.05, w_ali=5.0, w_coh=0.1):
    """Ускорение робота i по трём правилам Рейнольдса (пособие, §5.3).

    Соседи — роботы в радиусе r_view (сам i не сосед).
    разделение:   Σ (pᵢ − pⱼ) / ‖pᵢ − pⱼ‖²  по соседям ближе r_sep;
    выравнивание: v̄ соседей − vᵢ;
    сплочение:    p̄ соседей − pᵢ.
    Без соседей ускорение нулевое. Итог — взвешенная сумма трёх векторов."""
    # >>> STUDENT ПР2-шаг1: реализуйте по формулам; расстояние до себя исключите.
    raise NotImplementedError("ПР2-шаг1: boids_acceleration")
    # <<< STUDENT


def boids_step(p, v, dt, vmin, vmax, **params):
    """Синхронный шаг стаи: все ускорения считаются по старому состоянию,
    модуль скорости удерживается в [vmin, vmax] (у стаи нет «стоячих» особей;
    без нижней границы сплочение собирает рой в неподвижный ком)."""
    acc = np.array([boids_acceleration(p, v, i, **params) for i in range(len(p))])
    v_new = v + dt * acc
    speed = np.linalg.norm(v_new, axis=1, keepdims=True)
    v_new = v_new * np.clip(speed, vmin, vmax) / np.maximum(speed, 1e-12)
    return p + dt * v_new, v_new


# ---------------------------------------------------------------------------
# Шаг 2. Параметр порядка и модель Вичека
# ---------------------------------------------------------------------------

def polarization(v):
    """Параметр порядка φ = ‖Σ vᵢ‖ / Σ ‖vᵢ‖ ∈ [0, 1]:
    1 — все движутся в одну сторону, около 0 — хаотическое движение."""
    # >>> STUDENT ПР2-шаг2: при нулевой сумме модулей верните 0.0.
    raise NotImplementedError("ПР2-шаг2: polarization")
    # <<< STUDENT


def vicsek_step(p, theta, speed, radius, eta, box, rng):
    """Шаг модели Вичека (1995) на торе box×box.

    θᵢ⁺ = arg( Σ_{j: ‖pᵢ−pⱼ‖≤r} e^{iθⱼ} ) + ξᵢ,  ξᵢ ~ U[−η/2, η/2]  (сам i входит в сумму);
    pᵢ⁺ = pᵢ + speed · (cos θᵢ⁺, sin θᵢ⁺),  координаты — по модулю box.
    Расстояние на торе: разность координат приводится к [−box/2, box/2]."""
    # >>> STUDENT ПР2-шаг2: средний угол — через atan2 от сумм синусов и косинусов.
    # >>> Шум: rng.uniform(-eta / 2, eta / 2, size=N).
    raise NotImplementedError("ПР2-шаг2: vicsek_step")
    # <<< STUDENT


def vicsek_order(n, box, radius, eta, steps, rng, speed=0.03, burn_in=None):
    """Средний параметр порядка после переходного процесса."""
    p = rng.uniform(0, box, (n, 2))
    th = rng.uniform(-np.pi, np.pi, n)
    burn_in = steps // 2 if burn_in is None else burn_in
    acc = []
    for k in range(steps):
        p, th = vicsek_step(p, th, speed, radius, eta, box, rng)
        if k >= burn_in:
            acc.append(polarization(np.c_[np.cos(th), np.sin(th)]))
    return float(np.mean(acc))


# ---------------------------------------------------------------------------
# Шаги 3–4. PSO как распределённый поиск источника
# ---------------------------------------------------------------------------

def ring_neighborhood(n, k=1):
    """Кольцевая топология lbest: сосед i — роботы i−k … i+k (включая i)."""
    return [[(i + d) % n for d in range(-k, k + 1)] for i in range(n)]


def pso_velocity(x, v, pbest, pbest_val, neigh, rng, w=0.72, c1=1.49, c2=1.49, vmax=None):
    """Новая скорость PSO на максимум поля (пособие, §5.6.2) с локальной топологией:

    vᵢ⁺ = w vᵢ + c₁ r₁ (pbestᵢ − xᵢ) + c₂ r₂ (lbestᵢ − xᵢ),
    где lbestᵢ — pbest с наибольшим pbest_val среди neigh[i];
    r₁, r₂ ~ U[0,1]² — независимые покоординатно (сначала r₁ для всех, затем r₂).
    При vmax не None модуль скорости каждого робота ограничивается vmax."""
    # >>> STUDENT ПР2-шаг3: lbest — по индексу argmax pbest_val внутри окрестности;
    # >>> случайные числа: r1 = rng.random((N, 2)), затем r2 = rng.random((N, 2)).
    raise NotImplementedError("ПР2-шаг3: pso_velocity")
    # <<< STUDENT


def update_best(x, val, pbest, pbest_val):
    """Обновить личные рекорды там, где новое измерение лучше. Возвращает копии."""
    pbest = np.array(pbest, dtype=float)
    pbest_val = np.array(pbest_val, dtype=float)
    better = np.asarray(val) > pbest_val
    pbest[better] = np.asarray(x)[better]
    pbest_val[better] = np.asarray(val)[better]
    return pbest, pbest_val


def pso_optimize(field, x0, steps, rng, k=1, vmax=None):
    """Классический PSO (частицы без тел): возвращает (лучшая точка, значение, история лучшего)."""
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    pbest, pbest_val = x.copy(), np.array([field(z) for z in x])
    neigh = ring_neighborhood(len(x), k)
    hist = [pbest_val.max()]
    for _ in range(steps):
        v = pso_velocity(x, v, pbest, pbest_val, neigh, rng, vmax=vmax)
        x = x + v
        pbest, pbest_val = update_best(x, [field(z) for z in x], pbest, pbest_val)
        hist.append(pbest_val.max())
    b = int(np.argmax(pbest_val))
    return pbest[b], float(pbest_val[b]), np.array(hist)


def repulsion(x, d_min, gain=1.0):
    """Отталкивание роботов ближе d_min: Σⱼ gain·(d_min − d)·(xᵢ − xⱼ)/d.
    У частиц PSO нет тел; у роботов есть — без этого члена рой «слипается»."""
    # >>> STUDENT ПР2-шаг4: пары с d < 1e-9 (совпадающие позиции) пропустите.
    raise NotImplementedError("ПР2-шаг4: repulsion")
    # <<< STUDENT


def robot_pso_search(field, x0, steps, rng, vmax=0.05, d_min=0.1, k=1, noise_std=0.0, forget=0.0):
    """Роевой поиск источника роботами. Отличия от классического PSO:
    робот не может сместиться больше чем на vmax за шаг; роботы отталкиваются;
    поле измеряется в ФАКТИЧЕСКОЙ позиции робота и с шумом noise_std.
    forget > 0 — «забывание» рекордов: pbest_val уменьшается на forget·|pbest_val|
    за шаг, чтобы случайно завышенное шумом измерение не держало робота вечно.
    Возвращает (траектории (steps+1, N, 2), истинное значение поля в лучшей pbest по шагам)."""
    n = len(x0)

    def measure(z):
        return field(z) + (rng.normal(0, noise_std) if noise_std > 0 else 0.0)

    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    pbest, pbest_val = x.copy(), np.array([measure(z) for z in x])
    neigh = ring_neighborhood(n, k)
    traj, best = [x.copy()], [field(pbest[np.argmax(pbest_val)])]
    for _ in range(steps):
        if forget > 0:
            pbest_val = pbest_val - forget * np.abs(pbest_val)
        v = pso_velocity(x, v, pbest, pbest_val, neigh, rng)
        step = v + repulsion(x, d_min)
        s = np.linalg.norm(step, axis=1, keepdims=True)
        step = step * np.minimum(1.0, vmax / np.maximum(s, 1e-12))
        x = x + step
        v = step  # инерция робота — его фактическое смещение
        pbest, pbest_val = update_best(x, [measure(z) for z in x], pbest, pbest_val)
        traj.append(x.copy())
        best.append(field(pbest[np.argmax(pbest_val)]))
    return np.array(traj), np.array(best)


def gaussian_source(center, sigma=0.3, amp=1.0, distractors=()):
    """Поле концентрации: главный источник и, возможно, ложные локальные максимумы
    distractors = [(центр, sigma, amp), ...]."""
    center = np.asarray(center, dtype=float)

    def f(z):
        z = np.asarray(z, dtype=float)
        val = amp * np.exp(-np.sum((z - center) ** 2) / (2 * sigma ** 2))
        for c, s, a in distractors:
            val += a * np.exp(-np.sum((z - np.asarray(c)) ** 2) / (2 * s ** 2))
        return float(val)
    return f
