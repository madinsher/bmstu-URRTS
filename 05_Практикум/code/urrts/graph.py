"""Граф связи группы роботов (пособие, §4.1).

Все функции работают с неориентированными графами без петель, заданными
симметричной матрицей смежности A формы (N, N) с элементами 0/1 или весами.
Лапласиан и алгебраическую связность бригада реализует сама в ПР1
(`pr1_consensus.laplacian`, `pr1_consensus.algebraic_connectivity`);
здесь — вспомогательные функции, которые нужны всем практикумам.
"""
import numpy as np


def disk_graph(p, radius, alive=None):
    """Матрица смежности графа «связь есть, если расстояние не больше radius».

    p      — позиции, массив (N, 2);
    alive  — необязательная маска (N,) работоспособных роботов: отказавший
             робот не имеет рёбер.
    """
    p = np.asarray(p, dtype=float)
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    A = (d <= radius).astype(float)
    np.fill_diagonal(A, 0.0)
    if alive is not None:
        m = np.asarray(alive, dtype=bool)
        A[~m, :] = 0.0
        A[:, ~m] = 0.0
    return A


def neighbors(A, i):
    """Индексы соседей вершины i."""
    return np.flatnonzero(np.asarray(A)[i] > 0)


def components(A):
    """Разбиение вершин на компоненты связности (список списков индексов)."""
    A = np.asarray(A)
    n = A.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in np.flatnonzero(A[v] > 0):
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        comps.append(sorted(comp))
    return comps


def is_connected(A, subset=None):
    """Связен ли граф (или его подграф на вершинах subset)."""
    A = np.asarray(A)
    if subset is not None:
        idx = np.asarray(subset)
        A = A[np.ix_(idx, idx)]
    if A.shape[0] <= 1:
        return True
    return len(components(A)) == 1


def ring_graph(n):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1.0
    return A


def path_graph(n):
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    return A


def star_graph(n):
    A = np.zeros((n, n))
    A[0, 1:] = A[1:, 0] = 1.0
    return A


def complete_graph(n):
    return np.ones((n, n)) - np.eye(n)


def random_geometric_graph(n, radius, rng, size=1.0, connected=True, max_tries=1000):
    """Случайный геометрический граф на квадрате size×size.

    При connected=True генерирует до получения связного графа.
    Возвращает (позиции, матрица смежности).
    """
    for _ in range(max_tries):
        p = rng.uniform(0.0, size, size=(n, 2))
        A = disk_graph(p, radius)
        if not connected or is_connected(A):
            return p, A
    raise RuntimeError("не удалось получить связный граф: увеличьте radius")


def metropolis_weights(A):
    """Веса Метрополиса — Гастингса: двоякостохастическая матрица W для
    дискретного усреднения без знания числа агентов (пособие, §4.3.2).
    Каждому агенту нужны только свои степень и степени соседей."""
    A = (np.asarray(A) > 0).astype(float)
    deg = A.sum(axis=1)
    n = A.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j in np.flatnonzero(A[i]):
            W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
        W[i, i] = 1.0 - W[i].sum()
    return W
