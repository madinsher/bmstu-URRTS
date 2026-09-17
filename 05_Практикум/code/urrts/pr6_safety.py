"""ПР6. Безопасность группы: барьерные функции (пособие §16.4.3, гл. 13; лекции 13, 16).

Шаги бригады:
    шаг 1 — ограничение барьерной функции для пары роботов;
    шаг 2 — безопасный фильтр: ближайшая к номинальной допустимая скорость (QP на плоскости);
    шаг 3 — децентрализованный регулятор «к цели + фильтр» для группы;
    шаг 4 — выход из взаимной блокировки (симметричный обмен местами);
    шаг 5 — устаревшие данные о соседе: запас на неопределённость и полная ответственность.

Модель: pᵢ' = uᵢ. Барьер пары h = ‖pᵢ − pⱼ‖² − d²; безопасное множество h ≥ 0.
Условие ḣ ≥ −γ h гарантирует, что h ≥ 0 сохраняется (h(t) ≥ h(0)e^{−γt}).
ḣ = 2(pᵢ − pⱼ)ᵀ(uᵢ − uⱼ); при взаимном доверии каждый берёт половину:
    2(pᵢ − pⱼ)ᵀ uᵢ ≥ −γ h / 2   (Wang, Li, Egerstedt, 2017; так работает Robotarium).
Ограничение записывается как aᵀu ≥ b.
"""
import itertools

import numpy as np


def pair_constraint(pi, pj, d_safe, gamma, share=0.5):
    """(a, b) ограничения aᵀuᵢ ≥ b для робота i относительно j.
    a = 2(pᵢ − pⱼ), b = −share·γ·h,  h = ‖pᵢ − pⱼ‖² − d²."""
    # >>> STUDENT ПР6-шаг1: по формулам из заголовка модуля.
    raise NotImplementedError("ПР6-шаг1: pair_constraint")
    # <<< STUDENT


def speed_polygon(vmax, sides=16):
    """Ограничение |u| ≤ vmax, аппроксимированное вписанным многоугольником: aₖᵀu ≥ −vmax·cos(π/sides)."""
    th = 2 * np.pi * np.arange(sides) / sides
    r = vmax * np.cos(np.pi / sides)
    return [(-np.array([np.cos(t), np.sin(t)]), -r) for t in th]


def safe_velocity(u_nom, constraints, tol=1e-9):
    """Решить QP  min ‖u − u_nom‖²  при aₖᵀu ≥ bₖ  на плоскости точно — перебором
    активных множеств: оптимум лежит среди кандидатов
        (1) сама u_nom;
        (2) проекции u_nom на прямые aₖᵀu = bₖ;
        (3) точки пересечения пар прямых (если прямые не параллельны).
    Из кандидатов, удовлетворяющих ВСЕМ ограничениям с допуском tol, вернуть
    ближайший к u_nom. Если допустимых нет — вернуть нулевой вектор
    (при h ≥ 0 и |u| ≤ vmax ноль всегда допустим: остановка безопасна)."""
    # >>> STUDENT ПР6-шаг2: проекция на прямую: u_nom + (b − aᵀu_nom)/‖a‖² · a;
    # >>> пересечение пары прямых — правило Крамера, пары с |det| < 1e-12 пропустить.
    # >>> Считайте всех кандидатов массивами numpy: фильтр вызывается тысячи раз за прогон.
    raise NotImplementedError("ПР6-шаг2: safe_velocity")
    # <<< STUDENT


def go_to_goal(p, goals, gain, vmax):
    """Номинальный регулятор: u = k(goal − p), ограниченный vmax."""
    u = gain * (np.asarray(goals, dtype=float) - np.asarray(p, dtype=float))
    n = np.linalg.norm(u, axis=1, keepdims=True)
    return u * np.minimum(1.0, vmax / np.maximum(n, 1e-12))


def cbf_controller(p, goals, d_safe, sense_radius, gamma=1.0, gain=1.0, vmax=0.2):
    """Децентрализованный безопасный регулятор: каждый робот i берёт номинальную
    скорость к цели и фильтрует её safe_velocity с ограничениями от всех соседей
    ближе sense_radius (share = 0.5) и многоугольником скорости speed_polygon(vmax).
    Возвращает (u_safe (N, 2), u_nom (N, 2))."""
    # >>> STUDENT ПР6-шаг3: соседи — по расстоянию в текущих позициях p.
    raise NotImplementedError("ПР6-шаг3: cbf_controller")
    # <<< STUDENT


def rotate(v, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def unstuck_nominal(p, goals, u_prev, hold, gain, vmax, stuck_speed=0.02, goal_tol=0.05,
                    angle=-np.pi / 4, hold_steps=50):
    """Правило «держись правее» с гистерезисом.

    Робот считается застрявшим, если он дальше goal_tol от цели, а его продвижение
    к цели на прошлом шаге — проекция u_prev на направление к цели — меньше
    stuck_speed. Проекция, а не модуль скорости: робот, скользящий вбок вдоль
    соседа, тоже не продвигается. Застрявшему роботу счётчик hold[i] взводится
    в hold_steps. Пока hold[i] > 0, номинальная скорость поворачивается на angle
    (−45° — вправо), и счётчик уменьшается на 1 за шаг. Все роботы поворачивают
    в одну сторону — симметрия ломается согласованно. Без гистерезиса правило
    отключается на первом же шаге бокового движения, и группа возвращается в тупик.
    Возвращает (номинальные скорости (N, 2), новый массив hold)."""
    # >>> STUDENT ПР6-шаг4: начните с go_to_goal; hold не изменяйте на месте — верните копию.
    raise NotImplementedError("ПР6-шаг4: unstuck_nominal")
    # <<< STUDENT


def stale_constraint(pi, pj_last, age, vmax_j, d_safe, gamma):
    """Ограничение относительно соседа, о котором известна лишь позиция age секунд
    назад. Сосед мог сместиться не больше чем на r = vmax_j·age, поэтому
    безопасная дистанция увеличивается: d' = d_safe + r. Сосед мог и не
    соблюдать своё ограничение (связь с ним потеряна), поэтому робот i берёт
    на себя всю ответственность: share = 1, и компенсирует возможное сближение
    соседа: b = −γ h' + 2‖pᵢ − pⱼ‖·vmax_j (худший случай uⱼ, направленной на i).
    Возвращает (a, b)."""
    # >>> STUDENT ПР6-шаг5: h' = ‖pᵢ − pⱼ‖² − d'²; a — как в pair_constraint.
    raise NotImplementedError("ПР6-шаг5: stale_constraint")
    # <<< STUDENT


def antipodal(n, radius, center=(0.0, 0.0)):
    """Роботы на окружности, цели — диаметрально противоположные точки."""
    th = 2 * np.pi * np.arange(n) / n
    p = np.c_[np.cos(th), np.sin(th)] * radius + np.asarray(center)
    return p, -(p - np.asarray(center)) + np.asarray(center)


def min_pair_distance(p):
    p = np.asarray(p)
    d = np.linalg.norm(p[:, None] - p[None], axis=2) + np.eye(len(p)) * 1e9
    return float(d.min())
