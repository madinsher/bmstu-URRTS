"""Распределённое резервирование клеток (тактический уровень робота), без ROS.

Робот держит («holds») клетку, в которой стоит, и до lookahead клеток маршрута
впереди, в которые ему разрешён въезд («reserved»). Пока в очереди есть место,
робот заявляет («want») следующую клетку маршрута с меткой времени — в том числе
на ходу, не дожидаясь прибытия. Заявленная клетка переходит в reserved, если
одновременно:
  1) ни один свежий сосед не держит эту клетку;
  2) среди свежих соседей, заявивших ту же клетку, у робота наивысший приоритет:
     более ранняя метка заявки, при равенстве — меньшее имя робота;
  3) с момента своей заявки прошло не меньше settle секунд;
  4) каждый свежий сосед **подтвердил осведомлённость**: его последнее состояние
     отправлено позже, чем self.want_since + settle, то есть он уже мог услышать
     заявку и учесть её в своём решении.

Правило 4 важнее, чем кажется: одного ожидания settle недостаточно, если узлы
подтормаживают (загруженная машина, ускоренное время) — тогда сосед просто не
успевает ни услышать заявку, ни сообщить о своей, и двое въезжают в одну клетку.
Проверка по меткам времени превращает «подождать и надеяться» в подтверждение.

Состояние соседа — словарь dict(stamp, holds: [[r, c], …], want: [r, c] | None, want_since: t).
Свежесть — как в CBBA: now − stamp ≤ stale_after. Сосед, молчащий дольше, для
резервирования не существует: безопасность при потере связи здесь НЕ гарантируется
(это обсуждается в отчёте и решается защитным слоем этапа 4).

Взаимная блокировка: если заявка ждёт слишком долго, робот строит объездной
маршрут от последней зарезервированной клетки, считая клетки, которые держат
свежие соседи, препятствиями. Чтобы два робота не объезжали друг друга одновременно
и не встречались снова («вечная вежливость»), порог несимметричен: робот ждёт
wait_limit, если клетку держит робот со старшим (меньшим) именем, и 2·wait_limit,
если младший, — первым уступает младший.
"""


class Traffic:
    def __init__(self, name, warehouse, settle=0.3, stale_after=2.0, wait_limit=6.0, lookahead=2):
        self.name = name
        self.wh = warehouse
        self.settle = settle
        self.stale_after = stale_after
        self.wait_limit = wait_limit
        self.lookahead = lookahead
        self.cell = None           # текущая клетка
        self.route = []            # клетки впереди, ещё не зарезервированные
        self.reserved = []         # клетки впереди, въезд в которые разрешён (по порядку)
        self.goal = None
        self.want = None
        self.want_since = None
        self.replans = 0

    # --- маршрут --------------------------------------------------------------
    def set_goal(self, cell, goal):
        """Новая цель. Уже зарезервированные клетки сохраняются: робот может быть на ходу."""
        self.cell, self.goal = tuple(cell), tuple(goal)
        start = self.reserved[-1] if self.reserved else self.cell
        path = self.wh.path(start, self.goal)
        self.route = [] if path is None else path[1:]
        self.want = self.want_since = None
        return path is not None

    def clear(self):
        """Отказаться от маршрута и заявки; зарезервированные клетки доехать."""
        self.route = []
        self.want = self.want_since = None

    def holds(self):
        return ([self.cell] if self.cell is not None else []) + list(self.reserved)

    def done(self):
        return not self.route and not self.reserved and self.cell == self.goal

    def export_state(self, now):
        return {"stamp": now, "holds": [list(c) for c in self.holds()],
                "want": list(self.want) if self.want else None, "want_since": self.want_since}

    def fresh(self, neighbours, now):
        return {k: v for k, v in neighbours.items() if k != self.name and now - v["stamp"] <= self.stale_after}

    # --- решение на такте ---------------------------------------------------------
    def may_enter(self, cell, neighbours, now):
        """Можно ли зарезервировать клетку cell сейчас (правила 1–4 из заголовка модуля).
        neighbours — уже отфильтрованные свежие соседи."""
        # >>> STUDENT ДЗ-э3-шаг1: ключ приоритета — кортеж (want_since, имя робота);
        # >>> своя заявка должна быть именно на эту клетку (self.want == cell);
        # >>> подтверждение соседа — его st["stamp"] > self.want_since + self.settle.
        raise NotImplementedError("ДЗ-э3-шаг1: may_enter")
        # <<< STUDENT

    def step(self, neighbours, now):
        """Такт резервирования: заявить или зарезервировать следующую клетку.
        Возвращает True, если очередь reserved пополнилась."""
        if not self.route or len(self.reserved) >= self.lookahead:
            return False
        fresh = self.fresh(neighbours, now)
        nxt = self.route[0]
        if self.want != nxt:
            self.want, self.want_since = nxt, now
            return False
        if self.may_enter(nxt, fresh, now):
            self.reserved.append(self.route.pop(0))
            self.want = self.want_since = None
            return True
        blockers = [r for r, v in fresh.items() if any(tuple(h) == tuple(nxt) for h in v["holds"])]
        limit = self.wait_limit if any(r < self.name for r in blockers) else 2 * self.wait_limit
        if now - self.want_since > limit:
            self.replan_around(fresh, now)
        return False

    def next_target(self):
        """Клетка, в которую ехать сейчас, или None."""
        return self.reserved[0] if self.reserved else None

    def arrived(self):
        """Робот доехал до первой зарезервированной клетки: она становится текущей."""
        if self.reserved:
            self.cell = self.reserved.pop(0)

    def replan_around(self, fresh, now):
        """Объезд: маршрут до цели от последней зарезервированной клетки (или текущей),
        в котором клетки, занятые свежими соседями, считаются препятствиями.
        Если объезда нет или он совпадает с текущим маршрутом — продолжать ждать
        (want_since = now, чтобы не удерживать высокий приоритет вечно) и вернуть False.
        Возвращает True, если маршрут изменился."""
        # >>> STUDENT ДЗ-э3-шаг2: временно пометьте занятые соседями клетки (кроме цели) как стеллажи
        # >>> в self.wh.grid, найдите путь self.wh.path и обязательно верните сетку (try/finally).
        raise NotImplementedError("ДЗ-э3-шаг2: replan_around")
        # <<< STUDENT
