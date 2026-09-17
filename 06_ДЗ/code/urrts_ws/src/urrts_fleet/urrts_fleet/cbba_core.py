"""Ядро CBBA для потока заказов на складе (без ROS).

Отличия от ПР3:
  * набор заказов меняется: заказы появляются, берутся в работу и доставляются;
  * задачи адресуются идентификаторами (строки), а не индексами;
  * агент знает соседей только по радио и только с задержкой;
  * заказ, взятый роботом в работу («закреплённый»), из аукциона выходит;
  * победитель, которого давно не слышно, считается потерянным (этап 4).

Оценка маршрута: Σ R·λ^{t_завершения}, где t — время от «сейчас», включая
доезд до станции приёмки, погрузку, доезд до станции отгрузки и разгрузку.
"""
import math

EPS = 1e-9


class Order:
    def __init__(self, oid, pickup, drop, reward):
        self.id, self.pickup, self.drop, self.reward = oid, pickup, drop, float(reward)


class CbbaAgent:
    """Состояние агента: пакет, маршрут, ставки y, победители z, закреплённые заказы.

    start_station  — станция, в которой робот освободится (текущая или конец текущего заказа);
    start_delay    — через сколько секунд он там освободится.
    """

    def __init__(self, name, warehouse, capacity=3, discount=0.98, speed=0.3, handle_time=3.0):
        self.name = name
        self.wh = warehouse
        self.capacity = capacity
        self.discount = discount
        self.speed = speed
        self.handle_time = handle_time
        self.orders = {}           # id -> Order (известные открытые заказы)
        self.bundle, self.path = [], []
        self.y, self.z = {}, {}    # id -> ставка, id -> победитель ("" — нет)
        self.committed = set()     # заказы, взятые в работу кем-либо
        self.done = set()
        self.start_station = None
        self.start_delay = 0.0
        self.last_heard = {}       # робот -> время последнего сообщения

    # --- заказы ---------------------------------------------------------
    def add_order(self, order):
        if order.id in self.done or order.id in self.committed:
            return
        if order.id not in self.orders:
            self.orders[order.id] = order
            self.y.setdefault(order.id, 0.0)
            self.z.setdefault(order.id, "")

    def remove_order(self, oid, done=False):
        """Заказ закреплён или доставлен: убрать из аукциона, пакета и маршрута."""
        if done:
            self.done.add(oid)
        self.committed.add(oid)
        self.orders.pop(oid, None)
        if oid in self.bundle:
            k = self.bundle.index(oid)
            self._release_from(k, keep_first=False)
        self.path = [o for o in self.path if o in self.bundle]

    # --- оценка маршрута ------------------------------------------------
    def order_time(self, at_station, order):
        """Время выполнения заказа, начиная со станции at_station, с."""
        t = self.wh.station_distance(at_station, order.pickup) / self.speed
        t += self.handle_time
        t += self.wh.station_distance(order.pickup, order.drop) / self.speed
        t += self.handle_time
        return t

    def path_score(self, path):
        """Σ R·λ^t по маршруту из идентификаторов заказов."""
        t, at, s = self.start_delay, self.start_station, 0.0
        for oid in path:
            o = self.orders[oid]
            t += self.order_time(at, o)
            if math.isinf(t):
                return -math.inf
            s += o.reward * self.discount ** t
            at = o.drop
        return s

    # --- фаза 1: пакет ----------------------------------------------------
    def build_bundle(self):
        """Дополнить пакет, пока есть допустимые ставки и место.

        Для каждого открытого заказа вне маршрута: предельная выгода — наибольший
        прирост path_score по позициям вставки; ставка допустима, если выгода
        больше известной ставки y[o] на EPS. Взять заказ с наибольшей выгодой
        (при равенстве — меньший id), вставить в маршрут, добавить в конец пакета,
        y[o] = выгода, z[o] = self.name. Возвращает список добавленных заказов."""
        # >>> STUDENT ДЗ-э2-шаг1: как cbba_build_bundle в ПР3, но заказы — по id из self.orders,
        # >>> а закреплённые (self.committed) и доставленные (self.done) не рассматриваются.
        raise NotImplementedError("ДЗ-э2-шаг1: build_bundle")
        # <<< STUDENT

    # --- фаза 2: консенсус ------------------------------------------------
    def _release_from(self, k, keep_first=False):
        """Освободить заказы пакета начиная с позиции k (не включая k при keep_first)."""
        start = k + 1 if keep_first else k
        for oid in self.bundle[start:]:
            if self.z.get(oid) == self.name:
                self.y[oid], self.z[oid] = 0.0, ""
        removed = set(self.bundle[k:])
        self.bundle = self.bundle[:k]
        self.path = [o for o in self.path if o not in removed]

    def consensus(self, neighbour_states, now, stale_after=3.0):
        """Слияние с состояниями соседей.

        neighbour_states — словарь {робот: dict(stamp, y: {id: ставка}, z: {id: победитель},
        committed: [ids])}. Состояния старше stale_after секунд игнорируются.
        1) Закреплённые соседями заказы — remove_order.
        2) Для каждого заказа из self.orders: наибольшая ставка среди себя и свежих
           соседей; при равенстве — лексикографически меньшее имя победителя;
           пустой победитель не выигрывает.
        3) Освобождение: первый заказ пакета, у которого z ≠ self.name, и все
           после него (у последующих своих — сброс y = 0, z = "").
        Возвращает True, если что-либо изменилось."""
        # >>> STUDENT ДЗ-э2-шаг2: как cbba_consensus в ПР3, но по словарям и только со свежими соседями.
        # >>> Не забудьте обновить self.last_heard[робот] = stamp для свежих состояний.
        raise NotImplementedError("ДЗ-э2-шаг2: consensus")
        # <<< STUDENT

    # --- этап 4: потерянные победители --------------------------------------
    def forget_silent_winners(self, now, lost_after):
        """Победитель, которого не слышно дольше lost_after секунд, считается отказавшим:
        его ставки сбрасываются (y = 0, z = ""), заказы возвращаются в аукцион.
        Робот, от которого не было ни одного сообщения, не считается потерянным до
        истечения lost_after от первого вызова. Возвращает список освобождённых заказов."""
        # >>> STUDENT ДЗ-э4-шаг1: храните момент первого вызова; себя не забывайте никогда.
        return []
        # <<< STUDENT

    # --- обмен ------------------------------------------------------------
    def export_state(self, now):
        return {"stamp": now,
                "y": {o: self.y[o] for o in self.orders},
                "z": {o: self.z[o] for o in self.orders},
                "committed": sorted(self.committed)}

    def head(self):
        """Первый заказ маршрута — кандидат на взятие в работу."""
        return self.path[0] if self.path else None
