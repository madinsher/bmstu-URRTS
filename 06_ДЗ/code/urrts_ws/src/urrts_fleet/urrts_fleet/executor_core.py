"""Автомат исполнителя заказа (стратегический уровень робота), без ROS.

Состояния:
    IDLE        — свободен, ждёт заказ;
    TO_PICKUP   — едет на станцию приёмки;
    LOADING     — погрузка;
    TO_DROP     — едет на станцию отгрузки;
    UNLOADING   — разгрузка;
    TO_HOME     — возвращается на стоянку (нет заказов);
    FAILED      — отказ: робот остановлен, выйти можно только событием reset.

События (кортеж: имя, данные):
    ("order", id)        — CBBA закрепил за роботом заказ id;
    ("arrived", None)    — навигация достигла цели;
    ("nav_failed", None) — навигация не смогла достичь цели;
    ("handled", None)    — погрузка или разгрузка завершена;
    ("handle_failed", None);
    ("no_orders", None)  — заказов нет дольше порога простоя;
    ("stop", None)       — внешний аварийный останов;
    ("reset", None).

Команды (что сделать узлу, список кортежей):
    ("goto", станция), ("handle", "pick"|"drop"), ("release", id) — вернуть заказ в аукцион,
    ("delivered", id), ("cancel", None) — отменить текущее действие.
"""

STATES = ("IDLE", "TO_PICKUP", "LOADING", "TO_DROP", "UNLOADING", "TO_HOME", "FAILED")


class Executor:
    def __init__(self, home, max_retries=1):
        self.state = "IDLE"
        self.home = home
        self.order = None          # id текущего заказа
        self.pickup = None
        self.drop = None
        self.retries = 0
        self.max_retries = max_retries
        self.at_home = True

    def assign(self, oid, pickup, drop):
        """Новый заказ: данные заказа и событие ("order", id). Если автомат занят,
        данные текущего заказа не трогаются и команд нет."""
        if self.state not in ("IDLE", "TO_HOME"):
            return []
        self.pickup, self.drop = pickup, drop
        return self.step(("order", oid))

    def step(self, event):
        """Обработать событие; вернуть список команд. Необработанное событие в
        состоянии — пустой список команд, состояние не меняется."""
        name, data = event
        # >>> STUDENT ДЗ-э2-шаг3: реализуйте таблицу переходов (раздел 2 инструкции ДЗ).
        # >>> Требования: из любого состояния, кроме FAILED, событие stop ведёт в FAILED с командой cancel,
        # >>> а заказ, если он был и ещё не погружен (TO_PICKUP), возвращается командой release;
        # >>> nav_failed на пути к приёмке — повтор goto до max_retries раз, затем release и IDLE;
        # >>> nav_failed на пути к отгрузке — повтор без ограничения (груз уже на борту);
        # >>> order в TO_HOME прерывает возврат (cancel) и начинает заказ.
        raise NotImplementedError("ДЗ-э2-шаг3: Executor.step")
        # <<< STUDENT
