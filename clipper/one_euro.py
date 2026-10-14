"""Адаптивное сглаживание траектории камеры (One Euro Filter).

Алгоритм из статьи «1€ Filter: A Simple Speed-based Low-pass Filter for
Noisy Input in Interactive Systems» (Casiez, Roussel, Vogel, CHI 2012) —
публичный, реализован здесь с нуля.

Суть: обычный низкочастотный фильтр либо дрожит (мало сглаживания), либо
запаздывает (много). One Euro решает компромисс адаптивно — он смотрит на
СКОРОСТЬ сигнала: когда камера должна стоять (медленный дрейф) — сглаживает
сильно, камера стоит мёртво; когда цель резко прыгнула (сменился говорящий)
— почти не сглаживает, реагирует мгновенно.

Для нас это убирает «вялый дрейф» на группах/панораме, не внося задержки в
реакцию на смену говорящего.
"""
import math


class _LowPass:
    __slots__ = ("_y", "_has")

    def __init__(self):
        self._y = 0.0
        self._has = False

    def __call__(self, x: float, alpha: float) -> float:
        if not self._has:
            self._y = x
            self._has = True
        else:
            self._y = alpha * x + (1.0 - alpha) * self._y
        return self._y

    @property
    def last(self) -> float:
        return self._y

    @property
    def ready(self) -> bool:
        return self._has


class OneEuroFilter:
    """Одномерный One Euro Filter.

    min_cutoff: базовая частота среза (Гц). Ниже — сильнее сглаживание в покое.
    beta:       коэффициент скорости. Выше — быстрее реакция на резкие движения.
    d_cutoff:   частота среза для оценки скорости.
    """

    def __init__(self, min_cutoff: float = 0.15, beta: float = 0.05,
                 d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_t = None

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._last_t is None:
            self._last_t = t
            return self._x(x, 1.0)      # первый отсчёт — без сглаживания
        dt = t - self._last_t
        self._last_t = t
        if dt <= 0:
            return self._x.last
        # оценка скорости сигнала
        dx = (x - self._x.last) / dt
        edx = self._dx(dx, self._alpha(dt, self.d_cutoff))
        # динамическая частота среза: чем быстрее движется цель, тем меньше
        # сглаживаем (камера успевает за резкой сменой говорящего)
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x(x, self._alpha(dt, cutoff))


def smooth_series(points: list, min_cutoff: float = 0.15, beta: float = 0.05):
    """Сглаживает список (t, x) One Euro-фильтром. Возвращает список x.

    Устойчиво к неравномерным интервалам между кадрами (использует реальное dt).
    """
    f = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
    return [f(x, t) for t, x in points]
