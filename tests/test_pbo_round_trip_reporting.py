"""样本诊断里的回合数必须是数出来的，不是垫出来的。

`pbo_from_matrix` 把调用方数出来的平均完整回合数交给
`sample_size_diagnosis`，路上做了一次 `max(1, round(avg_round_trips))`。
那个 `max(1, ...)` 会把"一个回合都没有"报成"约 1 笔完整交易"。

本模块对这件事有明确立场（`pbo_from_matrix` 的 docstring）：
`avg_round_trips=None` 表示"数不出来"，诊断里留空，"比塞一个猜出来的数诚实"。
0 是**数出来的 0**，不是数不出来 —— 把它垫成 1 等于在一份专治自欺的报告里
自己造了一个没发生过的回合。零成交网格恰恰是这套判据最该喊出来的情形
（`run_walk_forward` 另有"样本外零成交"守卫，理由同源）。
"""

from __future__ import annotations

import numpy as np

from vnpy_ctastrategy.overfitting import PBOStudy, pbo_from_matrix


def noise_matrix(n_obs: int = 240, n_configs: int = 12) -> np.ndarray:
    rng = np.random.default_rng(20260726)
    return rng.standard_normal((n_obs, n_configs)) * 0.01


def study_with(avg_round_trips: float | None) -> PBOStudy:
    return pbo_from_matrix(
        noise_matrix(),
        n_blocks=8,
        n_null_sims=0,
        avg_round_trips=avg_round_trips,
    )


def test_zero_round_trips_is_reported_as_zero() -> None:
    """一个回合都没数到时，诊断里就是 0，不是 1。"""
    diagnosis = study_with(0.0).diagnosis

    assert diagnosis.trades_total == 0
    assert diagnosis.trades_per_side == 0.0


def test_zero_round_trips_is_not_dressed_up_in_the_notes() -> None:
    """人读的那几行也不许出现"约 1 笔完整交易"。"""
    text = " ".join(study_with(0.0).diagnosis.notes)

    assert "全样本约 0 笔完整交易" in text
    assert "约 1 笔完整交易" not in text


def test_uncounted_round_trips_stay_empty() -> None:
    """None 仍然表示"数不出来"，与数出来的 0 是两件事。"""
    diagnosis = study_with(None).diagnosis

    assert diagnosis.trades_total is None
    assert diagnosis.trades_per_side is None


def test_counted_round_trips_pass_through_rounded() -> None:
    """真数出来的回合数照原样进诊断（四舍五入到整数）。"""
    assert study_with(23.4).diagnosis.trades_total == 23
    assert study_with(1.0).diagnosis.trades_total == 1
