from collections import defaultdict
from datetime import (
    date as Date,
    datetime,
    timedelta
)
from typing import cast, Any, TYPE_CHECKING
from collections.abc import Callable
from functools import lru_cache, partial
import traceback

import numpy as np
from pandas import DataFrame, Series
from pandas.core.window import ExponentialMovingWindow
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vnpy.trader.constant import (
    Direction,
    Offset,
    Exchange,
    Interval,
    Status
)
from vnpy.trader.database import get_database, BaseDatabase
from vnpy.trader.object import OrderData, TradeData, BarData, TickData
from vnpy.trader.utility import round_to, extract_vt_symbol
from vnpy.trader.optimize import (
    OptimizationSetting,
    check_optimization_setting,
    run_bf_optimization,
    run_ga_optimization
)

from .permutation_test import (
    PERMUTATION_FIELD_DEFAULTS,
    PERMUTATION_SETTING_KEYS,
    attach_permutation_statistics,
)
from .robust_metrics import RobustMetrics, calculate_robust_metrics
from .sharpe_inference import SharpeInference, sharpe_inference, statistics_fields
from .base import (
    BacktestingMode,
    EngineType,
    STOPORDER_PREFIX,
    StopOrder,
    StopOrderStatus,
    INTERVAL_DELTA_MAP
)
from .template import CtaTemplate
from .locale import _

if TYPE_CHECKING:                                       # pragma: no cover
    # 运行期只在方法体内 import：optimization_gates → overfitting → backtesting
    # 是一条真实的导入环，模块级 import 会在解释器加载 backtesting 时炸掉。
    from .optimization_gates import (
        OptimizationGateConfig,
        OptimizationResults,
        ReturnsPayload,
    )


class BacktestingEngine:
    """"""

    engine_type: EngineType = EngineType.BACKTESTING
    gateway_name: str = "BACKTESTING"

    def __init__(self) -> None:
        """"""
        self.vt_symbol: str = ""
        self.symbol: str = ""
        self.exchange: Exchange
        self.start: datetime
        self.end: datetime
        self.rate: float = 0
        self.slippage: float = 0
        self.size: float = 1
        self.pricetick: float = 0
        self.capital: int = 1_000_000
        self.risk_free: float = 0
        self.annual_days: int = 240
        self.half_life: int = 120
        self.mode: BacktestingMode = BacktestingMode.BAR

        self.strategy_class: type[CtaTemplate]
        self.strategy: CtaTemplate
        self.tick: TickData
        self.bar: BarData
        self.datetime: datetime = datetime(1970, 1, 1)

        self.interval: Interval
        self.days: int = 0
        self.callback: Callable
        self.history_data: list = []

        self.stop_order_count: int = 0
        self.stop_orders: dict[str, StopOrder] = {}
        self.active_stop_orders: dict[str, StopOrder] = {}

        self.limit_order_count: int = 0
        self.limit_orders: dict[str, OrderData] = {}
        self.active_limit_orders: dict[str, OrderData] = {}

        self.trade_count: int = 0
        self.trades: dict[str, TradeData] = {}

        self.logs: list = []

        self.daily_results: dict[Date, DailyResult] = {}
        self.daily_df: DataFrame = DataFrame()

        # Block-permutation significance test (fork addition). Opt-in because
        # it costs ~90ms per call at B=1000/n=600, which a 5000-combo grid
        # search would pay 5000 times. See enable_permutation_test().
        self.permutation_enabled: bool = False
        self.permutation_settings: dict = {}

    def clear_data(self) -> None:
        """
        Clear all data of last backtesting.
        """
        self.stop_order_count = 0
        self.stop_orders.clear()
        self.active_stop_orders.clear()

        self.limit_order_count = 0
        self.limit_orders.clear()
        self.active_limit_orders.clear()

        self.trade_count = 0
        self.trades.clear()

        self.logs.clear()
        self.daily_results.clear()

    def set_parameters(
        self,
        vt_symbol: str,
        interval: Interval,
        start: datetime,
        rate: float,
        slippage: float,
        size: float,
        pricetick: float,
        capital: int = 0,
        end: datetime | None = None,
        mode: BacktestingMode = BacktestingMode.BAR,
        risk_free: float = 0,
        annual_days: int = 240,
        half_life: int = 120
    ) -> None:
        """"""
        self.mode = mode
        self.vt_symbol = vt_symbol
        self.interval = Interval(interval)
        self.rate = rate
        self.slippage = slippage
        self.size = size
        self.pricetick = pricetick
        self.start = start

        self.symbol, exchange_str = self.vt_symbol.split(".")
        self.exchange = Exchange(exchange_str)

        self.capital = capital

        if not end:
            end = datetime.now()
        self.end = end.replace(hour=23, minute=59, second=59)

        self.mode = mode
        self.risk_free = risk_free
        self.annual_days = annual_days
        self.half_life = half_life

    def enable_permutation_test(
        self, enabled: bool = True, /, **settings: object
    ) -> None:
        """Turn on the block-permutation significance test in calculate_statistics.

        Off by default. Every other number in the statistics dict is in-sample
        description; this one answers "could this be luck?" by reshuffling the
        exposure series in blocks while holding the market path fixed, and
        reporting where the observed statistic falls in that null distribution.

        settings is forwarded verbatim to permutation_statistics(); see that
        function for the full parameter list. Unknown keys raise immediately
        rather than being swallowed by the fail-open wrapper downstream.

        enabled is positional-only so that **settings can never accidentally
        bind to it — a setting named "enabled" would otherwise silently flip
        the switch instead of being rejected as unknown.

        Read permutation_test.py section 5 before wiring perm_p_value into an
        optimization target: minimising p over a grid re-introduces exactly the
        selection bias the test is meant to expose.
        """
        unknown: set = set(settings) - PERMUTATION_SETTING_KEYS
        if unknown:
            raise ValueError(
                _("未知的重排检验参数：{}，可选：{}").format(
                    sorted(unknown), sorted(PERMUTATION_SETTING_KEYS)
                )
            )
        self.permutation_enabled = enabled
        self.permutation_settings = dict(settings)

    def add_strategy(self, strategy_class: type[CtaTemplate], setting: dict) -> None:
        """"""
        self.strategy_class = strategy_class
        self.strategy = strategy_class(
            self, strategy_class.__name__, self.vt_symbol, setting
        )

    def load_data(self) -> None:
        """"""
        self.output(_("开始加载历史数据"))

        if not self.end:
            self.end = datetime.now()

        if self.start >= self.end:
            self.output(_("起始日期必须小于结束日期"))
            return

        self.history_data.clear()       # Clear previously loaded history data

        # Load 30 days of data each time and allow for progress update
        total_days: int = (self.end - self.start).days
        progress_days: int = max(int(total_days / 10), 1)
        progress_delta: timedelta = timedelta(days=progress_days)
        interval_delta: timedelta = INTERVAL_DELTA_MAP[self.interval]

        start: datetime = self.start
        end: datetime = self.start + progress_delta
        progress: float = 0

        while start < self.end:
            progress_bar: str = "#" * int(progress * 10 + 1)
            self.output(_("加载进度：{} [{:.0%}]").format(progress_bar, progress))

            end = min(end, self.end)  # Make sure end time stays within set range

            if self.mode == BacktestingMode.BAR:
                data: list[BarData] | list[TickData] = load_bar_data(
                    self.symbol,
                    self.exchange,
                    self.interval,
                    start,
                    end
                )
            else:
                data = load_tick_data(
                    self.symbol,
                    self.exchange,
                    start,
                    end
                )

            self.history_data.extend(data)

            progress += progress_days / total_days
            progress = min(progress, 1)

            start = end + interval_delta
            end += progress_delta

        self.output(_("历史数据加载完成，数据量：{}").format(len(self.history_data)))

    def run_backtesting(self) -> None:
        """"""
        if self.mode == BacktestingMode.BAR:
            func: Callable[[Any], None] = self.new_bar
        else:
            func = self.new_tick

        self.strategy.on_init()
        self.strategy.inited = True
        self.output(_("策略初始化完成"))

        self.strategy.on_start()
        self.strategy.trading = True
        self.output(_("开始回放历史数据"))

        total_size: int = len(self.history_data)
        batch_size: int = max(int(total_size / 10), 1)

        for ix, i in enumerate(range(0, total_size, batch_size)):
            batch_data: list = self.history_data[i: i + batch_size]
            for data in batch_data:
                try:
                    func(data)
                except Exception:
                    self.output(_("触发异常，回测终止"))
                    self.output(traceback.format_exc())
                    return

            progress = min(ix / 10, 1)
            progress_bar: str = "=" * (ix + 1)
            self.output(_("回放进度：{} [{:.0%}]").format(progress_bar, progress))

        self.strategy.on_stop()
        self.output(_("历史数据回放结束"))

    def calculate_result(self) -> DataFrame:
        """"""
        self.output(_("开始计算逐日盯市盈亏"))

        if not self.trades:
            self.output(_("回测成交记录为空"))

        # Add trade data into daily reuslt.
        for trade in self.trades.values():
            if not trade.datetime:
                continue

            d: Date = trade.datetime.date()
            daily_result: DailyResult = self.daily_results[d]
            daily_result.add_trade(trade)

        # Calculate daily result by iteration.
        pre_close: float = 0
        start_pos: float = 0

        for daily_result in self.daily_results.values():
            daily_result.calculate_pnl(
                pre_close,
                start_pos,
                self.size,
                self.rate,
                self.slippage
            )

            pre_close = daily_result.close_price
            start_pos = daily_result.end_pos

        # Generate dataframe
        results: defaultdict = defaultdict(list)

        for daily_result in self.daily_results.values():
            for key, value in daily_result.__dict__.items():
                results[key].append(value)

        if results:
            self.daily_df = DataFrame.from_dict(results).set_index("date")

        self.output(_("逐日盯市盈亏计算完成"))
        return self.daily_df

    def calculate_statistics(
        self,
        df: DataFrame | None = None,
        output: bool = True
    ) -> dict:
        """"""
        self.output(_("开始计算策略统计指标"))

        # Check DataFrame input exterior
        if df is None:
            if self.daily_df.empty:
                self.output(_("回测结果为空，无法计算绩效统计指标"))
                return {}

            df = self.daily_df

        # Init all statistics default value
        start_date: str = ""
        end_date: str = ""
        total_days: int = 0
        profit_days: int = 0
        loss_days: int = 0
        end_balance: float = 0
        max_drawdown: float = 0
        max_ddpercent: float = 0
        max_drawdown_duration: int = 0
        total_net_pnl: float = 0
        daily_net_pnl: float = 0
        total_commission: float = 0
        daily_commission: float = 0
        total_slippage: float = 0
        daily_slippage: float = 0
        total_turnover: float = 0
        daily_turnover: float = 0
        total_trade_count: int = 0
        daily_trade_count: float = 0
        total_return: float = 0
        annual_return: float = 0
        daily_return: float = 0
        return_std: float = 0
        sharpe_ratio: float = 0
        ewm_sharpe: float = 0
        return_drawdown_ratio: float = 0
        rgr_ratio: float = 0

        # Robustness metrics (fork addition) — see robust_metrics.py
        regressed_annual_return: float = 0
        r_cubed: float = 0
        robust_sharpe: float = 0
        drawdown_episode_count: int = 0

        # Sharpe significance (fork addition) — see sharpe_inference.py.
        # Every metric above is in-sample description; these answer "could this
        # be luck?". Defaults mean "not computed", not "insignificant".
        inference: SharpeInference | None = None
        sharpe_fields: dict = {
            "sharpe_se": 0.0,
            "sharpe_tstat": 0.0,
            "sharpe_pvalue": 1.0,
            "sharpe_pvalue_one_sided": 1.0,
            "sharpe_ci_low": 0.0,
            "sharpe_ci_high": 0.0,
            "sharpe_significant": False,
            "sharpe_method": "not_computed",
            "sharpe_hac_lags": 0,
            "sharpe_hac_inflation": 0.0,
            "sharpe_bootstrap_pvalue": 1.0,
            "sharpe_required_for_significance": 0.0,
            "sharpe_skew": 0.0,
            "sharpe_kurtosis": 0.0,
            "sharpe_autocorr_lag1": 0.0,
            "sharpe_ljung_box_p": 1.0,
        }

        # Block-permutation test (fork addition) — see permutation_test.py.
        # Defaults mean "not computed": p=1.0 and significant=False, so a
        # target function that reads these keys degrades to "no evidence"
        # rather than raising KeyError when the test is switched off.
        perm_fields: dict = dict(PERMUTATION_FIELD_DEFAULTS)

        # Check if balance is always positive
        positive_balance: bool = False

        if df is not None:
            # Calculate balance related time series data
            df["balance"] = df["net_pnl"].cumsum() + self.capital

            # When balance falls below 0, set daily return to 0
            pre_balance: Series = df["balance"].shift(1)
            pre_balance.iloc[0] = self.capital
            x: Series = df["balance"] / pre_balance
            x[x <= 0] = np.nan
            df["return"] = np.log(x).fillna(0)

            df["highlevel"] = df["balance"].rolling(min_periods=1, window=len(df), center=False).max()
            df["drawdown"] = df["balance"] - df["highlevel"]
            df["ddpercent"] = df["drawdown"] / df["highlevel"] * 100

            # All balance value needs to be positive
            positive_balance = bool((df["balance"] > 0).all())
            if not positive_balance:
                self.output(_("回测中出现爆仓（资金小于等于0），无法计算策略统计指标"))

        # Calculate statistics value
        if positive_balance:
            # Calculate statistics value
            start_date = df.index[0]
            end_date = df.index[-1]

            total_days = len(df)
            profit_days = len(df[df["net_pnl"] > 0])
            loss_days = len(df[df["net_pnl"] < 0])

            end_balance = df["balance"].iloc[-1]
            max_drawdown = df["drawdown"].min()
            max_ddpercent = df["ddpercent"].min()
            max_drawdown_end = df["drawdown"].idxmin()

            if isinstance(max_drawdown_end, Date):
                max_drawdown_start = df["balance"][:max_drawdown_end].idxmax()
                max_drawdown_duration = (max_drawdown_end - max_drawdown_start).days
            else:
                max_drawdown_duration = 0

            total_net_pnl = df["net_pnl"].sum()
            daily_net_pnl = total_net_pnl / total_days

            total_commission = df["commission"].sum()
            daily_commission = total_commission / total_days

            total_slippage = df["slippage"].sum()
            daily_slippage = total_slippage / total_days

            total_turnover = df["turnover"].sum()
            daily_turnover = total_turnover / total_days

            total_trade_count = df["trade_count"].sum()
            daily_trade_count = total_trade_count / total_days

            total_return = (end_balance / self.capital - 1) * 100
            annual_return = total_return / total_days * self.annual_days
            daily_return = df["return"].mean() * 100
            return_std = df["return"].std() * 100

            if return_std:
                daily_risk_free: float = self.risk_free / np.sqrt(self.annual_days)
                sharpe_ratio = (daily_return - daily_risk_free) / return_std * np.sqrt(self.annual_days)

                ewm_window: ExponentialMovingWindow = df["return"].ewm(halflife=self.half_life)
                ewm_mean: Series = ewm_window.mean() * 100
                ewm_std: Series = ewm_window.std() * 100
                ewm_sharpe = ((ewm_mean - daily_risk_free) / ewm_std).iloc[-1] * np.sqrt(self.annual_days)
            else:
                sharpe_ratio = 0
                ewm_sharpe = 0

            if max_ddpercent:
                return_drawdown_ratio = -total_return / max_ddpercent
            else:
                return_drawdown_ratio = 0

            # Robustness metrics (fork addition). Unlike annual_return, which
            # only reads the first and last balance, these regress the whole
            # equity curve and so react to its shape. See robust_metrics.py —
            # including why RAR must not be used as an optimization target.
            robust: RobustMetrics = calculate_robust_metrics(
                balance=df["balance"].to_numpy(),
                daily_returns=df["return"].to_numpy(),
                capital=self.capital,
                annual_days=self.annual_days,
            )
            regressed_annual_return = robust.regressed_annual_return
            r_cubed = robust.r_cubed
            robust_sharpe = robust.robust_sharpe
            drawdown_episode_count = robust.drawdown_episode_count

            # Sharpe significance (fork addition). Lo (2002) standard error with
            # a Newey-West HAC correction for serial dependence, plus a
            # stationary-bootstrap second opinion. Never let a statistics
            # failure kill a backtest — degrade to the "not computed" defaults.
            if return_std:
                try:
                    inference = sharpe_inference(
                        df["return"].to_numpy(),
                        annual_days=self.annual_days,
                        # Unit care: upstream builds sharpe_ratio from
                        # daily_return / return_std, which are percent-scale
                        # (df["return"] * 100), so its daily_risk_free is also
                        # percent-scale. We feed raw df["return"], so the same
                        # rate must be divided by 100 to land in fraction scale.
                        # Without the /100 a risk_free of 2 shifts the Sharpe by
                        # ~180 units. Pinned by test_risk_free_matches_upstream.
                        risk_free_period=(
                            self.risk_free / np.sqrt(self.annual_days) / 100.0
                        ),
                        method="hac",
                        n_boot=999,
                    )
                    sharpe_fields = statistics_fields(inference)
                except (ValueError, FloatingPointError) as exc:
                    self.output(_("Sharpe 显著性检验跳过：{}").format(exc))

            # Block-permutation significance test (fork addition). Reshuffles
            # the exposure series in blocks with the market path held fixed, so
            # long-only market beta sits in both the observed value and the null
            # centre and cancels; what is left is timing skill. Costly enough to
            # be opt-in, and fail-open — a diagnostic must never kill a backtest.
            if self.permutation_enabled:
                # Engine values are defaults; explicit settings win, so a caller
                # can override size/risk_free without a duplicate-kwarg TypeError.
                perm_kwargs: dict = {
                    "size": self.size,
                    "risk_free": self.risk_free,
                    **self.permutation_settings,
                }
                perm_fields = attach_permutation_statistics(
                    perm_fields,
                    df,
                    capital=self.capital,
                    annual_days=self.annual_days,
                    **perm_kwargs,
                )

            # Calculate GRR indicator
            cagr_value: float = annual_return / 100

            if return_std > 0:
                stability_return: float = 1 / (1 + return_std / 100)
            else:
                stability_return = 0

            returns_series: Series = df["return"]
            downside_diff: np.ndarray = np.minimum(returns_series.values, 0.0)
            downside_std: float = np.sqrt(np.mean(downside_diff ** 2))
            annual_downside_risk: float = downside_std * np.sqrt(252)
            return_skew: float = cast(float, returns_series.skew())
            return_kurt: float = cast(float, returns_series.kurt())
            sorted_returns: np.ndarray = np.sort(returns_series.values)
            cutoff_index: int = int(np.ceil(len(sorted_returns) * 0.05))
            cvar_95: float = np.mean(sorted_returns[:cutoff_index])

            rgr_ratio = calc_rgr_ratio(
                cagr_value,
                stability_return,
                annual_downside_risk,
                max_ddpercent,
                return_skew,
                return_kurt,
                cvar_95
            )

        # Output
        if output:
            self.output("-" * 30)
            self.output(_("首个交易日：\t{}").format(start_date))
            self.output(_("最后交易日：\t{}").format(end_date))

            self.output(_("总交易日：\t{}").format(total_days))
            self.output(_("盈利交易日：\t{}").format(profit_days))
            self.output(_("亏损交易日：\t{}").format(loss_days))

            self.output(_("起始资金：\t{:,.2f}").format(self.capital))
            self.output(_("结束资金：\t{:,.2f}").format(end_balance))

            self.output(_("总收益率：\t{:,.2f}%").format(total_return))
            self.output(_("年化收益：\t{:,.2f}%").format(annual_return))
            self.output(_("最大回撤: \t{:,.2f}").format(max_drawdown))
            self.output(_("百分比最大回撤: {:,.2f}%").format(max_ddpercent))
            self.output(_("最大回撤天数: \t{}").format(max_drawdown_duration))

            self.output(_("总盈亏：\t{:,.2f}").format(total_net_pnl))
            self.output(_("总手续费：\t{:,.2f}").format(total_commission))
            self.output(_("总滑点：\t{:,.2f}").format(total_slippage))
            self.output(_("总成交金额：\t{:,.2f}").format(total_turnover))
            self.output(_("总成交笔数：\t{}").format(total_trade_count))

            self.output(_("日均盈亏：\t{:,.2f}").format(daily_net_pnl))
            self.output(_("日均手续费：\t{:,.2f}").format(daily_commission))
            self.output(_("日均滑点：\t{:,.2f}").format(daily_slippage))
            self.output(_("日均成交金额：\t{:,.2f}").format(daily_turnover))
            self.output(_("日均成交笔数：\t{}").format(daily_trade_count))

            self.output(_("日均收益率：\t{:,.2f}%").format(daily_return))
            self.output(_("收益标准差：\t{:,.2f}%").format(return_std))
            self.output(f"Sharpe Ratio：\t{sharpe_ratio:,.2f}")
            self.output(f"EWM Sharpe：\t{ewm_sharpe:,.2f}")
            self.output(_("收益回撤比：\t{:,.2f}").format(return_drawdown_ratio))
            self.output(f"RGR Ratio：\t{rgr_ratio:,.2f}")
            self.output(_("回归年化收益：\t{:,.2f}%").format(regressed_annual_return))
            self.output(f"R-Cubed：\t{r_cubed:,.2f}")
            self.output(f"Robust Sharpe：\t{robust_sharpe:,.2f}")
            self.output(_("回撤段数：\t{}").format(drawdown_episode_count))

            if inference is not None:
                self.output("-" * 30)
                self.output(_("Sharpe 显著性检验"))
                self.output(inference.report())

            if self.permutation_enabled:
                self.output("-" * 30)
                self.output(_("分块重排显著性检验"))
                error: object = perm_fields.get("perm_error")
                if error:
                    self.output(_("跳过：{}").format(error))
                else:
                    self.output(
                        _("统计量 {}：观测 {:,.4f}，零分布中心 {:,.4f}（std {:,.4f}）").format(
                            perm_fields["perm_statistic"],
                            perm_fields["perm_observed"],
                            perm_fields["perm_null_mean"],
                            perm_fields["perm_null_std"],
                        )
                    )
                    self.output(
                        _("p 值 {:.4f}（B={}，块长 {}）").format(
                            perm_fields["perm_p_value"],
                            perm_fields["perm_n_permutations"],
                            perm_fields["perm_block_length"],
                        )
                    )
                    if not perm_fields["perm_has_power"]:
                        self.output(_("⚠ 零分布退化，本次检验没有功效，p 值不可解读"))
                    else:
                        self.output(
                            _("80% 功效门槛：择时增量需 ≥ {:,.4f}；本次实测增量 {:,.4f}").format(
                                perm_fields["perm_min_detectable"],
                                perm_fields["perm_observed"] - perm_fields["perm_null_mean"],
                            )
                        )

        statistics: dict = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            "max_drawdown": max_drawdown,
            "max_ddpercent": max_ddpercent,
            "max_drawdown_duration": max_drawdown_duration,
            "total_net_pnl": total_net_pnl,
            "daily_net_pnl": daily_net_pnl,
            "total_commission": total_commission,
            "daily_commission": daily_commission,
            "total_slippage": total_slippage,
            "daily_slippage": daily_slippage,
            "total_turnover": total_turnover,
            "daily_turnover": daily_turnover,
            "total_trade_count": total_trade_count,
            "daily_trade_count": daily_trade_count,
            "total_return": total_return,
            "annual_return": annual_return,
            "daily_return": daily_return,
            "return_std": return_std,
            "sharpe_ratio": sharpe_ratio,
            "ewm_sharpe": ewm_sharpe,
            "return_drawdown_ratio": return_drawdown_ratio,
            "rgr_ratio": rgr_ratio,
            "regressed_annual_return": regressed_annual_return,
            "r_cubed": r_cubed,
            "robust_sharpe": robust_sharpe,
            "drawdown_episode_count": drawdown_episode_count,
            **sharpe_fields,
            **perm_fields,
        }

        # Filter potential error infinite value
        for key, value in statistics.items():
            if value in (np.inf, -np.inf):
                value = 0
            statistics[key] = np.nan_to_num(value)

        self.output(_("策略统计指标计算完成"))
        return statistics

    def show_chart(self, df: DataFrame | None = None) -> go.Figure:
        """"""
        # Check DataFrame input exterior
        if df is None:
            df = self.daily_df

        # Check for init DataFrame
        if df.empty:
            return

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Balance", "Drawdown", "Daily Pnl", "Pnl Distribution"],
            vertical_spacing=0.06
        )

        balance_line = go.Scatter(
            x=df.index,
            y=df["balance"],
            mode="lines",
            name="Balance"
        )

        drawdown_scatter = go.Scatter(
            x=df.index,
            y=df["drawdown"],
            fillcolor="red",
            fill='tozeroy',
            mode="lines",
            name="Drawdown"
        )
        pnl_bar = go.Bar(y=df["net_pnl"], name="Daily Pnl")
        pnl_histogram = go.Histogram(x=df["net_pnl"], nbinsx=100, name="Days")

        fig.add_trace(balance_line, row=1, col=1)
        fig.add_trace(drawdown_scatter, row=2, col=1)
        fig.add_trace(pnl_bar, row=3, col=1)
        fig.add_trace(pnl_histogram, row=4, col=1)

        fig.update_layout(height=1000, width=1000)
        return fig

    def _finish_optimization(
        self,
        raw_results: list,
        target_name: str,
        output: bool,
        gates: bool,
        gate_config: "OptimizationGateConfig | None",
        collect_returns: bool,
        source: str,
    ) -> "OptimizationResults":
        """寻优收尾：拆掉逐日收益、打印、跑 DSR/PBO 闸。

        返回 `OptimizationResults` —— 一个 list 子类，元素仍是原来的
        `(setting, target, statistics)` 三元组。逐日收益（`collect_returns=True`
        时 evaluate 多返回的第 4 个元素）在这里被剥掉、转成 (T, N) 矩阵喂给 PBO，
        **不进返回值的元组**，这样既有调用方（UI / get_target_value /
        deflate_optimization / `for setting, target, stats in results`）逐字不变。
        """
        # 运行期延迟 import：optimization_gates 依赖 overfitting，overfitting 依赖本模块
        from .optimization_gates import OptimizationResults, run_optimization_gates

        # 局部变量注解不在运行期求值（PEP 526），所以这里可以直接用
        # TYPE_CHECKING 下才存在的名字；方法签名里的注解必须继续加引号。
        payloads: list[ReturnsPayload | None] | None = None
        if collect_returns:
            payloads = [
                row[3] if len(row) >= 4 else None
                for row in raw_results
            ]
        results: list = [tuple(row[:3]) for row in raw_results]

        if output:
            for result in results:
                msg: str = _("参数：{}, 目标：{}").format(result[0], result[1])
                self.output(msg)

        report = None
        if gates and results:
            report = run_optimization_gates(
                results,
                target_name=target_name,
                annual_days=self.annual_days,
                capital=self.capital,
                payloads=payloads,
                config=gate_config,
                source=source,
            )
            if output:
                self.output(report.text())

        return OptimizationResults(results, gates=report)

    def run_bf_optimization(
        self,
        optimization_setting: OptimizationSetting,
        output: bool = True,
        max_workers: int | None = None,
        gates: bool = True,
        gate_config: "OptimizationGateConfig | None" = None,
        collect_returns: bool = True,
    ) -> "OptimizationResults":
        """穷举寻优。返回值是 list 子类，元素仍是 (setting, target, statistics)。

        gates           跑完后自动算 DSR 与 PBO，挂在返回值的 `.gates` 上。
                        扫参数是多重比较真正发生的地方，单次回测的 t 检验与分块
                        重排在这里全部失效，所以默认开。关掉只在"我确实只想要
                        参数表"时才合理。零分布模拟约 0.09 秒 × n_null_sims，
                        嫌慢就传 `gate_config=OptimizationGateConfig(n_null_sims=0)`，
                        而不是把整道闸关掉。
        collect_returns 让每个子进程把已经算出来的逐日盈亏带回来（约 14 KiB/组）。
                        **这是 PBO 的唯一数据来源** —— 关掉则 PBO 无法计算，
                        报告里会如实写"未收集"，不做任何替代估计。
        """
        if not check_optimization_setting(optimization_setting):
            from .optimization_gates import OptimizationResults
            return OptimizationResults()

        evaluate_func: Callable = wrap_evaluate(
            self, optimization_setting.target_name, collect_returns=collect_returns
        )
        raw_results: list = run_bf_optimization(
            evaluate_func,
            optimization_setting,
            get_target_value,
            max_workers=max_workers,
            output=self.output
        )

        return self._finish_optimization(
            raw_results,
            target_name=optimization_setting.target_name,
            output=output,
            gates=gates,
            gate_config=gate_config,
            collect_returns=collect_returns,
            source="bf",
        )

    run_optimization = run_bf_optimization

    def run_ga_optimization(
        self,
        optimization_setting: OptimizationSetting,
        output: bool = True,
        max_workers: int | None = None,
        pop_size: int = 100,
        ngen: int = 30,
        mu: int | None = None,
        lambda_: int | None = None,
        cxpb: float = 0.95,
        mutpb: float | None = None,
        indpb: float = 1.0,
        gates: bool = True,
        gate_config: "OptimizationGateConfig | None" = None,
        collect_returns: bool = True,
    ) -> "OptimizationResults":
        """遗传寻优。参数与返回值语义同 `run_bf_optimization`。

        ⚠️ GA 的 DSR 系统性偏乐观：遗传搜索向高适应度收敛，缓存里的试验夏普被
        截尾，试验间 std 偏小 → 零技能最大夏普期望 SR* 偏低 → DSR 偏高。
        闸会把这条写进 `report.notes`，但它无法自动修正 —— 要拿 GA 的 DSR 当
        验收闸，必须另用穷举在同一参数空间采样估 std。
        """
        if not check_optimization_setting(optimization_setting):
            from .optimization_gates import OptimizationResults
            return OptimizationResults()

        evaluate_func: Callable = wrap_evaluate(
            self, optimization_setting.target_name, collect_returns=collect_returns
        )
        raw_results: list = run_ga_optimization(
            evaluate_func,
            optimization_setting,
            get_target_value,
            max_workers=max_workers,
            pop_size=pop_size,
            ngen=ngen,
            mu=mu,
            lambda_=lambda_,
            cxpb=cxpb,
            mutpb=mutpb,
            indpb=indpb,
            output=self.output
        )

        return self._finish_optimization(
            raw_results,
            target_name=optimization_setting.target_name,
            output=output,
            gates=gates,
            gate_config=gate_config,
            collect_returns=collect_returns,
            source="ga",
        )

    def update_daily_close(self, price: float) -> None:
        """"""
        d: Date = self.datetime.date()

        daily_result: DailyResult | None = self.daily_results.get(d, None)
        if daily_result:
            daily_result.close_price = price
        else:
            self.daily_results[d] = DailyResult(d, price)

    def new_bar(self, bar: BarData) -> None:
        """"""
        self.bar = bar
        self.datetime = bar.datetime

        self.cross_limit_order()
        self.cross_stop_order()
        self.strategy.on_bar(bar)

        self.update_daily_close(bar.close_price)

    def new_tick(self, tick: TickData) -> None:
        """"""
        self.tick = tick
        self.datetime = tick.datetime

        self.cross_limit_order()
        self.cross_stop_order()
        self.strategy.on_tick(tick)

        self.update_daily_close(tick.last_price)

    def cross_limit_order(self) -> None:
        """
        Cross limit order with last bar/tick data.
        """
        if self.mode == BacktestingMode.BAR:
            long_cross_price = self.bar.low_price
            short_cross_price = self.bar.high_price
            long_best_price = self.bar.open_price
            short_best_price = self.bar.open_price
        else:
            long_cross_price = self.tick.ask_price_1
            short_cross_price = self.tick.bid_price_1
            long_best_price = long_cross_price
            short_best_price = short_cross_price

        for order in list(self.active_limit_orders.values()):
            # Push order update with status "not traded" (pending).
            if order.status == Status.SUBMITTING:
                order.status = Status.NOTTRADED
                self.strategy.on_order(order)

            # Check whether limit orders can be filled.
            long_cross: bool = (
                order.direction == Direction.LONG
                and order.price >= long_cross_price
                and long_cross_price > 0
            )

            short_cross: bool = (
                order.direction == Direction.SHORT
                and order.price <= short_cross_price
                and short_cross_price > 0
            )

            if not long_cross and not short_cross:
                continue

            # Push order udpate with status "all traded" (filled).
            order.traded = order.volume
            order.status = Status.ALLTRADED
            self.strategy.on_order(order)

            if order.vt_orderid in self.active_limit_orders:
                self.active_limit_orders.pop(order.vt_orderid)

            # Push trade update
            self.trade_count += 1

            if long_cross:
                trade_price = min(order.price, long_best_price)
                pos_change = order.volume
            else:
                trade_price = max(order.price, short_best_price)
                pos_change = -order.volume

            trade: TradeData = TradeData(
                symbol=order.symbol,
                exchange=order.exchange,
                orderid=order.orderid,
                tradeid=str(self.trade_count),
                direction=order.direction,
                offset=order.offset,
                price=trade_price,
                volume=order.volume,
                datetime=self.datetime,
                gateway_name=self.gateway_name,
            )

            self.strategy.pos += pos_change
            self.strategy.on_trade(trade)

            self.trades[trade.vt_tradeid] = trade

    def cross_stop_order(self) -> None:
        """
        Cross stop order with last bar/tick data.
        """
        if self.mode == BacktestingMode.BAR:
            long_cross_price = self.bar.high_price
            short_cross_price = self.bar.low_price
            long_best_price = self.bar.open_price
            short_best_price = self.bar.open_price
        else:
            long_cross_price = self.tick.last_price
            short_cross_price = self.tick.last_price
            long_best_price = long_cross_price
            short_best_price = short_cross_price

        for stop_order in list(self.active_stop_orders.values()):
            # Check whether stop order can be triggered.
            long_cross: bool = (
                stop_order.direction == Direction.LONG
                and stop_order.price <= long_cross_price
            )

            short_cross: bool = (
                stop_order.direction == Direction.SHORT
                and stop_order.price >= short_cross_price
            )

            if not long_cross and not short_cross:
                continue

            # Create order data.
            self.limit_order_count += 1

            order: OrderData = OrderData(
                symbol=self.symbol,
                exchange=self.exchange,
                orderid=str(self.limit_order_count),
                direction=stop_order.direction,
                offset=stop_order.offset,
                price=stop_order.price,
                volume=stop_order.volume,
                traded=stop_order.volume,
                status=Status.ALLTRADED,
                gateway_name=self.gateway_name,
                datetime=self.datetime
            )

            self.limit_orders[order.vt_orderid] = order

            # Create trade data.
            if long_cross:
                trade_price = max(stop_order.price, long_best_price)
                pos_change = order.volume
            else:
                trade_price = min(stop_order.price, short_best_price)
                pos_change = -order.volume

            self.trade_count += 1

            trade: TradeData = TradeData(
                symbol=order.symbol,
                exchange=order.exchange,
                orderid=order.orderid,
                tradeid=str(self.trade_count),
                direction=order.direction,
                offset=order.offset,
                price=trade_price,
                volume=order.volume,
                datetime=self.datetime,
                gateway_name=self.gateway_name,
            )

            self.trades[trade.vt_tradeid] = trade

            # Update stop order.
            stop_order.vt_orderids.append(order.vt_orderid)
            stop_order.status = StopOrderStatus.TRIGGERED

            if stop_order.stop_orderid in self.active_stop_orders:
                self.active_stop_orders.pop(stop_order.stop_orderid)

            # Push update to strategy.
            self.strategy.on_stop_order(stop_order)
            self.strategy.on_order(order)

            self.strategy.pos += pos_change
            self.strategy.on_trade(trade)

    def load_bar(
        self,
        vt_symbol: str,
        days: int,
        interval: Interval,
        callback: Callable,
        use_database: bool
    ) -> list[BarData]:
        """"""
        self.callback = callback

        init_end = self.start - INTERVAL_DELTA_MAP[interval]
        init_start = self.start - timedelta(days=days)

        symbol, exchange = extract_vt_symbol(vt_symbol)

        bars: list[BarData] = load_bar_data(
            symbol,
            exchange,
            interval,
            init_start,
            init_end
        )

        return bars

    def load_tick(self, vt_symbol: str, days: int, callback: Callable) -> list[TickData]:
        """"""
        self.callback = callback

        init_end = self.start - timedelta(seconds=1)
        init_start = self.start - timedelta(days=days)

        symbol, exchange = extract_vt_symbol(vt_symbol)

        ticks: list[TickData] = load_tick_data(
            symbol,
            exchange,
            init_start,
            init_end
        )

        return ticks

    def send_order(
        self,
        strategy: CtaTemplate,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        stop: bool,
        lock: bool,
        net: bool
    ) -> list:
        """"""
        price = round_to(price, self.pricetick)
        if stop:
            vt_orderid: str = self.send_stop_order(direction, offset, price, volume)
        else:
            vt_orderid = self.send_limit_order(direction, offset, price, volume)
        return [vt_orderid]

    def send_stop_order(
        self,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float
    ) -> str:
        """"""
        self.stop_order_count += 1

        stop_order: StopOrder = StopOrder(
            vt_symbol=self.vt_symbol,
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            datetime=self.datetime,
            stop_orderid=f"{STOPORDER_PREFIX}.{self.stop_order_count}",
            strategy_name=self.strategy.strategy_name,
        )

        self.active_stop_orders[stop_order.stop_orderid] = stop_order
        self.stop_orders[stop_order.stop_orderid] = stop_order

        return stop_order.stop_orderid

    def send_limit_order(
        self,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float
    ) -> str:
        """"""
        self.limit_order_count += 1

        order: OrderData = OrderData(
            symbol=self.symbol,
            exchange=self.exchange,
            orderid=str(self.limit_order_count),
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=Status.SUBMITTING,
            gateway_name=self.gateway_name,
            datetime=self.datetime
        )

        self.active_limit_orders[order.vt_orderid] = order
        self.limit_orders[order.vt_orderid] = order

        return order.vt_orderid

    def cancel_order(self, strategy: CtaTemplate, vt_orderid: str) -> None:
        """
        Cancel order by vt_orderid.
        """
        if vt_orderid.startswith(STOPORDER_PREFIX):
            self.cancel_stop_order(strategy, vt_orderid)
        else:
            self.cancel_limit_order(strategy, vt_orderid)

    def cancel_stop_order(self, strategy: CtaTemplate, vt_orderid: str) -> None:
        """"""
        if vt_orderid not in self.active_stop_orders:
            return
        stop_order: StopOrder = self.active_stop_orders.pop(vt_orderid)

        stop_order.status = StopOrderStatus.CANCELLED
        self.strategy.on_stop_order(stop_order)

    def cancel_limit_order(self, strategy: CtaTemplate, vt_orderid: str) -> None:
        """"""
        if vt_orderid not in self.active_limit_orders:
            return
        order: OrderData = self.active_limit_orders.pop(vt_orderid)

        order.status = Status.CANCELLED
        self.strategy.on_order(order)

    def cancel_all(self, strategy: CtaTemplate) -> None:
        """
        Cancel all orders, both limit and stop.
        """
        vt_orderids: list = list(self.active_limit_orders.keys())
        for vt_orderid in vt_orderids:
            self.cancel_limit_order(strategy, vt_orderid)

        stop_orderids: list = list(self.active_stop_orders.keys())
        for vt_orderid in stop_orderids:
            self.cancel_stop_order(strategy, vt_orderid)

    def write_log(self, msg: str, strategy: CtaTemplate | None = None) -> None:
        """
        Write log message.
        """
        msg = f"{self.datetime}\t{msg}"
        self.logs.append(msg)

    def send_notification(self, msg: str, strategy: CtaTemplate | None = None) -> None:
        """
        Push notification through all configured channels.
        """
        pass

    send_email = send_notification

    def sync_strategy_data(self, strategy: CtaTemplate) -> None:
        """
        Sync strategy data into json file.
        """
        pass

    def get_engine_type(self) -> EngineType:
        """
        Return engine type.
        """
        return self.engine_type

    def get_pricetick(self, strategy: CtaTemplate) -> float:
        """
        Return contract pricetick data.
        """
        return self.pricetick

    def get_size(self, strategy: CtaTemplate) -> float:
        """
        Return contract size data.
        """
        return self.size

    def put_strategy_event(self, strategy: CtaTemplate) -> None:
        """
        Put an event to update strategy status.
        """
        pass

    def output(self, msg: str) -> None:
        """
        Output message of backtesting engine.
        """
        print(f"{datetime.now()}\t{msg}")

    def get_all_trades(self) -> list:
        """
        Return all trade data of current backtesting result.
        """
        return list(self.trades.values())

    def get_all_orders(self) -> list:
        """
        Return all limit order data of current backtesting result.
        """
        return list(self.limit_orders.values())

    def get_all_daily_results(self) -> list:
        """
        Return all daily result data.
        """
        return list(self.daily_results.values())


class DailyResult:
    """"""

    def __init__(self, date: Date, close_price: float) -> None:
        """"""
        self.date: Date = date
        self.close_price: float = close_price
        self.pre_close: float = 0

        self.trades: list[TradeData] = []
        self.trade_count: int = 0

        self.start_pos: float = 0
        self.end_pos: float = 0

        self.turnover: float = 0
        self.commission: float = 0
        self.slippage: float = 0

        self.trading_pnl: float = 0
        self.holding_pnl: float = 0
        self.total_pnl: float = 0
        self.net_pnl: float = 0

    def add_trade(self, trade: TradeData) -> None:
        """"""
        self.trades.append(trade)

    def calculate_pnl(
        self,
        pre_close: float,
        start_pos: float,
        size: float,
        rate: float,
        slippage: float
    ) -> None:
        """"""
        # If no pre_close provided on the first day,
        # use value 1 to avoid zero division error
        if pre_close:
            self.pre_close = pre_close
        else:
            self.pre_close = 1

        # Holding pnl is the pnl from holding position at day start
        self.start_pos = start_pos
        self.end_pos = start_pos

        self.holding_pnl = self.start_pos * (self.close_price - self.pre_close) * size

        # Trading pnl is the pnl from new trade during the day
        self.trade_count = len(self.trades)

        for trade in self.trades:
            if trade.direction == Direction.LONG:
                pos_change = trade.volume
            else:
                pos_change = -trade.volume

            self.end_pos += pos_change

            turnover: float = trade.volume * size * trade.price
            self.trading_pnl += pos_change * \
                (self.close_price - trade.price) * size
            self.slippage += trade.volume * size * slippage

            self.turnover += turnover
            self.commission += turnover * rate

        # Net pnl takes account of commission and slippage cost
        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission - self.slippage


@lru_cache(maxsize=999)
def load_bar_data(
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    start: datetime,
    end: datetime
) -> list[BarData]:
    """"""
    database: BaseDatabase = get_database()

    return database.load_bar_data(symbol, exchange, interval, start, end)


@lru_cache(maxsize=999)
def load_tick_data(
    symbol: str,
    exchange: Exchange,
    start: datetime,
    end: datetime
) -> list[TickData]:
    """"""
    database: BaseDatabase = get_database()

    return database.load_tick_data(symbol, exchange, start, end)


def evaluate(
    target_name: str,
    strategy_class: type[CtaTemplate],
    vt_symbol: str,
    interval: Interval,
    start: datetime,
    rate: float,
    slippage: float,
    size: float,
    pricetick: float,
    capital: int,
    end: datetime,
    mode: BacktestingMode,
    setting: dict,
    risk_free: float = 0,
    annual_days: int = 240,
    half_life: int = 120,
    collect_returns: bool = False
) -> tuple:
    """
    Function for running in multiprocessing.pool

    The four trailing parameters are keyword-only in practice: wrap_evaluate
    binds everything up to `mode` positionally, the optimizer appends `setting`,
    and these arrive as keywords. Adding them at the tail (rather than before
    `setting`) keeps that positional binding impossible to get wrong.

    risk_free / annual_days / half_life used to be dropped on the floor here:
    the child process always fell back to BacktestingEngine's defaults
    (0 / 240 / 120) no matter what the parent engine was configured with, so an
    HK grid search set to 247 trading days was silently ranked on 240. They are
    forwarded now, defaulting to exactly those old fallbacks.

    collect_returns adds a 4th element to the returned tuple: the per-day mark
    -to-market series `(dates, net_pnl)` that calculate_result() computes and
    that this function used to throw away. It is the only way to get the (T, N)
    return matrix CSCV/PBO needs, and it costs ~14 KiB per parameter set.
    """
    engine: BacktestingEngine = BacktestingEngine()

    engine.set_parameters(
        vt_symbol=vt_symbol,
        interval=interval,
        start=start,
        rate=rate,
        slippage=slippage,
        size=size,
        pricetick=pricetick,
        capital=capital,
        end=end,
        mode=mode,
        risk_free=risk_free,
        annual_days=annual_days,
        half_life=half_life
    )

    engine.add_strategy(strategy_class, setting)
    engine.load_data()
    engine.run_backtesting()
    df: DataFrame = engine.calculate_result()
    statistics: dict = engine.calculate_statistics(output=False)

    target_value: float = statistics.get(target_name, 0)
    if not collect_returns:
        return (setting, target_value, statistics)

    if df is None or df.empty or "net_pnl" not in df.columns:
        payload: tuple = ([], np.zeros(0, dtype=float))
    else:
        payload = (list(df.index), df["net_pnl"].to_numpy(dtype=float))
    return (setting, target_value, statistics, payload)


def wrap_evaluate(
    engine: BacktestingEngine,
    target_name: str,
    collect_returns: bool = False
) -> Callable:
    """
    Wrap evaluate function with given setting from backtesting engine.
    """
    func: Callable = partial(
        evaluate,
        target_name,
        engine.strategy_class,
        engine.vt_symbol,
        engine.interval,
        engine.start,
        engine.rate,
        engine.slippage,
        engine.size,
        engine.pricetick,
        engine.capital,
        engine.end,
        engine.mode,
        risk_free=engine.risk_free,
        annual_days=engine.annual_days,
        half_life=engine.half_life,
        collect_returns=collect_returns
    )
    return func


def calc_rgr_ratio(
    cagr_value: float,
    stability_return: float,
    annual_downside_risk: float,
    max_drawdown_percent: float,
    return_skew: float,
    return_kurt: float,
    c_var: float
) -> float:
    """"""
    # Apply log for diminishing marginal utility
    if cagr_value > 0:
        gain: float = np.log(1 + cagr_value)
    else:
        gain = -np.log(1 - cagr_value)

    # Skewness adjustment factor
    skew_factor: float = 1 + 0.1 * np.tanh(return_skew)

    # Kurtosis penalty for fat tail
    kurt_factor: float = 1 / (1 + 0.05 * max(return_kurt - 3, 0))

    # Combined risk calculation
    downside_risk: float = max(annual_downside_risk, 1e-6)
    max_dd: float = abs(max_drawdown_percent) / 100.0
    if c_var != 0:
        cvar_risk: float = abs(c_var)
    else:
        cvar_risk = max_dd * 0.5
    combined_risk: float = 0.5 * downside_risk + 0.3 * max_dd + 0.2 * cvar_risk

    # Prevent division by zero
    if combined_risk < 1e-9:
        combined_risk = 1e-9

    # Final RGR calculation
    rgr_ratio: float = (gain * stability_return * skew_factor * kurt_factor) / combined_risk

    return rgr_ratio


def get_target_value(result: list | tuple) -> float:
    """
    Get target value for sorting optimization results.
    """
    return cast(float, result[1])
