# Fork 说明

这是 [vnpy/vnpy_ctastrategy](https://github.com/vnpy/vnpy_ctastrategy) 的 fork(MIT)。

## 为什么 fork

本项目已经在直接依赖它:`vnpy_app/strategies/long_only_turtle_strategy.py`
继承 `CtaTemplate`,`run_gui.py` 加载 `CtaStrategyApp`。而我们需要改动它本身:

1. **回测评价指标不够** —— `calculate_statistics` 只有端点法 `annual_return`、
   `sharpe_ratio`、`return_drawdown_ratio`。缺少对"曲线形状"敏感的稳健性指标
   (回归年化收益及其衍生比率),而本项目已经吃过亏:某策略样本外 −19.5%
   却没有任何统计检验拦住它。
2. **开平仓语义是期货的** —— `Offset.OPEN/CLOSE/CLOSETODAY` 来自期货保证金
   交易。本项目只做**港股与美股**,是净持仓、无平今概念。

三方包装在 `.venv` 里无法修改,也不该修改(会被升级覆盖)。

## 与上游的关系

- 远端 `upstream` 指向上游仓库,`origin` 指向本项目 fork。
- **同步基点**:`6ef7698`(上游 v1.4.1,2026-05-06)。
- 同步方式:`git fetch upstream && git merge upstream/main`。
- 原则:**改动尽量克制、集中、可解释**,以便合并上游时冲突面最小。
  新增能力优先放进独立文件,而不是散落着改上游函数体。

## 本 fork 的改动

改动记录在此处,便于同步上游时逐条复核。

（尚无功能性改动;当前仅新增本文件与 pyright 配置。）
