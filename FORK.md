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

### 1. 稳健性业绩指标 RAR / R-Cubed / Robust Sharpe

来源:社区帖 <https://www.vnpy.com/forum/topic/32894>

- **新增** `vnpy_ctastrategy/robust_metrics.py`(独立文件,与上游无重叠)
- **改动** `backtesting.py`:仅 4 处最小接入 —— import 一行、变量初始化、
  `calculate_statistics` 里调用一次、输出与 statistics 字典各加四行。
  算法全部在新文件里,同步上游时冲突面仅限这几行。

三个指标的共同点:常规 `annual_return` 只看首尾净值,一条"最后几天暴涨拉起来"
的曲线与稳步上行的曲线会给出相同数字;这三个改为对累计收益曲线做回归,对形状敏感。

实测(700.SEHK 2024-01~2026-07,611 日):
端点年化 3.14% vs 回归年化 3.44%(差 +0.30pp,收益偏前段),回撤段数 18。

**必读的两条性质**(已写进模块 docstring 并有测试钉住):
- RAR 对单期收益的隐含权重 ∝ (n²−j²),**随时间单调递减到接近零**
  (n=250 时首期 1.50×、末期 0.012×)。它系统性低估最近期表现,
  因此**不可用作参数寻优目标**,也不适合单独用来判断"近期是否仍然有效"。
- R³ 的分母(前 5 大回撤的平均)必然小于最大回撤,所以在分母层面比
  `return_drawdown_ratio` 宽松;"稳健"指估计量方差小,不是结论更保守。
  但两者分子口径不同(RAR 年化 vs 总收益),**不可直接比大小**。

三者都是样本内描述统计,**不是统计检验**,不回答"这是不是运气"。
