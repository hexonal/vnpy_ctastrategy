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

### 2. 样本内外三段切分 TRAIN / VALID / TEST

对齐 vnpy.alpha 与官方投研系列第 3、4 篇（AlphaDataset 构造即划分三段、
`AlphaModel.predict(dataset, segment)` 强制显式传段）。

- **新增** `vnpy_ctastrategy/segments.py`(独立文件)
- **`backtesting.py` 零修改** —— 三段能力全部长在新文件里。
  引擎本身没有插入点(`set_parameters` 只有单段 start/end、`run_backtesting`
  是单趟回放),切分只可能发生在"喂什么起止"这一层,
  这正是 `overfitting.EngineRunner` 已经在做的事,本模块沿用同一形态。
- **新增** `tests/test_segment_split.py`(46 例)

补的缺口:`robust_metrics` / `overfitting` / `deflated_sharpe` 三套统计闸
**都不知道自己算的是哪一段** —— 样本内的 `sharpe_significant=True`
与样本外的长得一模一样。

四件东西:

1. `Segment` / `ThreeWaySplit` / `make_three_way_split` / `split_by_ratio`
   —— 边界取自真实 K 线时间戳,`__post_init__` 强制
   `train_end < valid_start` 且 `valid_end < test_start`(三段不共享任何一根 K 线)。
2. `SegmentedRunner` —— 包住 `BacktestRunner`,按段取数;
   每段的 statistics 注入 `segment` 与 `is_out_of_sample` 两个键,
   daily_df 打 `attrs["segment"]`。
3. `run_holdout` —— TRAIN 扫网格选参 → VALID 复核(样本内)→ TEST 只跑一次;
   TEST 段附 block-bootstrap 显著性。
4. `SegmentGuardedEngine` —— `BacktestingEngine` 子类,
   `use_segment(segment, ...)` 让窗口由切分决定(消灭"改了段忘了改日期"),
   两个寻优入口在窗口与 TEST 段有交集时抛 `SegmentLeakError`。
   两个覆写是纯 `*args/**kwargs` 透传,不抄父类签名 ——
   否则父类新增的 `gates` / `collect_returns` 会被静默挡在门外。

**VALID 归样本内**(官方原文:"只要我们根据验证集表现调整因子、调参数、
选择训练轮数或更换模型,它就已经参与了研究决策")。
`is_out_of_sample()` 只对 TEST 返回 True。

三道闸(都是 exit 码,不是注释):
- 在 TEST 上扫参数 → `SegmentLeakError`(`SegmentedRunner.scan` 与
  `SegmentGuardedEngine` 两条路都堵)
- 第二次查看 TEST → `SegmentBudgetExhaustedError`,每次查看进 `test_audit`;
  重开必须 `reset_test_budget(reason=...)`,reason 为空被拒
- 取某段结果的 API,`segment` 一律无默认值(抄官方 `predict(dataset, segment)`)

实测(700.SEHK 734 日线、DoubleMaStrategy 6 组网格、420/140/140 切分):
TRAIN sharpe 0.5006 / VALID 0.5963 / **TEST −0.2495**(p_block=0.85 不显著)。
样本内两段都是正的,样本外翻负 —— 这正是本层要让人看见的东西。

**诚实边界**:一次性 holdout 不能替代 Walk-Forward。它只有一段样本外、
一个参数集,回答不了"参数是否随时间漂移"。它便宜(1×网格+2 次回测),
便宜的用途是"别因为太贵就干脆不看样本外",不是 WF 的替代品。

### 3. 参数寻优的多重比较闸 DSR + PBO

补的缺口:单次回测已经有 t 检验(`sharpe_inference`)与分块重排
(`permutation_test`),但**一旦开始扫参数就裸奔** —— 而扫参数正是多重比较发生的
地方。25 组参数取最好的一组,等于做了 25 次检验只报最显著的那次;
它的 t 值、p 值、Sharpe 全部被选择偏差污染。

- **新增** `vnpy_ctastrategy/optimization_gates.py`(独立文件,闸的全部逻辑在这)
- **新增** `tests/test_optimization_gates.py`(25 例,含负对照/正对照对拍)
- **改动** `backtesting.py`:四处
  1. `evaluate()` 尾部加 4 个带默认值的参数(`risk_free` / `annual_days` /
     `half_life` / `collect_returns`),`wrap_evaluate()` 以**关键字**传入
     (partial 是位置绑定,加在 `setting` 之后 + 关键字传 = 位置错位不可能发生);
  2. `run_bf_optimization` / `run_ga_optimization` 加 3 个带默认值的关键字参数
     (`gates` / `gate_config` / `collect_returns`);
  3. 新增私有方法 `_finish_optimization()`,把两个入口的收尾逻辑合并;
  4. `TYPE_CHECKING` 块(闸模块 → overfitting → backtesting 是真实导入环,
     运行期 import 只能放在方法体内)。
- **改动** `overfitting.py`:抽出 `pbo_from_matrix()`,`pbo_from_settings()`
  改为委托给它(行为逐字不变,由 `test_pbo_from_matrix_matches_pbo_from_settings`
  钉住)。新入口让"已有收益矩阵"的场景不必重跑网格。

**返回值向后兼容是硬要求**:`OptimizationResults` 是 `list` 子类,
元素仍是按目标值降序的 `(setting, target, statistics)` 三元组,
`len` / 下标 / 解包 / 迭代 / `get_target_value` / `deflate_optimization` 全部不变;
两道闸的结果挂在**新增属性** `.gates` 上。逐日收益(`collect_returns=True` 时
`evaluate` 多返回的第 4 个元素)在 `_finish_optimization` 里就被剥掉,
不进返回值的元组。

顺带修掉的一个真 bug:`wrap_evaluate` 原先丢掉 `annual_days` / `risk_free` /
`half_life`,子进程一律用 `BacktestingEngine` 的默认值(240 / 0 / 120)。
港股设 247 时,寻优排出来的 sharpe 是 0.759290881(按 240 算),
而单跑 247 是 0.770284289 —— 整个排名、DSR 的年化折算、`ewm_sharpe`
都建在错的年化基数上。现已透传,`test_wrap_evaluate_forwards_engine_annual_days`
钉住。

PBO 的数据来源:寻优流程自己产出。CSCV 需要 (T, N) 收益矩阵,而寻优只返回标量
statistics —— 每组参数的 `daily_df` 在子进程里算完就被丢掉。`overfitting_audit`
绕过这一点的办法是用 `EngineRunner` 把网格**重跑一遍**,那是一条与寻优平行的
代码路径(分块查询 vs 单次查询,取到的 K 线根数都可能不同)。这里改为让
`evaluate` 把已经算出来的逐日盈亏带回来(约 14 KiB/组),PBO 因此测的是
**寻优当时那一批回测**。

判据(两道闸都是否决工具,不是背书工具):
- `report.dsr_value` 取 headline 与详版的**较小者**(详版用真实 γ₃/γ₄,
  真实收益多为负偏厚尾,正态假设会高估显著性,拿不准就用低的那个)
- 任一闸算不出来 = `passed` 为 False。**缺证据不是证据** ——
  非夏普族目标 / 头名爆仓 / 未收集逐日收益,一律记 None + 备注,绝不默认放行
- GA 的 DSR 系统性偏乐观(收敛截尾 → 试验 std 偏小 → SR* 偏低),
  闸无法自动修正,强制写进 `notes`

实测(700.SEHK 2023-07~2026-07 共 729 日、DoubleMaStrategy 25 组网格、
`annual_days=247`、真多进程):

    最优参数 fast=10 / slow=30,年化 Sharpe 0.770
    PSR(0) = 0.902   ← 只看这一条曲线,"显著"
    DSR    = 0.511   ← 扣掉"试了 25 组",不显著(SR* = 0.75,几乎等于观测值)
    PBO    = 0.653   ← 高于零分布中位 0.474,p = 0.822;选参【反向有害】

也就是说:这个网格挑出来的参数,样本内看着能打,实际上整条"挑参数"的流程
不携带任何信息。这正是本层要拦下的东西。

**诚实边界**:本项目的典型样本(单标的 600-730 日线、十几到几十笔完整交易)
处于低功效区 —— 上面那份报告自己就写着"能以 80% 功效检出的最小真实 Sharpe ≈ 2.32"。
低功效下 PBO 天然向 0.5 靠拢、DSR 天然偏低,此时"不显著"的正确读法是
**"数据不足以支持从这个网格里挑参数"**,而不是"策略一定是假的"。
两种情况的处置相同(别用寻优出来的参数),结论不同。

### 4. `load_data` 的时区 / 分块两处取数 bug

**症状(本机 QuestDB,`database.timezone` = UTC,机器在美东)**:同一区间、
同一份数据,取到的 K 线根数不一样 ——

| 取法 | 边界 | 根数 |
|---|---|---|
| 库中实有(`get_bar_overview`) | — | 700 |
| `database.load_bar_data` 单次 | 裸 datetime | 699 |
| `BacktestingEngine.load_data` 分块 | 裸 datetime | 693 |
| 两者 | tz-aware UTC | 700 / 700 |

`load_bar_data('2024-01-26', '2024-01-26 23:59')` 返回 0 根,而那天库里有 bar。

**三个独立病根**:

1. **裸 datetime 被按机器时区解读。** 所有 database driver 的查询边界都过
   `vnpy.trader.database.convert_tz`,它调 `datetime.astimezone()`;对裸
   datetime 这个方法把值读成**宿主机**本地时间。于是同一个
   `datetime(2024,1,26)` 在美西笔记本和港股服务器上是不同的时刻,窗口整体
   平移一个 UTC 偏移量,边界上的 bar 掉出去。
2. **分块循环把块与块之间的缝隙跳过了。** 分块查询两端都是闭区间,而循环
   用 `start = end + interval_delta` 前进 —— 时间戳落在
   `(end, end + interval_delta)` 开区间里的 bar,两个块都没查。**只要 bar
   的标签不正好落在块边界上就会丢**,与时区无关(实测 700.SEHK 用
   tz-aware UTC 边界照丢 7 根)。
3. **不足一个自然日的窗口 ZeroDivisionError** —— `progress_days / total_days`
   没有防 `total_days == 0`。

**改动**(三处,均在 fork 侧,未动 `vnpy/`):

- **新增** `vnpy_gatewaykit/query_window.py`:`localize_bound(moment, exchange)`
  把裸边界读成**交易所自己的墙钟**,时区取自
  `vnpy_gatewaykit.market_clock.market_tz` —— 这是本项目"交易所 → 时区"的
  单一真相源,gateway 写入 bar 时用的就是它,两侧因此对齐同一口时钟。
  market_clock 没映射的交易所(上游 CFFEX/SHFE 等)退回 `DB_TZ`,即配置项
  `database.timezone`;那仍是**声明出来的**时区,不是宿主机碰巧所在的时区,
  且在默认安装(`database.timezone` 保持机器时区)下与原行为逐位一致。
  (先落在本包内,后下沉到 gatewaykit:`vnpy_replay` 的回放窗口与 `vnpy_app`
  的数据管理/复盘图有同一个病,而 gateway 基础设施与 GUI 都不该为了一个时区
  读法去依赖一个策略应用包。三个包本来就都依赖 gatewaykit。)
- **改动** `backtesting.py` 模块级 `load_bar_data` / `load_tick_data`:调
  database 前过 `localize_bound`。放在这一层而不是各调用点,是因为
  `load_data` 的分块循环、`load_bar` 的预热窗口、`overfitting` 的整段缓存
  三条路径共用它,一处改完三处都正。
- **改动** `load_data` 循环:块推进改成 `start = end`(相邻块共享缝隙时刻),
  靠记住上一块最后一根的时间戳去重,而不是把下一块的起点推过缝隙;
  `total_days` 加 1 天地板。

**新增依赖** `vnpy_gatewaykit>=0.1.0`(pyproject)。把那张时区表在本包里再抄
一份会让"2024-01-26 是什么时刻"有两个真相源,正是本改动要消除的那类 bug。

**测试** `tests/test_load_data_timezone.py`(11 条):用内存 database double
复现,不需要 QuestDB;double 用真的 `convert_tz` 过滤,所以第 1 条病根走的是
和真 driver 完全相同的代码路径。`TZ` 固定成美西,保证宿主机时区与市场时区
必然不同 —— 否则断言全部落空。回归前实测 5 red(693 / 699 / 0 / ZeroDivision),
修复后 11 green。
