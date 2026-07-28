# A + B → C:社会影响的行为校准与迁移

把三个已实现的实验环境转成一个校准实验:在 A、B 上训练"像人",零样本迁移到 C。

- **A / `study_016`**:信息级联。顺序观察他人的公开选择,再结合自己的私有信号。
- **B / `study_017` Task 3C**:顾问选择。从完整反馈历史里区分"准确"和"只会同意你"。
- **C / `study_019` Study 2**:医疗权威。顺序诊断中加入 `medical director` 的权威身份。
- **D / `study_019` Study 1**:urn 级联。和 C 同一批被试,但**没有**权威操纵。

完整实验设计见 [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)。

## 这是校准问题,不是正确性问题

目标是匹配**人类反应分布**,不是给出贝叶斯最优答案。主指标是
**MAE(模型分布 vs 人类分布)**,准确率只作次要指标——因为"更像人"和"更理性"
经常是相反方向,合成一个数会把这件事藏起来。

每一行数据都带着它那个状态的人类比例,全部来自原论文:

| 效应 | 人类比例粒度 | 来源 |
|---|---|---|
| A | bucket 级(4 个已校准 bucket,覆盖 126 个状态中的 114 个) | Anderson & Holt 1997 正文统计 |
| B | 实验级(3C = 0.17 agreeing pick rate) | Jaquiery & Yeung 2024 |
| C | **逐场景**(40 场景 × n=40) | Figshare 原始 workbook |
| D | **逐场景**(24 场景 × n≈40) | 同上,Study 1 |

没有公开比例的状态标记为未校准,**排除出训练**,不编造标签。A 的覆盖率
(114/126 = 90.5%)写在 manifest 里。

## 主方法:soft-label 分布匹配

```
loss = − Σ_c  p_human(c) · log softmax([logit_X, logit_Y])[c]
```

最优点**恰好**是人类分布。不需要 reference model(显存减半、约 2× 吞吐),
不需要 β,而且训练和评估是同一个口径。

**DPO 是对照方法,不是主方法。** 在 DPO 最优点处

```
logit(p_model) = logit(p_base) + logit(p_human) / β
```

位移的符号永远跟随人类多数派。所以当 base 已经比人类更极端(同方向过冲)时,
**任何正 β 都够不到目标,只会推得更远**——比例化 DPO 只能锐化,不能软化。
Gate 0 的 `dpo_unreachable_rate` 会在花 GPU 之前告诉你有多少比例的评估集
结构上够不到。这个 negative result 本身值得报告。

## 参照标尺(不需要 GPU,先跑)

MAE 单看没有意义。`reference_models.py` 用闭式预测器把标尺两端钉死:

| 参照模型 | C_test | A_test | B_test | B_control | D_test |
|---|---:|---:|---:|---:|---:|
| 噪声地板(完美模型) | **0.035** | 0.024 | 0.054 | 0.074 | 0.041 |
| `bayesian_hard` 完全理性 | **0.091** | 0.156 | 0.170 | 0.510 | 0.173 |
| `bayesian_soft` 概率匹配 | 0.151 | 0.205 | — | — | 0.166 |
| `uniform_half` 恒定 0.5 | 0.350 | 0.366 | 0.330 | **0.010** | 0.352 |

**要打败的是 0.091,不是 0.35。** 一个完全理性的 agent 在 C 上离人类只有 0.091,
地板是 0.035,总头寸只有 0.056。拿 uniform 当基线会严重夸大改进。

`oracle` 标记的参照模型读取了评估集自己的人类比例,只能当参考线,**不能当模型分数**。

## 防泄漏:split 级,不是 effect 级

Gate 1 必须在 C 的场景上训练(那是 ceiling),所以按 effect 名字一刀切要么挡住
测量、要么被人绕过。改成两把独立的锁:

1. 每行一个 `trainable` 布尔字段,由所在目录交叉验证(`dpo/` 和 `cv/*_train` 可训练,
   `eval/` 和 `cv/*_test` 不可);
2. 不可训练的行**根本不带 chosen/rejected**,训练入口读不到东西。

`eval/C_test.jsonl` 永远不可训练;`cv/` 的每一折都会校验 train/test 场景不相交。

## 快速开始

```bash
python -m effect_algebra.run_plan --format shell --output run_all.sh
```

会按顺序生成全部命令,每个阶段前带停止条件和预算估计(当前总计 12.5 GPU-h)。
前两个阶段不需要 GPU,**先在本地跑完再开 Colab**:

```bash
python -m effect_algebra.build_datasets --repo-root . --output-dir effect_algebra/data
python -m effect_algebra.reference_models \
  --dataset C_test=effect_algebra/data/eval/C_test.jsonl \
  --output-dir results/reference
```

必须看到 `errors: 0` 和 `c_test_used_for_training: false`。

## Colab

A100/L4 用默认 7B;只拿到 16 GB T4 就换 `Qwen/Qwen2.5-3B-Instruct`,
但 **A、B、C、D 和所有 baseline 必须用同一个 base**,中途不能换。

实测 prompt 长度:A 227 token、B 897、C 267、D ~250,所以
`--max-prompt-length 1024` 足够,不必用 2048。

纪律:一个 gate 一个 session;adapter 和结果直接写 Drive;A 和 B 之间清 GPU;
超参只能看 dev 和 Gate 1 的 fold 0,**任何情况下不能看 C_test 调参**。

## 三个 Gate

| Gate | 内容 | 停止条件 |
|---|---|---|
| 0 | base 画像 + 知识探针 + 过冲诊断 | MAE 已经很低 → 没有差距可补;探针显示模型不知道这些论文 → framing 站不住 |
| 1 | C 直训,5 折分层 CV → ceiling | ceiling 相对 base 的改进小于噪声地板 → 方法不成立,不要测迁移 |
| 2 | A+B 单个联合 adapter → C 零样本 | 先确认 A-only 在 A_test、B-only 在 B_test 上确实提升了 |

```
recovery fraction = (base − transfer) / (base − ceiling)
```

Gate 1 用 CV 而不是单次 20/20 划分,因为 Gate 0 和 Gate 2 都在全部 40 个场景上
评估;单次划分会让 ceiling 落在另一个测试集上,recovery fraction 混两个集合。

## 数据飞轮

`flywheel.py` 回答:**来源数据变多时,对未见范式的零样本校准会变好吗?
是"更多样"还是"更多量"?**

- **多样性轴**:A / B / D / A+B / A+D / B+D / A+B+D,**所有条件训练行数相同**。
  不固定行数的话,A+B 比 A 多一倍数据,提升归因不清。
- **数据量轴**:A+B+D 池的 12.5% / 25% / 50% / 100%,范式集固定。

子采样按 (label_group, response_code, label_side) 分层,所以人类比例和 X/Y 平衡
是构造性精确的,不是期望意义上的。

怎么读:多样性升、数据量平 → 该加论文;数据量升、多样性平 → 先扩样本;
两条都平 → 是方法问题,回去看 Gate 1。

## 报告纪律

不要把 C 折叠成一个"变好了"的数字。同时报:

1. 到人类分布的距离(MAE),分条件拆开——`opposes_private` 和 `indifference`
   两个子集是主战场,`supports_private` 已接近地板;
2. 残差里权威偏差是**过冲还是不足**。

更"服从权威"不等于更 Bayesian,这两个方向可能相反。

## 文件

| 文件 | 作用 |
|---|---|
| `human_priors.py` | 三篇论文的人类比例常量;噪声地板;DPO 可达性判据 |
| `datasets.py` | A/B/C/D 的行生成、分桶、比例化标签、CV 划分 |
| `build_datasets.py` | 生成 manifest 和全部 JSONL |
| `validate_datasets.py` | 标签重算、比例校验、split 级防泄漏 |
| `train_soft.py` | **主方法**:soft-label 分布匹配 |
| `train_dpo.py` | 对照:成对 DPO(带方向盲区说明) |
| `train_sft.py` | 对照:QLoRA-SFT |
| `evaluate_choices.py` | 单次 forward 读答案 token;MAE 为主指标 |
| `evaluate_suite.py` | 一次加载,多数据集评估 |
| `reference_models.py` | 免 GPU 的参照标尺 |
| `knowledge_probe.py` | Gate 0 知识探针 |
| `flywheel.py` | 数据飞轮的子集构造与 run plan |
| `run_plan.py` | 生成全部 Colab 命令 |
| `plot_calibration.py` | 校准散点图(纯 SVG,无依赖) |
| `compare_results.py` | 汇总 CSV / Markdown |
| `merge_adapters.py` | adapter 算术(消融用) |

技术依据:

- [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)
- [Transformers 4-bit NF4/QLoRA](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [PEFT model merging](https://huggingface.co/docs/peft/developer_guides/model_merging)
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
