# A + B → C 实验计划(Colab Pro 版)

目标:在 A(信息级联)和 B(顾问选择)上训练社会影响的行为校准,零样本迁移到
C(医疗权威),回答"社会影响的 agent 模拟能不能做成通用能力"。

主指标是 **MAE(模型选择分布 vs 人类选择分布)**,不是准确率。

---

## 0. 资源约束

Colab Pro,不是 server。这个约束决定了下面所有设计:

| 约束 | 影响 |
|---|---|
| 每月 ~100 compute units | L4 大约 4–5 units/h(以实际费率为准)→ 全月约 20–25 GPU-h |
| 空闲 ~90 min 断连,单 session 最长 12–24 h | 每个训练 run 必须 < 3 h,且能从 Drive checkpoint 恢复 |
| GPU 型号不保证 | 所有 run 必须能在 L4 24 GB 上跑;T4 16 GB 作为降级预案 |
| 磁盘非持久 | adapter / 结果 / 数据全部落 Google Drive |

实测 prompt 长度(`build_*_rows` 直接量的字符数 ÷ 4):

| | 平均 token | 最大 token |
|---|---:|---:|
| A | 227 | 232 |
| B | 897 | 899 |
| C | 267 | 281 |

**结论:`max_prompt_length` 可以从默认 2048 降到 1024**,显存和速度都受益。
7B NF4 + LoRA r16 + gradient checkpointing + batch 1 在 L4 上余量充足,T4 也能跑。

---

## 1. 五个关键设计决策

### 1.1 主方法改为 soft-label 分布匹配,DPO 降为对照

这是相对原 plan 最重要的一处改动,理由是一个**结构性问题**,不是调参问题。

比例化 DPO 在总体损失最优点处满足:

```
logit(p_model) = logit(p_base) + logit(p_human) / β
```

要让 `p_model = p_human`,需要 `β = logit(p_h) / (logit(p_h) − logit(p_base))`,而 β 必须为正。
展开这个条件会得到:

> **base 必须比人类更靠近 0.5。**
> 如果 base 已经比人类更极端(同方向过冲),任何正 β 都只会把模型推得**更**极端。

举例(已验算):

| 人类 | base | β* | 结果 |
|---|---|---|---|
| 0.75 | 0.60 | +1.59 | 可达 |
| 0.43 | 0.90 | +0.11 | 可达 |
| 0.75 | 0.95 | −0.60 | **无解** |
| 0.95 | 0.99 | −1.78 | **无解** |

也就是说 **比例化 DPO 只能锐化,不能软化**。而 instruct 模型在 forced-choice 上
系统性过度自信,C 里有 16/40 个场景人类比例落在 0.15–0.85,这些正是最可能踩坑的。
β sweep 解决不了——β 只控制幅度,不控制方向。

**替代方案:直接在两个答案 token 上做 soft-label 交叉熵。**

```
loss = − Σ_c p_human(c) · log softmax([logit_X, logit_Y])[c]     c ∈ {X, Y}
```

优点全是 Colab 上的硬收益:

- 最优点**恰好**是人类分布,无方向盲区,无 β
- 不需要 reference model → 显存减半,速度约 2×
- 不需要成对样本 → 每个 prompt 一行,数据量减少
- 直接优化最终汇报的指标(MAE / 交叉熵),训练与评估同一口径

**保留 DPO 作为对照方法**,并把"方向盲区"作为论文里的一个 negative result 报出来
(这比单纯报 DPO 效果差更有价值)。

### 1.2 不做 N 倍样本复制

原 plan 的 `for i in range(N)` 复制会把数据放大 20 倍,直接击穿 Colab 预算。
不需要:**现有行数本身就能承载比例**。

- A:512 行分布在约 3 个 bucket 上,每 bucket ~170 行,把其中 73.2% 标成"跟随 Bayes"
  即可精确表达 0.732,零额外成本
- B:1024 行同属 3C 一个 bucket,83% 标 accurate
- C(Gate 1):每场景 20 行,分辨率 0.05,对 n=40 的人类估计足够

如果走 1.1 的 soft-label 路线,连标签分配都不需要——直接把 `p_human` 写进行里。
比例化标签只在 DPO 对照分支使用。

**注意**:标签语义与 X/Y response code 必须**独立**随机化。现在
`RESPONSE_CODES[index % 2]` 把 code 绑死在 target 上;一旦按比例翻转语义,
code 平衡就会被破坏并重新引入 token 偏置。

### 1.3 Gate 1 用 5-fold CV,不用单次 20/20 划分

40 个场景按 posterior × authority 交叉后有 11 个非空格,最小的只有 2 个场景:

```
              baseline  supports  opposes
p=0.50            3         2        5
p=0.67            4         2        7
p=0.80            3         5        2
p=0.89            2         5        0
```

单次 20/20 分层划分会让测试集里出现 n=1 的格。更要命的是**口径不一致**:
Gate 0 和 Gate 2 都在全部 40 个场景上评估,Gate 1 只在 20 个上评估,
recovery fraction 会混两个不同的测试集。

改成 5-fold 分层 CV,每个场景都被留出恰好一次,ceiling 同样基于全部 40 个场景,
三个 gate 完全可比。成本可控(见 §7),因为 C 的训练集本来就小。

### 1.4 主线是单个联合 adapter,adapter 算术降为消融

原 README 把 `merge_adapters.py --combination-type cat` 当主线。改成:
用 `AB_train.jsonl` 训**一个** adapter。理由是避免 merge 系数成为额外的失败模式——
A+B 迁移失败时无法区分是"能力不可组合"还是"合并权重没调好"。
`cat` / `linear` 合并作为消融保留,回答"delta 是否可加"这个独立问题。

### 1.5 先建参照标尺,再解释任何 MAE

人类比例本身是有限样本估计,一个**完美**模型的 MAE 不是 0。这些参照模型
**不需要 GPU**,已经跑完(`reference_models.py`):

| 参照模型 | C_test | A_test | B_test | B_control | D_test | 性质 |
|---|---:|---:|---:|---:|---:|---|
| 噪声地板(完美模型) | **0.035** | 0.024 | 0.054 | 0.074 | 0.041 | 下界 |
| `bayesian_hard`(完全理性) | **0.091** | 0.156 | 0.170 | 0.510 | 0.173 | 真实上界 |
| `bayesian_soft`(概率匹配后验) | 0.151 | 0.205 | — | — | 0.166 | |
| `uniform_half`(恒定 0.5) | 0.350 | 0.366 | 0.330 | **0.010** | 0.352 | 无信息 |
| `condition_mean_oracle` | 0.345 | 0.366 | 0.330 | 0.010 | 0.352 | oracle |

四条必须写进论文的结论:

1. **真正要打败的是 0.091,不是 0.35。** 一个完全理性的贝叶斯 agent 在 C 上
   已经离人类只有 0.091,而地板是 0.035——总共只有 **0.056 的头寸**。
   拿 uniform 的 0.35 当基线会严重夸大改进幅度。
2. **`bayesian_soft` 比 `bayesian_hard` 更差**(0.151 vs 0.091)。人类比后验本身
   更果断,不做概率匹配。这本身是个可报告的发现。
3. **B_control 的最优行为是"没有偏好"**:恒定 0.5 的 MAE 只有 0.010,而
   `bayesian_hard` 是 0.510。3B 是检验模型知不知道**何时不该有立场**的探针,
   比 3C 更难,也更有信息量。
4. **`condition_mean_oracle` 几乎等于 `uniform_half`**,说明条件均值解释不了
   逐场景的变异——item 级信息是必需的,这正当化了逐场景校准。

C 的分条件头寸(`bayesian_hard` 的残差,即真正的可改进空间):

```
medical_director_opposes_private   MAE=0.134  n=14   ← 权威偏差所在,头寸最大
baseline                           MAE=0.094  n=12
medical_director_supports_private  MAE=0.046  n=14   ← 已接近地板,几乎没有空间
indifference subset                MAE=0.133  n=10   ← 三个参照模型全是 0.133
```

`supports_private` 只剩 0.046,基本触底。**主战场是 opposes 和 indifference 两个
子集**,合计 24 个场景。总 MAE 会被 supports 子集稀释,所以必须分条件报告。

---

## 2. 阶段 P0:数据层改造(CPU,不烧 GPU)

在碰 GPU 之前全部做完并通过测试。

| # | 改动 | 文件 |
|---|---|---|
| P0.1 | 每行增加 `human_probability_by_code` 字段;A/B/C 都要有 | `datasets.py:224` |
| P0.2 | 重做 A 的分桶:cascade / posterior-tie / position,删掉死分支 `public_private_conflict` | `datasets.py:198` |
| P0.3 | 把 A 的 8 个人类比例写成常量表并挂到 bucket 上 | 新 `human_priors.py` |
| P0.4 | B 注入 3C 比例 0.17;若扩到 9 个实验则注入 9 个 | `datasets.py:538` |
| P0.5 | C 从 `scenarios.json` 读逐场景 `option_1_rate`(已有) | `datasets.py:709` |
| P0.6 | code ↔ 语义标签解耦,各自独立平衡 | `datasets.py:216` |
| P0.7 | 防泄漏从 effect 级降到 split 级:放行 `C_train`,继续拒绝 `C_test` | `validate_datasets.py:198`,`train_dpo.py:30` |
| P0.8 | 生成 5-fold 分层 CV 划分文件(按 posterior × authority) | `build_datasets.py` |

A 的人类比例常量表(全部来自 Anderson & Holt 1997,已核对原文):

| bucket | 比例 | 出处 |
|---|---|---|
| cascade 机会,跟随 Bayes | 41/56 = 0.732 | Results p.9 |
| posterior = 1/2 且与前一决策冲突,跟私有信号 | 57/68 = 0.838 | Biases pp.16–17 |
| 第 2 轮与第 1 轮冲突,跟私有信号 | 0.95 | p.14 |
| 按位置偏离率(分母各 90) | 4, 3, 6, 14, 13, 7 | Table 3 |

位置偏离率的口径要标注:那是相对递归 logit 模型的偏离,不是相对纯 Bayes。
**落不进上表的 A 状态标记为未校准,不进训练,并在论文里报覆盖率。**

验收:`pytest tests/` 全绿 + `validate_datasets` 报 `errors: 0` 且
`C_test_used_for_training: false`。

---

## 3. 阶段 P1:评估器改造(CPU 写,GPU 短验)

| # | 改动 | 文件 |
|---|---|---|
| P1.1 | 单次 forward 读末位 logits,取代两次完整 forward(约 2× 提速) | `evaluate_choices.py:198` |
| P1.2 | MAE 提为主指标,accuracy 降为次要 | `evaluate_choices.py:75` |
| P1.3 | 按 bucket 拆 MAE:authority condition × posterior;单独报 indifference 子集 | `evaluate_choices.py:101` |
| P1.4 | 导出散点图数据(逐场景 human % vs model %) | 新 `plot_calibration.py` |
| P1.5 | 输出噪声地板和两个平凡基线,与 MAE 并排 | `compare_results.py` |
| P1.6 | 新增知识探针脚本(问模型三篇论文的结论) | 新 `knowledge_probe.py` |

C 的分组信号强度(人类 private-choice rate,已算):

```
baseline                    n=12   0.748   [0.25, 1.00]
director supports private   n=14   0.932   [0.60, 1.00]
director opposes private    n=14   0.434   [0.12, 0.90]
indifference subset         n=10   0.497   [0.30, 0.75]
```

supports 0.932 vs opposes 0.434 是 **0.50 的摆幅**,这就是要复现的权威效应本体。
全 40 个场景的总 MAE 会被那些人类比例接近 1.0 的简单场景稀释,
**必须同时报 opposes 子集和 indifference 子集的 MAE**,否则看不出模型学没学到权威效应。

---

## 4. Gate 0 — Base 画像

**跑什么**

1. base 模型在 A_test(256)/ B_test(256)/ B_control(256)/ C_test(40)上的
   forced-choice 分布 → 每个 study 的 MAE + 散点图
2. 知识探针:直接问模型三篇论文各自的发现
3. **过冲诊断(新增,决定主方法)**:逐场景比较 `|logit(p_base)|` 与 `|logit(p_human)|`,
   统计 base 比人类更极端的场景占比

**停止条件**

- 如果三个 study 的 MAE 都 < 0.10 → 没有校准差距可补,报告并停止
- 如果知识探针显示模型**不知道**这三篇论文 → "污染不是真风险"这个 framing 站不住,
  改写 framing 而不是继续跑

**分叉点**

- 过冲场景占比 > 50% → 主方法用 soft-label(§1.1),DPO 作为对照
- 过冲场景占比 < 50% → 两条路都可行,仍推荐 soft-label(更省)

预算:~0.5 GPU-h

---

## 5. Gate 1 — C 直训(ceiling)

在 C 自己的场景上直接训练,得到"领域专用 adapter 能做到多好"的上界。

**流程**

1. 5-fold 分层划分(posterior × authority),每折 8 个测试场景 / 32 个训练场景
2. 每个训练场景展开 20 行 → 640 行/折
3. **fold 1 上做超参 sweep**:soft-label 分支扫 lr ∈ {1e-4, 2e-4, 5e-4};
   DPO 对照分支扫 β ∈ {0.1, 0.3, 1.0}
4. 固定超参后跑 fold 2–5,**ceiling 报 fold 2–5 的均值**(fold 1 已用于选参,排除)
5. 每折在自己的 8 个留出场景上评估,汇总覆盖全部 32 个场景

**停止条件**:如果 ceiling MAE 相对 Gate 0 没有明显下降(比如降幅 < 噪声地板 0.035),
说明校准方法本身不成立,报告并停止,不要去测迁移。

预算:soft-label 分支 3(sweep)+ 4(折)= 7 个 run,每个 ~0.25 h → ~2 GPU-h
DPO 对照分支同规模 → 再 ~3 GPU-h(需 reference model,更慢)

---

## 6. Gate 2 — A+B 联合训练,零样本测 C

**流程**

1. 用 `AB_train.jsonl`(A 512 + B 1024 交错)训**单个** adapter,超参沿用 Gate 1 选定值
2. C **完全不出现**在训练或 prompt 里
3. 在全部 40 个 C 场景上零样本评估
4. 先验证 A/B 各自学成:A-only 在 A_test、B-only 在 B_test 上要有稳定提升,
   否则 A+B 没有可验证的组成基础,先查训练曲线而不是解释 C

**主结果表**

| | C MAE(全部 40) | opposes 子集 | indifference 子集 | 残差方向 |
|---|---|---|---|---|
| 噪声地板 | 0.035 | | | — |
| 恒定 0.5 基线 | 0.350 | | | — |
| Base(Gate 0) | floor | | | |
| A+B 迁移(Gate 2) | ? | | | |
| C 直训(Gate 1) | ceiling | | | |

```
recovery fraction = (base − transfer) / (base − ceiling)
```

**报告纪律**:不要把 C 折叠成一个"变好了"的数字。同时报
(a) 到人类分布的距离 MAE,(b) 残差里权威偏差是过冲还是不足。
这两个方向可能相反——更"服从权威"不等于更 Bayesian。

预算:1536 行 × 2 epoch,B 占大头 → ~1.5 GPU-h(soft-label);DPO 约 2.5 GPU-h

---

## 6.5 新增环境 D:study_019 Study 1

翻数据时发现 `study_019` 的 **Study 1 完全没被用上**:24 个 urn 场景,逐场景人类
比例,n=40,和 C 同一批被试、同一个 workbook,但**没有权威操纵**。已实现为
effect D(`build_d_rows` / `build_d_training_rows`)。

它一次解决三个问题:

1. **飞轮的第四个来源**——一个可加进训练池的真实环境;
2. **A 的 item 级校验**——D 和 A 是同一个 urn 级联范式,但 A 只有 bucket 级比例,
   D 有逐场景比例。在 D 上测 A-only adapter,就能知道 bucket 级监督是否够用;
3. **权威 cue 的对照**——D 和 C 结构相同、只差权威操纵。如果 A+B → D 的迁移好而
   A+B → C 差,瓶颈就明确是权威 cue 本身,而不是范式差异。

## 7. 数据飞轮实验

核心问题:**随着来源数据增加,对未见范式的零样本校准会变好吗?是"更多样"还是
"更多量"在起作用?**(`flywheel.py`)

两条轴,刻意分开:

**多样性轴(固定总行数)**:`A` / `B` / `D` / `A+B` / `A+D` / `B+D` / `A+B+D`

这里的关键约束是**所有条件训练行数相同**(默认 480)。不固定的话,`A+B` 比 `A`
多一倍数据,任何提升都归因不清——是范式变多还是数据变多完全分不开。

**数据量轴(固定范式集)**:`A+B+D` 池的 12.5% / 25% / 50% / 100%

两条轴合起来才能区分"更多数据"和"更多种数据",并看出曲线在哪里饱和。

**子采样必须精确保比例**,不是期望意义上的。实现按 (label_group, response_code,
label_side) 单元分层抽样,所以人类比例和 X/Y 平衡都是构造性精确的。随机抽样在
小预算下漂移足够大,会改变模型被要求匹配的目标——那是唯一不能随数据量变化的东西。

实测验证:

```
条件                A 观测/目标      B 观测/目标      D 观测/目标      X/Y
A_plus_B_plus_D   0.856/0.852    0.825/0.830    0.475/0.502   240/240
volume_050pct     0.875/0.850    0.825/0.830    0.425/0.446   120/120
A_only            0.854/0.852         —              —        240/240
```

**怎么读结果**:

- 多样性曲线上升、数据量曲线平 → 飞轮该**加论文**,不是加行数
- 数据量曲线上升、多样性曲线平 → 现有范式还没榨干,先扩样本
- 两条都平 → 迁移不是数据问题,是方法问题,回去看 Gate 1 的 ceiling

分层交付(预算不够就砍后面的):`core` = A/B/A+B(3 run)→
`extended` = D/A+B+D(2 run)→ 数据量序列(3 run)。

## 8. 消融(只在主线跑通后做)

按优先级,预算够多少做多少:

| 优先级 | 消融 | 回答什么 | 预算 |
|---|---|---|---|
| 1 | A-only → C,B-only → C | 迁移来自哪个 study | ~1.5 GPU-h |
| 2 | DPO vs soft-label 同条件对比 | 方向盲区是否真的发生 | ~3 GPU-h |
| 3 | `cat` 合并 vs 联合训练 | delta 是否可加 | ~0.5 GPU-h |
| 4 | B 扩到 9 个实验的 pick rate | 更多校准点是否改善迁移 | ~2 GPU-h |
| 5 | 个体级校准(C 有 40×40 逐被试) | 能否复现个体差异而不只是均值 | ~3 GPU-h |
| 6 | study_019 Study 1(24 场景,无权威)作为第四个环境 | 权威 cue 是否是迁移瓶颈 | ~1.5 GPU-h |

消融 5 值得单独说:C 的 director alignment 在被试间从 0.50 跨到 1.00(中位 0.79),
总体均值 0.749 掩盖了这个双峰。总体校准做对了,个体分布仍可能完全错。

---

## 9. 预算表

`python -m effect_algebra.run_plan` 会按顺序生成全部命令,并在每个阶段前打印
停止条件和预算。当前总计 **12.5 GPU-h**:

| 阶段 | 需要 GPU | GPU-h | 累计 |
|---|---|---:|---:|
| `build_data` 生成 + 校验 | 否 | 0.0 | 0.0 |
| `reference_models` 参照标尺 | **否** | 0.0 | 0.0 |
| `base_profile` Gate 0 + 知识探针 | 是 | 0.5 | 0.5 |
| `gate_1_ceiling` 5 折 CV | 是 | 2.0 | 2.5 |
| `gate_2_transfer` A+B → C | 是 | 1.5 | 4.0 |
| `ablation_single_effect` | 是 | 1.5 | 5.5 |
| `ablation_dpo_objective` β sweep | 是 | 3.0 | 8.5 |
| `flywheel` 8 个条件 | 是 | 4.0 | 12.5 |
| `report` 汇总 + 出图 | 否 | 0.0 | 12.5 |

前两个阶段不用开 Colab,先在本地跑完——参照标尺定住之后,Gate 0 的数字才有意义。

L4 上 12.5 GPU-h,按 4–5 units/h 估算约 50–65 units,**在单月 100 units 内**,
留出余量给重跑。若改用 A100 会快约 2×,但单价约 2.5×,**不划算,默认用 L4**。

降级预案:如果反复拿不到 L4,换 `Qwen/Qwen2.5-3B-Instruct`,
但 **A、B、C、所有 baseline 必须用同一个 base**,中途不能换。

---

## 10. Colab 运行纪律

1. **每个 gate 一个 session**,开头 mount Drive,结尾把 adapter + results 写回 Drive
2. `save_strategy="epoch"` + `--resume-from-checkpoint`,断连能续
3. A 和 B 之间**必须清 GPU**(重启 runtime 或跑 notebook 里的清理 cell)
4. `max_prompt_length` 设 1024(实测 B 最长 899)
5. 每个 run 落一份 `experiment_manifest.json`,含数据 sha256、超参、log_history
6. **超参只能看 A/B dev 和 Gate 1 fold 1**,任何情况下不能看 C_test 调参
7. 长任务用 `nohup` 或 background cell,避免 90 min 空闲断连

---

## 11. 待办的外部依赖

| 项 | 状态 | 阻塞什么 |
|---|---|---|
| C 逐被试数据 | ✅ 已在仓库,MD5 已验,重算 alignment 0.7490 对上论文 838/1119 | 无 |
| B 逐被试数据 | ⬜ 需从 Zenodo 下载 `oxacclab/esmData` R package | 只阻塞消融 4–5 |
| A 完整数据 | ❌ 论文脚注 18:需向作者索取,无公开存档 | 只能 bucket 级,写进 limitation |

A 的数据可以发邮件索取,但**不要 block 主线**——bucket 级的 8 个比例已经够跑完三个 gate。

---

## 12. 论文骨架(数字齐了再填)

> 我们把社会影响的 agent 模拟视为校准问题。在三个经典范式上,base 模型的行为
> 与人类数据相差 ___(Gate 0),而知识探针显示模型能准确复述这三篇论文的结论,
> 说明这不是知识缺失。以人类反应比例作为训练目标,in-domain 可把差距压到
> ___(Gate 1,噪声地板 0.035)。仅在另外两个范式上训练的 adapter 零样本迁移到
> 未见过的权威场景集,回收了 in-domain 增益的 ___%(Gate 2),
> 说明社会行为的校准 ___(可以 / 不可以)以通用方式实现。
