# Effect A + B -> C: Colab 训练与评估

这个目录把三个已经实现的实验环境转成一个可运行的参数组合实验：

- **A / `study_016`**：顺序观察他人的公开选择，再结合自己的私有信号。
- **B / `study_017` Task 3C**：通过完整反馈历史区分“准确 advisor”和“仅仅同意自己
  的 advisor”。
- **C / `study_019` Study 2**：在顺序诊断中加入 `medical director` 的权威身份。

目标是分别训练 `Delta_A` 和 `Delta_B`，然后测试
`theta_0 + alpha * Delta_A + beta * Delta_B` 在 C 上是否产生可测量的迁移。

这不是数学意义上的严格 `A+B=C`。C 新增了 A/B 中没有直接监督的 authority cue，
所以实验检验的是跨任务机制组合，而不是预先保证成立的恒等式。

## 方法选择

| 名称 | 实际含义 | 是否作为主方法 |
| --- | --- | --- |
| 全参数 DPO | 更新 7B 模型全部参数 | 否，Colab 成本高，也失去独立 delta |
| LoRA-SFT | 只最大化正确答案 likelihood | 只作为 baseline |
| DPO + LoRA / LoRA-DPO | 用 DPO loss 训练 LoRA adapter | 两者是同一种组合 |
| QLoRA-DPO | 4-bit 冻结 base + LoRA 参数 + DPO loss | **是，主方法** |
| QDPO | Quantization-aware DPO，用全精度输出修复量化模型 | 否，目标不是学习 A/B effect |

主方法因此是 **Qwen2.5-7B-Instruct + 4-bit NF4 QLoRA + DPO**。A、B 必须：

1. 从同一个 base checkpoint 开始；
2. 使用完全相同的 LoRA rank、alpha、target modules；
3. 分别保存 adapter；
4. 在看 C 结果之前固定合并方式和权重。

默认配置为 rank 16、alpha 32、dropout 0.05、DPO beta 0.1、学习率
`2e-5`、两轮训练。
这是一组适合单卡 Colab 的首轮参数，不应在 C test 上调参。

## 为什么不让 OpenAI API 生成标签

A 和 B 都不需要 LLM 做 judge：

- A 的 chosen/rejected 可以从公开历史和私有信号精确计算。
- B 的 chosen advisor 可以从完整反馈 ledger 中重新计算平均绝对误差。
- C 是真正的 held-out test，绝不能进入生成或训练 prompt。

用 `gpt-4o-mini` 或 `gpt-5-mini` 生成几千个标签会引入 judge bias，并让所谓的
effect 参数变成对 OpenAI 模型偏好的蒸馏。当前实现只用程序化状态生成器产生数据。
后续确实需要语言鲁棒性时，可以让便宜模型生成**中性措辞模板**，但数值状态、标签和
split 都必须由本地 validator 重新验证。

## 数据边界

运行默认构建命令会生成：

| 文件 | 行数 | 用途 |
| --- | ---: | --- |
| `dpo/A_train.jsonl` | 512 | 训练 `Delta_A` |
| `dpo/A_dev.jsonl` | 128 | A 开发集 |
| `eval/A_test.jsonl` | 256 | A held-out wording |
| `dpo/B_train.jsonl` | 1024 | 训练 `Delta_B` |
| `dpo/B_dev.jsonl` | 256 | B 开发集 |
| `eval/B_test.jsonl` | 256 | B held-out episodes |
| `eval/B_no_feedback_control.jsonl` | 256 | 3B 控制，不含偏好标签 |
| `eval/C_test.jsonl` | 40 | `study_019` 的全部医疗场景 |

`AB_train/dev` 只是 A 和 B 的交错联合视图，用于 joint-training baseline；它们不是
额外数据。默认整棵数据树共 4648 行（包含联合视图），约 31 MB。

关键保护：

- A 系统枚举有限状态，不用重复样本伪造规模。
- X/Y response code 每条随机映射并严格平衡，防止学习固定 token。
- B 每条样本包含两位 advisor 各 15 轮的完整 episode。
- B ledger 的正确年份、误差和 chosen advisor 会在训练前重算。
- B 3B/no-feedback 没有 chosen/rejected，代码禁止把它送入训练。
- C 的 40 个材料 fingerprint 来自 `study_019`; 其中 10 个 Bayesian tie 保持
  `target_code=null`，不会被强行标注。
- 训练脚本发现 B prompt 超过 token limit 时直接失败，不允许静默裁剪早期历史。

## Colab 快速运行

推荐直接打开：

[`colab_effect_algebra.ipynb`](colab_effect_algebra.ipynb)

在 Colab 中选择 `Runtime > Change runtime type > GPU`。A100/L4 使用默认 7B；
如果只拿到 16 GB T4，先把模型改成 `Qwen/Qwen2.5-3B-Instruct`，并保证 A、B、
merge 和所有 baseline 都使用同一个 3B base。

### 1. 获取代码并安装

```bash
git clone --branch pipeline https://github.com/HumanStudy-Hub/HumanStudy-Bench.git
cd HumanStudy-Bench
pip install -q -r effect_algebra/requirements-colab.txt
```

如果仓库仍是私有仓库，不要把 GitHub token 写进 notebook。通过 Colab 的 GitHub
授权打开 notebook，或把 checkout 放进 Google Drive 后设置 `REPO_DIR`。

### 2. 生成并验证数据

```bash
python -m effect_algebra.build_datasets \
  --repo-root . \
  --output-dir /content/drive/MyDrive/effect_algebra_ab_c/data

python -m effect_algebra.validate_datasets \
  --data-dir /content/drive/MyDrive/effect_algebra_ab_c/data
```

必须看到 `"errors": 0` 和 `"C_used_for_training": false`。

### 3. 训练 A adapter

```bash
python -m effect_algebra.train_dpo \
  --train-file /content/drive/MyDrive/effect_algebra_ab_c/data/dpo/A_train.jsonl \
  --eval-file /content/drive/MyDrive/effect_algebra_ab_c/data/dpo/A_dev.jsonl \
  --output-dir /content/drive/MyDrive/effect_algebra_ab_c/adapters/A_dpo \
  --run-name effect-A-dpo \
  --base-model Qwen/Qwen2.5-7B-Instruct
```

### 4. 训练 B adapter

先重启 runtime 或执行 notebook 中的 GPU 清理 cell，再运行：

```bash
python -m effect_algebra.train_dpo \
  --train-file /content/drive/MyDrive/effect_algebra_ab_c/data/dpo/B_train.jsonl \
  --eval-file /content/drive/MyDrive/effect_algebra_ab_c/data/dpo/B_dev.jsonl \
  --output-dir /content/drive/MyDrive/effect_algebra_ab_c/adapters/B_dpo \
  --run-name effect-B-dpo \
  --base-model Qwen/Qwen2.5-7B-Instruct
```

不要把 B 切成单轮样本。最终 advisor choice 的学习信号依赖前面所有反馈。

### 5. 精确组合 A+B adapter

```bash
python -m effect_algebra.merge_adapters \
  --adapter-a /content/drive/MyDrive/effect_algebra_ab_c/adapters/A_dpo \
  --adapter-b /content/drive/MyDrive/effect_algebra_ab_c/adapters/B_dpo \
  --output-dir /content/drive/MyDrive/effect_algebra_ab_c/adapters/A_plus_B_cat \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --weight-a 1.0 \
  --weight-b 1.0 \
  --combination-type cat
```

`cat` 保存的是精确的 `Delta_A + Delta_B`，代价是合并 adapter 的 rank 从 16
增加到 32。`linear` 维持 rank，但不是同样的矩阵级精确表示；`ties` 应当作为后续
消融，不应替代第一条基线。

### 6. 一次加载模型并评估 A/B/C

下面以 A+B 为例：

```bash
python -m effect_algebra.evaluate_suite \
  --model-label A_plus_B \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter /content/drive/MyDrive/effect_algebra_ab_c/adapters/A_plus_B_cat \
  --dataset A_test=/content/drive/MyDrive/effect_algebra_ab_c/data/eval/A_test.jsonl \
  --dataset B_test=/content/drive/MyDrive/effect_algebra_ab_c/data/eval/B_test.jsonl \
  --dataset B_control=/content/drive/MyDrive/effect_algebra_ab_c/data/eval/B_no_feedback_control.jsonl \
  --dataset C_test=/content/drive/MyDrive/effect_algebra_ab_c/data/eval/C_test.jsonl \
  --output-dir /content/drive/MyDrive/effect_algebra_ab_c/results/A_plus_B
```

对 `base`、`A_dpo`、`B_dpo` 和 `A_plus_B` 分别运行一次。`evaluate_suite` 对每个
模型只加载一次权重，然后连续跑四个数据集。

## 最小实验矩阵

必须至少保留：

| 模型 | A test | B test | B no-feedback | C test |
| --- | --- | --- | --- | --- |
| Base | 必跑 | 必跑 | 必跑 | 必跑 |
| A-only | 必跑 | 必跑 | 必跑 | 必跑 |
| B-only | 必跑 | 必跑 | 必跑 | 必跑 |
| A+B exact `cat` | 必跑 | 必跑 | 必跑 | 必跑 |
| Joint A+B DPO | 推荐 | 推荐 | 推荐 | 推荐 |
| LoRA-SFT A/B | 可选 baseline | 可选 | 可选 | 可选 |

Joint DPO 命令使用 `dpo/AB_train.jsonl` 和 `dpo/AB_dev.jsonl`。SFT baseline 使用
`python -m effect_algebra.train_sft`，其余参数与 DPO 相同。

权重搜索只能看 A/B dev。建议第二轮再尝试
`alpha,beta in {0.5, 1.0, 1.5}`，先按 A/B dev 的最小值或调和平均数选一个组合，
冻结权重后只运行一次 C test。不能依据 C 结果挑权重。

## 如何读结果

`evaluate_suite` 不依赖生成文本或 parser，而是计算两个允许 completion 的条件
log-prob，因此比采样一次 `DECISION=X/Y` 稳定。

- `accuracy`：A/B 或 C 非 tie 场景的目标选择准确率。
- `mean_target_probability`：模型分配给目标选项的平均概率。
- `mean_preference_margin`：目标 completion 与另一个 completion 的 log-prob 差。
- `decision_x_rate`：检查随机 response code 是否仍有 token 偏置。
- `advisor_agreement`：B 控制中选择 agreeing advisor 的概率；3B 不作为正确率。
- `human_distribution.weighted_probability_mae`：C 模型分布与原始人类选择率的差。
- `authority.hard_alignment_rate`：存在 medical director 时模型跟随权威诊断的比例。

C 必须同时报告两类结果：

1. **normative**：是否符合 source-grounded Bayesian choice；
2. **human-like**：是否接近人类 authority alignment 分布。

这两个方向可能相反。模型更“服从权威”不等于 Bayesian 表现更好。

## 推荐运行顺序与 credit 使用

先做主线，不要一开始跑完整网格：

1. 生成和验证数据；
2. Base 在 A/B/C 上的 pre-test；
3. 训练 A DPO；
4. 训练 B DPO；
5. 合并 1.0A + 1.0B；
6. 对 Base/A/B/A+B 跑同一套测试；
7. 只有结果显示 A、B 各自学成功后，再跑 Joint DPO、SFT 和权重网格。

如果 A-only 在 A test 或 B-only 在 B test 上没有稳定提升，停止解释 C；此时
`A+B` 没有可验证的组成基础，应先检查训练曲线、token limit 和数据分布。

## 主要文件

- `datasets.py`：A/B 程序化状态、C source-grounded 转换。
- `build_datasets.py`：生成 manifest 和 JSONL。
- `validate_datasets.py`：标签重算、防泄漏和训练边界。
- `train_dpo.py`：主 QLoRA-DPO。
- `train_sft.py`：QLoRA-SFT baseline。
- `merge_adapters.py`：PEFT adapter arithmetic。
- `evaluate_choices.py`：forced-choice log-prob 与指标。
- `evaluate_suite.py`：一次加载，多数据集评估。
- `compare_results.py`：汇总 CSV/Markdown。

技术依据：

- [TRL DPOTrainer 数据格式与 PEFT 支持](https://huggingface.co/docs/trl/dpo_trainer)
- [Transformers 4-bit NF4/QLoRA](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [PEFT model merging](https://huggingface.co/docs/peft/developer_guides/model_merging)
- [QDPO 原论文](https://arxiv.org/abs/2407.03051)
- [Qwen2.5-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
