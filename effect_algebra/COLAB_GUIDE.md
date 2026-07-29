# Colab 操作指南

> 最后更新:2026-07-28,对应 commit `e90a6b1` 之后的最新 `pipeline` 分支。
> 每次开新 runtime 都从 GitHub 拉最新版,不要照抄旧的聊天记录。

配套文档:[`REPORT_CN.md`](REPORT_CN.md)(实验结果与日志)、
[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)(完整实验设计)。

---

## 0. 先解决丢数据的问题

已经丢过两次。Colab runtime 一断,`/content` 下**全部消失**——代码、数据、adapter、结果。
代码在 GitHub 上没事,数据一条命令能重建,但**训练出来的 adapter 和评估结果没有备份就是真没了**。

按价值排序:

| 产物 | 大小 | 丢了的代价 |
|---|---|---|
| 评估结果 JSON | 每个数据集约 40 KB–1 MB | **这是论文要的东西,必须保住** |
| 训练 adapter | 每个约 275 MB(14B,r16) | 1 折 ≈ 12 分钟 GPU |
| 数据树 | 58 MB | 1 分钟重建,不用存 |
| 代码 | — | 在 GitHub 上 |

**两个选项,挑一个,别裸奔:**

**选项 A(推荐):挂 Drive,所有输出直接写进去。** 在 **notebook cell**(不是终端)里运行:

```python
from google.colab import drive
drive.mount('/content/drive')
```

然后本指南里所有 `/content/ea` 换成 `/content/drive/MyDrive/ea`。共享账号介意的话,跑完 `drive.flush_and_unmount()` 即可。

**选项 B:每个阶段结束立刻打包下载。**

```bash
cd /content/ea && tar czf /content/results_$(date +%H%M).tgz results && ls -la /content/results_*.tgz
```

在左侧文件浏览器里右键 Download。

---

## 1. 冷启动(每次新 runtime 都要做)

`Runtime > Change runtime type > A100 GPU`(或 L4,见 §7 预算)。

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
```

```bash
cd /content && git clone --branch pipeline https://github.com/HumanStudy-Hub/HumanStudy-Bench.git
```

```bash
pip install -q -r /content/HumanStudy-Bench/effect_algebra/requirements-colab.txt
```

装完会报 `diffusers` 和 `gradio` 的版本冲突——**可以忽略**,那两个包我们不用。
装完**大概率需要 `Runtime > Restart session`**(不用重新 clone)。

验证环境:

```bash
cd /content/HumanStudy-Bench && python -c "import torch,transformers,peft,trl,bitsandbytes,datasets; print('torch',torch.__version__,'| tf',transformers.__version__,'| cuda',torch.cuda.is_available())"
```

> **所有后续命令都必须在 `/content/HumanStudy-Bench` 目录下运行**,否则会报
> `ModuleNotFoundError: No module named 'effect_algebra'`。断线重连后先 `cd` 回来。

---

## 2. 生成数据(不用 GPU,约 1 分钟)

```bash
cd /content/HumanStudy-Bench && python -m effect_algebra.build_datasets --repo-root . --output-dir /content/ea/data
```

必须看到 `"errors": 0` 和 `"c_test_used_for_training": false`。数据是确定性的(固定 seed +
sha256),任何时候重建都一模一样,所以不用备份。

---

## 3. 参照标尺(不用 GPU,几秒)

这一步把 MAE 的尺子两端钉死。**先跑它,再跑任何模型**,否则拿到的 MAE 无法解释。

```bash
python -m effect_algebra.reference_models --dataset A_test=/content/ea/data/eval/A_test.jsonl --dataset B_test=/content/ea/data/eval/B_test.jsonl --dataset B_control=/content/ea/data/eval/B_no_feedback_control.jsonl --dataset C_test=/content/ea/data/eval/C_test.jsonl --dataset D_test=/content/ea/data/eval/D_test.jsonl --dataset B_probe_r5=/content/ea/data/eval/B_probe_r5.jsonl --output-dir /content/ea/results/reference
```

应该复现 [`REPORT_CN.md`](REPORT_CN.md) §2 那张表。对不上说明环境有问题,停下来查。

---

## 4. Gate 0:base 画像(A100 约 0.5 h)

```bash
python -m effect_algebra.evaluate_suite --model-label qwen14b --base-model Qwen/Qwen2.5-14B-Instruct --dataset A_test=/content/ea/data/eval/A_test.jsonl --dataset B_test=/content/ea/data/eval/B_test.jsonl --dataset B_control=/content/ea/data/eval/B_no_feedback_control.jsonl --dataset C_test=/content/ea/data/eval/C_test.jsonl --dataset D_test=/content/ea/data/eval/D_test.jsonl --dataset B_probe_r5=/content/ea/data/eval/B_probe_r5.jsonl --output-dir /content/ea/results/qwen14b
```

```bash
python -m effect_algebra.knowledge_probe --model-label qwen14b --base-model Qwen/Qwen2.5-14B-Instruct --output /content/ea/results/qwen14b/knowledge_probe.json
```

首次会下载 14B(约 28 GB),下载占大头。看结果:

```bash
python -m effect_algebra.digest --results-dir /content/ea/results/qwen14b --reference-dir /content/ea/results/reference
```

---

## 5. Gate 1:C 直训 5 折 CV(A100 约 1.2 h)

**长任务一定用 `nohup` 挂后台**,终端断了也不影响:

```bash
cd /content/HumanStudy-Bench && nohup bash -c 'for f in 0 1 2 3 4; do python -m effect_algebra.train_soft --train-file /content/ea/data/cv/C_fold${f}_train.jsonl --eval-file /content/ea/data/cv/C_fold${f}_test.jsonl --output-dir /content/ea/adapters/C_fold$f --run-name c-fold$f --base-model Qwen/Qwen2.5-14B-Instruct; done' > /content/gate1.log 2>&1 &
```

看进度(断线重连后再跑一次就行):

```bash
tail -f /content/gate1.log
```

训练完**立刻**跑评估,拿到小体积的结果:

```bash
cd /content/HumanStudy-Bench && for f in 0 1 2 3 4; do python -m effect_algebra.evaluate_suite --model-label C_fold$f --base-model Qwen/Qwen2.5-14B-Instruct --adapter /content/ea/adapters/C_fold$f --dataset C_fold${f}_test=/content/ea/data/cv/C_fold${f}_test.jsonl --output-dir /content/ea/results/C_fold$f; done
```

```bash
for f in 0 1 2 3 4; do echo "=== fold $f ==="; python -m effect_algebra.digest --results-dir /content/ea/results/C_fold$f; done
```

**ceiling = fold 1–4 的均值**(fold 0 留给学习率选择,排除)。

某一折被打断了可以续:

```bash
python -m effect_algebra.train_soft --train-file /content/ea/data/cv/C_fold4_train.jsonl --eval-file /content/ea/data/cv/C_fold4_test.jsonl --output-dir /content/ea/adapters/C_fold4 --run-name c-fold4 --base-model Qwen/Qwen2.5-14B-Instruct --resume-from-checkpoint /content/ea/adapters/C_fold4/checkpoint-40
```

---

## 6. Gate 2:迁移(尚未实现,等代码更新)

计划是 `A → C`(远迁移)和 `D → C`(近迁移)。`D_train` 的数据生成还没写,
等 Gate 1 的 ceiling 确认后再加。届时本节会更新。

现在可以先跑的是 A 单源:

```bash
python -m effect_algebra.train_soft --train-file /content/ea/data/dpo/A_train.jsonl --eval-file /content/ea/data/eval/C_test.jsonl --output-dir /content/ea/adapters/A_soft --run-name a-soft --base-model Qwen/Qwen2.5-14B-Instruct
```

```bash
python -m effect_algebra.evaluate_suite --model-label A_soft --base-model Qwen/Qwen2.5-14B-Instruct --adapter /content/ea/adapters/A_soft --dataset C_test=/content/ea/data/eval/C_test.jsonl --dataset A_test=/content/ea/data/eval/A_test.jsonl --dataset D_test=/content/ea/data/eval/D_test.jsonl --output-dir /content/ea/results/A_soft
```

---

## 7. 预算

Colab Pro 每月 100 units。A100 约 11–12 units/h(**约 8.5 小时**),L4 约 4–5 units/h
(约 20 小时)。A100 快约 2 倍但单价约 2.5 倍,**它的价值是能装下 14B,不是更省**。

| 阶段 | 14B on A100 |
|---|---:|
| 数据 + 参照标尺 | 0(不用 GPU) |
| Gate 0 | 0.5 h |
| Gate 1(5 折) | 1.2 h |
| Gate 2(A→C, D→C) | 1.5 h |
| 消融 + 飞轮 | 3.5 h |
| **合计** | **6.7 h** |

32B 会超预算(约 11.5 h),7B 只要 4.2 h 但 Gate 0 显示它在 C 上差得多。**默认用 14B。**

---

## 8. 常见问题

| 症状 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'effect_algebra'` | 不在仓库目录 | `cd /content/HumanStudy-Bench` |
| `-bash: syntax error near unexpected token` | 把 Python 代码贴进了终端 | Drive 挂载只能在 notebook cell 里跑 |
| `NameError: name 'drive' is not defined` | `import` 那行没执行 | 两行放同一个 cell 一起跑 |
| pip 报 diffusers/gradio 冲突 | 预装包的依赖 | 忽略,我们不用这两个 |
| digest 输出是空表 | 结果目录不存在或为空 | 检查 `--results-dir` 路径 |
| 训练日志 loss 约 2.6 | 已知的日志缩放问题 | 见 REPORT_CN §6,不影响训练 |
| 终端显示 `[disconnected]` | 终端断了,VM 可能还活着 | 先 `ls /content/ea/adapters/` 确认 |

**评分逻辑修好之后不用重跑 GPU**:`digest` 会从保存的逐行分数重新计算指标。
拉一下代码再跑一次 `digest` 就是最新口径。

```bash
cd /content/HumanStudy-Bench && git pull && python -m effect_algebra.digest --results-dir /content/ea/results/qwen14b --reference-dir /content/ea/results/reference
```
