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

**首选:把结果同步进 Git 仓库。** 结果 JSON 一整套只有约 1 MB,进版本控制正合适,
而且这样每次分析都能直接从仓库拉最新的,不用来回贴数字。见 §9。

**另外两个选项(只保 adapter 用):**

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

**每月 300 credits,A100 计费 5.3 credits/小时 → 约 56 A100-小时。预算不是约束。**

| 阶段 | 14B on A100 |
|---|---:|
| 数据 + 参照标尺 | 0(不用 GPU) |
| Gate 0 | 0.5 h |
| Gate 1(5 折 + 评估) | 1.4 h |
| Gate 2(A→C, D→C) | 1.5 h |
| 7B / 32B 补充规模趋势 | 1.2 h |
| DPO 对照消融 | 1.5 h |
| 完整飞轮 8 条件 | 4.0 h |
| **合计** | **约 10 h(占额度 17%)** |

默认 **14B**:Gate 0 显示 7B 在 C 上差得多,而 32B 的边际收益主要在规模趋势那一条线上。
额度充裕,所以 32B 也值得跑一次 Gate 0(见 REPORT_CN §8 优先级 6)。

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

---

## 9. 把结果同步进仓库

评估结果是**唯一无法重新生成**的产物(数据确定性重建、adapter 可从数据重训),
而且一整套只有约 1 MB。收集脚本会把它们归档到 `effect_algebra/records/<标签>/`,
并刷新一份跨运行的索引。

每跑完一个阶段:

```bash
cd /content/HumanStudy-Bench && python -m effect_algebra.collect_results --from /content/ea/results/qwen14b --label qwen14b_gate0
```

Gate 1 的五折:

```bash
cd /content/HumanStudy-Bench && for f in 0 1 2 3 4; do python -m effect_algebra.collect_results --from /content/ea/results/C_fold$f --label gate1_C_fold$f; done
```

看跨运行汇总表:

```bash
python -m effect_algebra.collect_results --summary
```

### 推送到 GitHub

需要一个 GitHub personal access token。**这个 token 只贴进 Colab,不要贴进聊天、
不要写进任何会提交的文件。** 建议用 fine-grained token,权限只给
`HumanStudy-Bench` 一个仓库的 `Contents: Read and write`,并设一个短的过期时间。
组织仓库还可能需要管理员在 Settings → Third-party Access 里批准该 token。

先在**终端**里提交(不需要 token):

```bash
cd /content/HumanStudy-Bench && git config user.email "you@example.com" && git config user.name "Your Name"
```

```bash
cd /content/HumanStudy-Bench && git add effect_algebra/records && git commit -m "results: gate0 and gate1 on qwen14b"
```

然后在 **notebook cell**(不是终端)里推送:

```python
import getpass, subprocess
token = getpass.getpass("GitHub token: ")
url = f"https://{token}@github.com/HumanStudy-Hub/HumanStudy-Bench.git"
r = subprocess.run(["git", "-C", "/content/HumanStudy-Bench", "push", url, "pipeline"],
                   capture_output=True, text=True)
print((r.stdout + r.stderr).replace(token, "***"))
del token
```

> **必须在 notebook cell 里推,不能在终端。** Colab 的终端是独立进程,
> 不继承 notebook kernel 的环境变量,在 cell 里 `os.environ["GH_TOKEN"]=...`
> 之后到终端里 `$GH_TOKEN` 是空的,URL 会退化成 `https://@github.com/...`,
> git 转而提示输入用户名密码并失败。
>
> 上面的写法把 token 作为 `subprocess` 的参数传递而非经过 shell,所以
> **不会进入 shell 历史**,并且在打印前被替换成 `***`。

### 不想碰 token 的话

打包下载,把 tgz 放到本机仓库目录下,剩下的交给本地处理:

```bash
cd /content/ea && tar czf /content/records.tgz results && ls -la /content/records.tgz
```

在 Colab 左侧文件浏览器里右键 Download,存到本机的 `HumanStudy-Bench/` 目录即可。

---

## 10. 其他

**评分逻辑修好之后不用重跑 GPU**:`digest` 会从保存的逐行分数重新计算指标。
拉一下代码再跑一次 `digest` 就是最新口径。

```bash
cd /content/HumanStudy-Bench && git pull && python -m effect_algebra.digest --results-dir /content/ea/results/qwen14b --reference-dir /content/ea/results/reference
```
