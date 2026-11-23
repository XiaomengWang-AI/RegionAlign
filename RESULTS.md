# 实验结果索引（2026-11-23 整理）

`results/` 只新增、不进版本库，所以索引放在这里随代码一起维护。本文件只做导航：每个目录是什么、状态如何、用哪个代码版本能原样加载。
数值结论看各目录里的核查报告，不在这里重复。

## 怎么读

- 下文路径都相对于 `results/`；“主结果”和“注入 ASR”两张表里的目录都在 `results/campaigns/` 下。一个 **campaign** 目录 = 一次实验活动；里面的 **plan** 目录（含 `plan.json`）才是可执行单元。
- **plan 冻结源码。** `plan.json.sources` 记录每个源码文件的 sha256，代码一变就拒绝加载。
  表中所有 campaign 都产生于 2026-11-11 包分离之后（分离之前的 run 已不在本目录）。“可加载提交”是源码与该 plan 逐字节一致的 git 提交；要续跑或重评，按下一节在该提交的工作树里运行。
- 文件约定：
  - `artifacts/<条件>/` 是搜索产物，`eval/<条件>/` 是评价报告。每个条件 108 份 = 12 区域 × 3 重复 × 3 布局。
  - `inject/<条件>/` 是注入 ASR 结果。
  - `launch/` 放 sbatch 与日志，`verification/` 放核查报告。
- **条件 id** 的格式是 `模型__格式__任务`，例如 `qwen2.5-1.5b__int8__mmlu`。

## 代码版本（git 分支）

现在只有一个目录 `BFA_llm/regionalign`，当前分支是 `vqa-dev-matrix`。旧计划冻结的源码都保存在 git 分支里（也已推到 GitHub）：

| 分支 | 内容 |
|---|---|
| `vqa-dev-matrix` | 当前主线：合并了全部代码，包括 Gemma/VLM、三种注入评价器和 `vqa_reeval`。新实验用它 |
| `small-gemma-vlm-int8` | Gemma/VLM 计划的冻结源码（`0d48464`），另加 `vqa_reeval`（`22688fc`） |
| `injection-on-d11cc3f` | `d11cc3f` 加三个注入评价器（`d7a8784`），供 llama/qwen MMLU trace 计划使用 |

要加载一个旧计划（续跑或重评），先把它的可加载提交检出成临时工作树，并把 `results` 链接回主目录。
旧的 `launch/*.sbatch` 写死了原来的路径，所以建议用原路径重建：

```bash
cd /gpfs/work2/0/prjs1717/xwang/BFA_llm/regionalign
git worktree add /gpfs/work2/0/prjs1717/xwang/BFA_llm_vlm/regionalign small-gemma-vlm-int8   # 或 injection-on-d11cc3f、或某个提交号
ln -s /gpfs/work2/0/prjs1717/xwang/BFA_llm/regionalign/results /gpfs/work2/0/prjs1717/xwang/BFA_llm_vlm/regionalign/results
# 运行时 PYTHONPATH=/gpfs/work2/0/prjs1717/xwang/BFA_llm_vlm；用完 git worktree remove 即可
```

原先的 `BFA_llm_vlm/regionalign` 用 `small-gemma-vlm-int8`，`BFA_llm_inj_d11/regionalign` 用 `injection-on-d11cc3f`。

## 主结果（论文用）

| campaign / plan | 内容 | 状态 | 可加载提交 | 核查与汇总 |
|---|---|---|---|---|
| `20261112T235241_652037Z_small4_int8_baseline/plans/snellius` | llama3.2-1b、qwen2.5-1.5b 的 INT8 MMLU | 2 条件，各 108/108 | `9399d68`–`d11cc3f` | `verification/final_int8_check_*.md` |
| 同上 `plans/snellius_reuse` | 4 个小 LLM（llama3.2-1b、qwen2.5-1.5b、stablelm2-1.6b、smollm2-1.7b）× INT8/INT4 × MMLU/WikiText；其中 llama/qwen 的 INT8 MMLU 在上一行 | 14 条件，各 108/108 | `ed52753`–`83e6cbd` | `verification/final_int8_check_*.md`、`int4_matrix_summary_20261116.md`、`CANONICAL_COPY_20261115.md`（26 个单元以 Triton 副本为准，见 `imports/triton_20261115/canonical.py`） |
| `20261113T175625_031395Z_small_gemma_vlm_int8_baseline/plans/llm` | gemma2-2b、gemma3-1b × INT8/INT4 × MMLU/WikiText | 8 条件，各 108/108 | `0d48464`–`22688fc` | `verification/merged_final_check_*.md` |
| 同上 `plans/vqa` | smolvlm-instruct、qwen2.5-vl-3b × INT8/INT4 × OK-VQA/TextVQA | 8 条件，各 108/108 | 同上 | 同上 |
| 同上 `plans/vqa_qwen2vl2b` | qwen2-vl-2b × INT8/INT4 × OK-VQA/TextVQA | 4 条件，各 108/108 | 同上 | 同上 |
| 同上 `reeval/int4_vqa_none_safe_20261116/` | 上两行 INT4 VLM 里因 clean 分数为 0、在汇总时崩溃的 522 份报告，复用搜索产物重评（`experiments/vqa_reeval.py`） | 522/522 完成 | `22688fc` | 与原 `eval/` 中 `complete` 的报告合起来，每个条件 108 份；success 不可用，只看 ΔNLL |

## 注入 ASR（INT8，单区域、随机故障）

| campaign | 评价器 / 故障法则 | 条件 | 状态 | 可加载提交 |
|---|---|---|---|---|
| `20261114T_small4_int8_injection_pilot` | `llm_injection`，uniform | 4 模型 WikiText（计划含 6 条件，MMLU 两条在下一行的 Triton 跑） | 56 json | `f5eff1f`–`83e6cbd` |
| `20261114T_small4_int8_injection_pilot_aalto` | 同上，在 Triton 运行 | smollm2/stablelm2 MMLU | 28 json | 同上 |
| `20261114T_small4_int8_injection_highbias_pilot` | `llm_injection_biased`，highbias | 4 模型 WikiText（MMLU 两条见下一行） | 56 json | `24be8d7`–`83e6cbd` |
| `20261115T_small4_int8_injection_highbias_aalto` | 同上，在 Triton 运行 | smollm2/stablelm2 MMLU | 28 json | 同上 |
| `20261115T_small4_int8_injection_trace/{uniform,highbias}` | `llm_injection_trace`，逐 trial ΔNLL | 6 条件 × 2 种法则 | 168 json | `83e6cbd` |
| `20261115T_small4_int8_injection_trace_llamaqwen_mmlu/{uniform,highbias}` | 同上 | llama/qwen MMLU | 56 json | `d7a8784`（分支 `injection-on-d11cc3f`） |

注入区域的选法：在 12 个 dev 区域中，取 K=16 时 local 布局平均 dNLL 最大的区域（只计通过 clean gate 的布局，3 个重复取平均）；同一模型另一任务的试点已用过的区域跳过。
每个条件的区域写在各 campaign 的 `regions.json`（trace）或 `plan.json` 里。

## 汇总、论文与交付

| 目录 | 内容 |
|---|---|
| `campaigns/20261115T170101_892368Z_small4_int8_int4_summary/` | `reporting/campaign_costs.py` 对 small4 的汇总：`summary/`、`costs/` 下的 CSV 与图，来源记录在 `merge_provenance.json` |
| `paper_int8_20261114/` | 论文素材：`extract.py` 生成 `rows*.json`，`figs*.py` 出图，加上 `.tex` 段落；INT4 部分见 `INT4_INTEGRATION_20261116.md` |
| `hardware_handoff_20261116/` | 给硬件验证方的交付包：每个模型的区域、置换、翻位与权重码，以及 ASR 脚本；入口是 `README.md` |

## 早期或被取代的（只作证据保留）

| 目录 | 说明 |
|---|---|
| `campaigns/20261111T224825_997587Z_small_int8_baseline/` | 只有 qwen2.5/llama 的第一版计划，`snellius_fresh` 跑了 108 份；已被 small4 取代。可加载提交 `5fbccd7` |
| `campaigns/20261113T153525_223459Z_small4_int4_local/` | 在另一台本地集群上的 INT4 尝试，只完成 462 份（`plans/local_reuse`）；正式 INT4 结果以 `snellius_reuse` 为准 |

## 迁移与审计（`imports/`）

| 目录 | 内容 |
|---|---|
| `triton_20261115/` | Triton 全部结果的已校验副本及并入记录（`MERGE.md`）；`tree/` 保存冲突时的 Triton 版本 |
| `triton_highbias_20261115/` | Triton highbias 注入结果的传输与校验记录 |
| `trace_mmlu_to_triton_20261115/` | 推给 Triton 的 trace 代码；那批作业后来在 Snellius 上完成 |
| `triton_inventory_20261123/` | 2026-11-23 的 Triton 清单（代码 sha256、全部结果文件、作业）。比对结论：Triton 上的结果 Snellius 都有 |
| `triton_tools_20261123/` | Triton 独有的两份工具文件存档（`ARCHIVE.md`） |

## 归并存档（`archive_20261123/`）

2026-11-23 归并成一个目录时，从包外移进来的东西：包分离时的自检日志、第一版 qwen 计划（`transfer_in/qwen_delivery`）、small4 汇总所用的 reporting 代码快照（不在 git 里，sha256 与 `merge_provenance.json` 一致）以及 VLM 工作树的审查记录。详见该目录的 `README.md`。
