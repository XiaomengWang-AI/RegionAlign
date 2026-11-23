# 实验结果索引

`results/` 只新增、不进版本库，所以索引放在这里随代码一起维护。本文件只做导航：每个目录是什么、状态如何、用哪个代码版本能原样加载。
数值结论看各目录里的核查报告，不在这里重复。

## results/ 的结构

| 路径 | 内容 |
|---|---|
| `results/experiments/` | **全部实验结果**，按实验代码 → 实验 → 计划 → 条件组织，不区分在哪台服务器上跑 |
| `results/campaigns/`、`results/hardware_handoff_20261116/`、`results/imports/` | **旧路径别名，只读**。这里只有符号链接，指向 `experiments/` 里的同一份文件 |

**为什么保留旧路径别名：** 计划、日志、提交脚本和汇总文件里写着 2026-11-23 之前的绝对路径（如 `results/campaigns/<id>/plans/...`）。其中很多文件的 sha256 被计划绑定，不能改写。别名保证每个记录过的路径仍然指向同一份字节，所以从任何旧记录都能追到数据。
- 旧路径 → 新路径的逐文件对照表在 `results/experiments/_build/PATH_ALIASES.tsv`。
- 每个文件在 2026-11-23 整理前的来源和 sha256 在 `results/experiments/_build/BUILD_MANIFEST.tsv`。

**两条规则：**
- 新实验只写到 `results/experiments/` 下，不要再往别名目录里写。
- small4 里有 432 个单元在两台服务器各跑过一次（864 个文件），两份都保留，各自对应自己的冻结绑定：
  - `llm_matrix/small4/plan_main/` 与原计划目录逐字节相同（Snellius 版本）。在 Snellius 上准备的注入计划按 sha256 绑定的就是这份，旧路径 `campaigns/.../plans/snellius_reuse` 也指向它。
  - `llm_matrix/small4/plan_main_triton_copies/` 按相同相对路径存放这 864 个文件的 Triton 版本。按 2026-11-15 的决定，**正式数值一律用这份**（见 `llm_matrix/small4/checks/CANONICAL_COPY_20261115.md`）。汇总 `llm_matrix/small4/analysis/` 用的就是这份，在 Triton 上跑的两个 MMLU 注入计划（`injection/*/plan_mmlu`）也绑定这份；旧的 Triton 路径（`imports/triton_20261115/tree/...`）指向它。
  - 读取 small4 时：文件在 `plan_main_triton_copies/` 里有就用那份，没有就用 `plan_main/` 里的。
- 其他迁移审计记录已不保留。

文件约定：
- 一个**计划目录**（含 `plan.json`、`status.json`、`source/`）是可执行单元：`artifacts/<条件>/` 存搜索产物，`eval/<条件>/` 存评价报告，`inject/<条件>/` 存注入结果。
- 每个矩阵条件有 108 份评价 = 12 区域 × 3 重复 × 3 布局（noperm/local/random）。
- 条件 id 的格式是 `模型__格式__任务`，例如 `qwen2.5-1.5b__int8__mmlu`。

## 加载或续跑旧计划

计划冻结源码：`plan.json.sources` 记录每个源码文件的 sha256，代码一变就拒绝加载。下表的“可加载提交”是源码与该计划逐字节一致的 git 提交。
所有计划都产生于 2026-11-11 包分离之后。现在只有一个代码目录 `BFA_llm/regionalign`（分支 `vqa-dev-matrix`，合并了全部代码，新实验用它）。旧计划冻结的源码保存在下面这些分支里，也都推到了 GitHub：

| 分支 | 内容 |
|---|---|
| `small-gemma-vlm-int8` | Gemma/VLM 计划的冻结源码（`0d48464`），另加 `vqa_reeval`（`22688fc`） |
| `injection-on-d11cc3f` | `d11cc3f` 加上三个注入评价器（`d7a8784`），供 llama/qwen MMLU trace 计划使用 |

加载旧计划时，先把它的可加载提交检出成临时工作树，再把 `results` 链接回主目录。
旧的 `launch/*.sbatch` 写死了原来的代码路径，所以建议在原来的路径上重建工作树：

```bash
cd /gpfs/work2/0/prjs1717/xwang/BFA_llm/regionalign
git worktree add /gpfs/work2/0/prjs1717/xwang/BFA_llm_vlm/regionalign small-gemma-vlm-int8   # 或 injection-on-d11cc3f、或某个提交号
ln -s /gpfs/work2/0/prjs1717/xwang/BFA_llm/regionalign/results /gpfs/work2/0/prjs1717/xwang/BFA_llm_vlm/regionalign/results
# 运行时 PYTHONPATH=/gpfs/work2/0/prjs1717/xwang/BFA_llm_vlm；用完 git worktree remove 即可
```

计划可以用新路径加载，也可以用旧别名路径加载（2026-11-23 已逐个验证主矩阵与全部注入计划）。
**能加载，不等于能原样重放。** 下面两处例外：
- `vlm_matrix/int4_reeval` 的 `binding.json` 记录的是旧路径 `campaigns/.../plans/vqa`，所以续跑这批重评时，`--run` 必须写旧路径。
- `injection/{uniform,highbias}/plan_mmlu` 是在 Triton 上运行的。它们记录的 `source_run` 是 `campaigns/.../plans/snellius_reuse`，在这里解析到 Snellius 版本，但它们的 14 个 artifact 绑定的是 Triton 版本（`plan_main_triton_copies/`）。
  所以这两个计划能加载、能核对结果，但直接重跑其中的单元会报 `source artifact drift`。这两个计划已全部完成。真要重放，就在临时目录里用 `plan_main/` 加上 `plan_main_triton_copies/` 拼出 Triton 当时的源计划，并让 `source_run` 解析到那里；不要改计划文件，也不要改旧路径别名。

## llm_matrix/ — `experiments/llm.py`：LLM 的匹配矩阵（MMLU、WikiText）

| `results/experiments/` 下 | 内容 | 状态 | 可加载提交 | 旧路径 |
|---|---|---|---|---|
| `llm_matrix/small4/plan_main/`（正式副本覆盖层在 `plan_main_triton_copies/`） | llama3.2-1b、qwen2.5-1.5b、stablelm2-1.6b、smollm2-1.7b × INT8/INT4 × MMLU/WikiText，不含下一行的两个条件 | 14 × 108/108 | `ed52753`–`83e6cbd` | `campaigns/20261112T235241_652037Z_small4_int8_baseline/plans/snellius_reuse` |
| `llm_matrix/small4/plan_llama_qwen_int8_mmlu/` | llama3.2-1b、qwen2.5-1.5b 的 INT8 MMLU | 2 × 108/108 | `9399d68`–`d11cc3f` | 同上 `plans/snellius` |
| `llm_matrix/small4/checks/` | 终检 `final_int8_check_*`、`int4_matrix_summary_20261116.md`、`CANONICAL_COPY_20261115.md`；`scripts/` 是生成它们的脚本 | | | 同上 `verification/` |
| `llm_matrix/small4/analysis/` | `reporting/campaign_costs.py` 的汇总（`summary/`、`costs/`）；`code/` 是当时所用的 reporting 代码 | | | `campaigns/20261115T170101_892368Z_small4_int8_int4_summary` |
| `llm_matrix/small4/{launch,stages,preflight,tools}/` | 提交脚本与日志、阶段证据、预检 | | | 同上 small4 campaign |
| `llm_matrix/gemma/plan/` | gemma2-2b、gemma3-1b × INT8/INT4 × MMLU/WikiText | 8 × 108/108 | `0d48464`–`22688fc` | `campaigns/20261113T175625_031395Z_small_gemma_vlm_int8_baseline/plans/llm` |
| `llm_matrix/gemma/{checks,diagnostics,launch,inputs,preflight,tools}/` | 终检、gemma3 logit 探针、提交脚本与日志 | | | 同上 campaign |

## vlm_matrix/ — `experiments/vqa.py`、`vqa_reeval.py`：VLM 的匹配矩阵（OK-VQA、TextVQA）

| `results/experiments/` 下 | 内容 | 状态 | 可加载提交 | 旧路径 |
|---|---|---|---|---|
| `vlm_matrix/plan_smolvlm_qwen2.5vl/` | smolvlm-instruct、qwen2.5-vl-3b × INT8/INT4 × OK-VQA/TextVQA | 8 × 108/108 | `0d48464`–`22688fc` | `campaigns/20261113T175625_031395Z_small_gemma_vlm_int8_baseline/plans/vqa` |
| `vlm_matrix/plan_qwen2vl2b/` | qwen2-vl-2b × INT8/INT4 × OK-VQA/TextVQA | 4 × 108/108 | 同上 | 同上 `plans/vqa_qwen2vl2b` |
| `vlm_matrix/int4_reeval/` | 上面 6 个 INT4 条件里因 clean 分数为 0、在汇总时崩溃的 522 份报告的重评 | 522/522 | `22688fc` | 同上 `reeval/int4_vqa_none_safe_20261116` |
| `vlm_matrix/{checks,launch,inputs,preflight,tools}/` | 终检、提交脚本与日志、官方 OK-VQA 训练输入审计、预检 | | | 同上 campaign |

INT4 VLM 的读法：对每个条件，`int4_reeval/<vqa 或 vqa_qwen2vl2b>/eval/<条件>/` 里有的报告用重评版，没有的用计划目录 `eval/` 里状态为 `complete` 的报告，合起来是 108 份。
这些条件的 clean 分数为 0，success 字段不可用，只比较 ΔNLL。

## injection/ — `experiments/llm_injection*.py`：冻结布局上的单次随机注入 ASR（INT8）

| `results/experiments/` 下 | 评价器与故障法则 | 条件 | 结果数与可加载提交 | 旧路径（`campaigns/` 下） |
|---|---|---|---|---|
| `injection/uniform/plan_wikitext/` | `llm_injection.py`，按位均匀 | 4 个模型的 WikiText | 56 json；`f5eff1f`–`83e6cbd` | `20261114T_small4_int8_injection_pilot` |
| `injection/uniform/plan_mmlu/` | 同上 | smollm2、stablelm2 的 MMLU | 28 json；同上 | `20261114T_small4_int8_injection_pilot_aalto` |
| `injection/highbias/plan_wikitext/` | `llm_injection_biased.py`，偏向符号位和高位 | 4 个模型的 WikiText | 56 json；`24be8d7`–`83e6cbd` | `20261114T_small4_int8_injection_highbias_pilot` |
| `injection/highbias/plan_mmlu/` | 同上 | smollm2、stablelm2 的 MMLU | 28 json；同上 | `20261115T_small4_int8_injection_highbias_aalto` |
| `injection/trace/plan_main/{uniform,highbias}/` | `llm_injection_trace.py`，逐 trial 记录 ΔNLL | 6 个条件 × 2 种法则 | 168 json；`83e6cbd` | `20261115T_small4_int8_injection_trace` |
| `injection/trace/plan_llama_qwen_mmlu/{uniform,highbias}/` | 同上 | llama、qwen 的 MMLU | 56 json；`d7a8784` | `20261115T_small4_int8_injection_trace_llamaqwen_mmlu` |

注入区域的选法：在 12 个 dev 区域中，取 K=16 时 local 布局平均 dNLL 最大的区域（只计通过 clean gate 的布局，3 个重复取平均）；同一模型另一任务的试点已用过的区域跳过。
每个条件的区域写在各计划的 `regions.json`（trace）或 `plan.json` 里。`*/checks/` 里是 MMLU 注入的 ASR 核查报告。

## 其他

| `results/experiments/` 下 | 内容 | 旧路径 |
|---|---|---|
| `hardware_handoff/` | 给硬件验证方的交付包，入口是其中的 `README.md` | `hardware_handoff_20261116/` |
| `setup/` | 环境冻结清单（`freeze-*`、`pins-*`）、模型下载与 Arrow 生成脚本、下载校验、包分离时的自检日志 | |
| `superseded/llm_first_qwen_llama_plans/` | 最早的 qwen2.5/llama 计划（可加载提交 `5fbccd7`）和第一版 qwen 交付；已被 small4 取代 | `campaigns/20261111T224825_997587Z_small_int8_baseline` |
| `superseded/llm_small4_int4_cluster_attempt/` | 在另一台集群上的 INT4 尝试，只完成 462 份；正式 INT4 结果在 `llm_matrix/small4/plan_main/` | `campaigns/20261113T153525_223459Z_small4_int4_local` |
| `_build/` | 生成本目录与旧路径别名的脚本，以及 `BUILD_MANIFEST.tsv`、`PATH_ALIASES.tsv` | |
