# RegionAlign

本目录维护预定区域下的布局对准研究：同一模型、任务、4 KiB 区域、翻位预算和搜索预算下，比较 `local`、`noperm` 与 `random`。
攻击者只先拿到区域：区域内任意存储位都可翻、方向不限，没有可翻单元图。
合法重排必须先通过 clean gate；故障损伤相对于同一重排布局的无故障模型计算。

## 目录与文件用途

| 文件或目录 | 用途 |
|---|---|
| `AGENTS.md`、`CLAUDE.md` | 操作契约：研究假设、可改范围、自检与交叉审查 |
| `RESULTS.md` | 实验结果索引：每个 campaign/plan 的内容、状态、核查报告与可加载提交 |
| `config.py` | 本包自有的路径、离线环境设置和只新增的结果路径 |
| `runtime.json` | 本机配置：`hf_home` 指向模型与数据缓存 |
| `requirements.txt` | 本包直接依赖及实验所用版本 |
| `requirements-tf451.txt` | 第二套环境（transformers 4.51.3）：Gemma-3、SmolVLM、Qwen2.5-VL；计划冻结库版本 |
| `harness/core.py` | 自包含内核：权重量化、合法置换（含 Gemma 的 head_dim 与额外残差归一化）、翻位与 in-range 校验、基础 Runtime 及逐位不变的前向复用 |
| `harness/region.py` | 通用区域几何、合法布局、区域校验与计算预算上下文（含重复行与层前缀复用） |
| `harness/evidence.py` | JSON 数值清理等证据辅助函数 |
| `algorithm/search.py` | 正式区域实验共用的搜索算法 |
| `runtimes/text.py` | MMLU/WikiText 冻结 token 面板及正式文本目标 |
| `runtimes/vqa.py` | LLaVA/Qwen2-VL/Qwen2.5-VL/SmolVLM 的训练集面板、多模态计算及评分适配；完整模型与缓存图像特征两种运行时 |
| `runtimes/vqa_score.py` | 官方 VQA 答案归一化与一致性准确率 |
| `experiments/llm.py` | 已登记 LLM（计划可取任意非空子集）的正式开发集匹配矩阵 |
| `experiments/vqa.py` | 已登记开发 VLM 的正式开发集匹配矩阵、官方 OK-VQA 训练输入绑定及 GPU preflight |
| `experiments/vqa_reeval.py` | 复用冻结 VQA 计划的搜索结果，对因 clean 分数为 0 而在汇总时崩溃的评价，以不比较 None 的汇总重新评价，写入单独目录 |
| `experiments/official_okvqa.py` | 官方训练来源的独立归档、面板选择审计与四条件 parent |
| `experiments/clean_mmlu.py` | LLM 登记表、无故障 MMLU 量化基线、模型加载及输入/源码审计 |
| `experiments/clean_wikitext.py` | 无故障 WikiText 开发集基线 |
| `experiments/clean_benchmarks.py` | 完整 WikiText/VQA 无故障最终基线及按模型族选择的共用多模态加载器 |
| `experiments/clean_memory.py` | clean 基线的独立显存策略与合成内存标定入口 |
| `experiments/campaign_protocol.py` | 原字节继承面板的新计划、显式资源策略和独立计算预算 |
| `experiments/campaign_variants.py` | 消融变体的冻结注册项、参考基线与开发阶段执行 |
| `experiments/campaign_stage.py` | 阶段注册覆盖核验与独立完成标识 |
| `experiments/bounded_checksum.py` | 保持整数指纹值的分块内存实现 |
| `experiments/campaign_memory.py` | 分块指纹实现的独立计划身份与执行包装 |
| `experiments/layout_validation.py` | 真实权重的无故障布局等价性与独立梯度资源资格检查 |
| `experiments/nohup_campaign.py` | 冻结队列的依赖调度、Slurm 分配、nohup 和逐次执行证据 |
| `experiments/campaign_watch.py` | 等待外部完成证据时的队列守候与恢复 |
| `experiments/timeout_recovery.py` | 有明确 Slurm 时限证据的主队列有限续跑 |
| `experiments/campaign_evaluation.py` | 全部开发产物封存后的最终与跨任务独立评价 |
| `experiments/llm_injection.py` | 冻结 LLM 布局的单次注入 ASR 评价（区域内随机故障、冻结故障法则）|
| `experiments/llm_injection_biased.py` | 同上，但故障法则按位的位值 2^b 加权偏向符号/高位（每位仍有正概率，不排除任何位）；独立 scope 与计划，复用其无关法则的逻辑 |
| `experiments/llm_injection_trace.py` | 与上两者完全同威胁模型/故障法则/成功判据，仅额外逐 trial 记录 ΔNLL、保留翻位数与类别（供损伤超越曲线 Fig. B）；法则由 plan 字段 `fault_law_kind`（uniform/highbias）选择，独立 scope，复用两基座评价器的无关法则辅助函数，不改其字节 |
| `experiments/task_transfer.py` | 两份独立 VQA 开发计划之间的双向任务迁移评价 |
| `experiments/checkpoint_transfer.py` | 同架构 checkpoint 之间的独立迁移评价 |
| `experiments/input_audit.py` | 模型/数据缓存审计及显式输入下载（含全部已登记 LLM，可固定 Hub 版本） |
| `reporting/mmlu.py` | 完整 clean MMLU 结果的表格、图与校验 |
| `reporting/clean_matrix.py` | 已有 clean 矩阵汇总；不代替正式 RegionAlign 的状态检查 |
| `reporting/campaign_costs.py` | 冻结注册矩阵的搜索成本、开发损伤、匹配对照与覆盖汇总 |
| `tests/helpers.py` | 测试共用的小型随机模型与 Runtime 夹具 |
| `tests/test_core.py` | 内核测试：翻位可逆、置换保函数与保码（含 LayerNorm 解码器）、量化完整、in-range 规则 |
| `tests/test_region_core.py` | 通用区域、合法置换、算法接口、复用逐位一致及包自包含性测试 |
| `tests/test_llm_region_study.py` | 正式文本矩阵、匹配重复、隔离与源码快照测试 |
| `tests/test_vqa_region_study.py` | 正式 VQA 矩阵、输入身份与评价规则测试 |
| `tests/test_vqa_reeval.py` | None 安全汇总与原汇总等价、只挑汇总崩溃的报告 |
| `tests/test_official_okvqa.py` | 官方训练选择等价、来源绑定、图片失败不替换与新 parent 测试 |
| `tests/test_matrix_vqa_runtime.py` | 多模态运行时、GQA 与视觉通道同步测试 |
| `tests/test_vqa_cached_runtime.py` | 缓存图像特征运行时与完整模型的一致性、布局与复用测试 |
| `tests/test_decoder_families.py` | Gemma 结构（显式 head_dim、额外归一化、soft-cap）的内核规则测试 |
| `tests/test_mmlu_clean.py` | clean MMLU 评分、量化及证据协议测试 |
| `tests/test_clean_benchmarks.py` | 最终 clean 基线、预检及逐样本完整性测试 |
| `tests/test_clean_memory.py` | 显存策略隔离、计划身份与标定失败路径测试 |
| `tests/test_campaign_protocol.py` | 开发计划继承、独立预算与资源策略测试 |
| `tests/test_campaign_variants.py` | 消融注册范围、参考基线及变体身份测试 |
| `tests/test_campaign_stage.py` | 多文件阶段覆盖、证据绑定与恢复测试 |
| `tests/test_bounded_checksum.py` | 分块整数指纹等值性与临时内存边界测试 |
| `tests/test_campaign_memory.py` | 分块执行策略的源码冻结和包装测试 |
| `tests/test_layout_validation.py` | 布局等价性与资格结果状态测试 |
| `tests/test_llm_injection.py` | 注入 ASR 故障采样、成功判据、计划哈希与分片命名测试 |
| `tests/test_llm_injection_biased.py` | 偏置故障法则的高位采样、per-element 丢弃不变量与独立协议常量测试 |
| `tests/test_llm_injection_trace.py` | trace 评价器的法则映射、逐 trial 分类与基座一致、独立 scope 及源码快照测试 |
| `tests/test_nohup_campaign.py` | 队列依赖、资源分配、锁与失败恢复测试 |
| `tests/test_campaign_watch.py` | 队列守候、锁与外部证据到达测试 |
| `tests/test_timeout_recovery.py` | 调度超时证据、恢复次数与原阶段续跑测试 |
| `tests/test_campaign_evaluation.py` | 最终访问门槛、封存选择与后续评价身份测试 |
| `tests/test_campaign_followups.py` | 真实小型量化运行时的跨任务独立评价测试 |
| `tests/test_campaign_costs.py` | 搜索成本核算、缺失覆盖与匹配对照隔离测试 |
| `tests/test_task_transfer.py` | VQA 任务迁移的独立元数据及真实小型 INT8/INT4 运行时检查 |
| `tests/test_checkpoint_transfer.py` | 用不同的小 checkpoint 和真实评价器测试同一性迁移 |
| `tests/test_wikitext_clean_dev.py` | 开发集文本隔离与 token 加权测试 |
| `tests/test_config.py` | 输出边界、离线环境、拒绝覆盖及下载范围测试 |
| `results/` | 全部实验计划、结果、日志与迁移审计；只新增，不进版本库；索引见 `RESULTS.md` |
| `.claude/review-loop/` | Codex 只读交叉审查脚本 |

包初始化不启动实验。本包自包含：不导入包外代码，唯一的外部输入是 `runtime.json` 里 `hf_home` 指向的缓存。
命令在包的上一级目录以 `python -m regionalign...` 运行；依赖见 `requirements.txt`，Gemma-3、SmolVLM、Qwen2.5-VL 用 `requirements-tf451.txt` 的环境。

## 按用途找代码

| 我要…… | 入口模块 | 子命令 / 说明 |
|---|---|---|
| 跑 LLM 的 noperm/local/random 矩阵（MMLU、WikiText） | `experiments/llm.py` | `plan` → `preflight` → `search` → `evaluate`，用 `status` 查进度 |
| 跑 VLM 矩阵（OK-VQA、TextVQA） | `experiments/vqa.py` | 同上；`evaluate` 之前必须先在真实 GPU 上跑 `preflight` |
| 重评因 clean 分数为 0 而汇总崩溃的 VQA 报告 | `experiments/vqa_reeval.py` | 先 `init` 建一个新的输出目录，再按分片 `run` |
| 在冻结布局上测单次随机注入的 ASR | `experiments/llm_injection_trace.py`（新实验用它） | `prepare --source-run <LLM 计划> --law uniform\|highbias --regions '<JSON>'`，再 `run`；`llm_injection.py`、`llm_injection_biased.py` 是它复用的基础评价器 |
| 只要无故障（clean）基线 | `experiments/clean_mmlu.py`、`clean_wikitext.py`、`clean_benchmarks.py` | |
| 下载或审计模型与数据 | `experiments/input_audit.py` | |
| 汇总成本、损伤和匹配对照 | `reporting/campaign_costs.py` | |
| 改搜索算法 / 量化与翻位规则 / 区域几何 | `algorithm/search.py` / `harness/core.py` / `harness/region.py` | 这几处一改，全部旧计划都不能再加载，改前先读 `AGENTS.md` |

环境：Llama、Qwen2.5、StableLM2、SmolLM2、Gemma-2 用 `requirements.txt`；Gemma-3、SmolVLM、Qwen2.5-VL、Qwen2-VL 用 `requirements-tf451.txt`（Snellius 上是 `/projects/prjs1717/xwang/env/bfa_tf451`）。
Slurm 提交脚本的现成模板在各 campaign 的 `launch/snellius/*.sbatch` 里（每个 GPU 跑 PACK 个分片，先 search 后 evaluate）。

## 结果在哪

**`RESULTS.md` 是结果索引**：每个 campaign 和 plan 是什么、完成到哪一步、核查报告在哪，以及要用哪个 git 提交才能原样加载（plan 冻结了源码 sha256）。
每个 plan 目录里，`plan.json` 登记协议，`source/` 存当时的源码快照，`artifacts/` 存搜索产物，`eval/` 存独立评价，`inject/` 存注入结果。
目录或计划存在，不表示实验已经完成；以注册项的覆盖和有效报告为准。

## 保留边界与历史版本

- 本包于 2026-11-11 从 `layout_alignment_attack` 仓库分离。内核、路径配置和 VQA 评分已收进包内；内核只保留 in-range 一条数值规则，其余定义逐字未改。
- 那个仓库研究的是另一个假设（可翻单元图与受限轨道）。它的代码、文档与编号都不属于本包，本包不导入也不引用。
- 分离之前产生的 run 原样保留为证据，不能续跑，也不能通过改写哈希来续用。
- 分离之后的每个 plan 绑定生成时的源码身份：只能在 `RESULTS.md` 列出的对应提交上加载，不能改用新代码；新代码要生成新计划。
- 保留 run 的历史源码快照可能带有当时的包外依赖；这些快照作为证据原字节保留，不作为当前实验入口。
- 最终 test 访问追加到 `results/test_access.jsonl`。

## 验证与阅读顺序

先读 `AGENTS.md`，再读 `harness/core.py`、`harness/region.py`、所用 Runtime、`algorithm/search.py` 和对应 experiment；实现细节以各模块 docstring 为准。

在包的上一级目录运行 CPU 回归：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m unittest discover -s regionalign/tests -t . -v
```

操作要求与证据约束以本目录 `AGENTS.md` 为准。
