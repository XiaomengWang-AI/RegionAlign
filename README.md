# RegionAlign

本目录维护预定区域下的布局对准研究：同一模型、任务、4 KiB 区域、翻位预算和搜索预算下，比较 `local`、`noperm` 与 `random`。
攻击者只先拿到区域：区域内任意存储位都可翻、方向不限，没有可翻单元图。
合法重排必须先通过 clean gate；故障损伤相对于同一重排布局的无故障模型计算。

## 目录与文件用途

| 文件或目录 | 用途 |
|---|---|
| `AGENTS.md`、`CLAUDE.md` | 操作契约：研究假设、可改范围、自检与交叉审查 |
| `config.py` | 本包自有的路径、离线环境设置和只新增的结果路径 |
| `runtime.json` | 本机配置：`hf_home` 指向模型与数据缓存 |
| `requirements.txt` | 本包直接依赖及实验所用版本 |
| `harness/core.py` | 自包含内核：权重量化、合法置换、翻位与 in-range 校验、基础 Runtime 及逐位不变的前向复用 |
| `harness/region.py` | 通用区域几何、合法布局、区域校验与计算预算上下文（含重复行与层前缀复用） |
| `harness/evidence.py` | JSON 数值清理等证据辅助函数 |
| `algorithm/search.py` | 正式区域实验共用的搜索算法 |
| `runtimes/text.py` | MMLU/WikiText 冻结 token 面板及正式文本目标 |
| `runtimes/vqa.py` | LLaVA/Qwen2-VL 的训练集面板、多模态计算及评分适配 |
| `runtimes/vqa_score.py` | 官方 VQA 答案归一化与一致性准确率 |
| `experiments/llm.py` | 已登记 LLM（计划可取任意非空子集）的正式开发集匹配矩阵 |
| `experiments/vqa.py` | LLaVA/Qwen2-VL 的正式开发集匹配矩阵及 GPU preflight |
| `experiments/official_okvqa.py` | 官方训练来源的独立归档、面板选择审计与四条件 parent |
| `experiments/clean_mmlu.py` | 无故障 MMLU 量化基线、模型加载及输入/源码审计 |
| `experiments/clean_wikitext.py` | 无故障 WikiText 开发集基线 |
| `experiments/clean_benchmarks.py` | 完整 WikiText/VQA 无故障最终基线及共用多模态加载器 |
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
| `tests/test_official_okvqa.py` | 官方训练选择等价、来源绑定、图片失败不替换与新 parent 测试 |
| `tests/test_matrix_vqa_runtime.py` | 多模态运行时、GQA 与视觉通道同步测试 |
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
| `tests/test_wikitext_clean_dev.py` | 开发集文本隔离与 token 加权测试 |
| `tests/test_config.py` | 输出边界、离线环境、拒绝覆盖及下载范围测试 |
| `results/` | 分离前产生的结果、运行计划、日志与迁移审计；只新增，不进版本库 |
| `.claude/review-loop/` | Codex 只读交叉审查脚本 |

包初始化不启动实验。本包自包含：不导入包外代码，唯一的外部输入是 `runtime.json` 里 `hf_home` 指向的缓存。
命令在包的上一级目录以 `python -m regionalign...` 运行；依赖见 `requirements.txt`。

## 数据与结果导航

| 结果分组 | 内容 |
|---|---|
| `results/llm_region_study/` | 正式 LLM 开发矩阵的冻结计划与后续产物 |
| `results/vqa_region_study/` | 正式 VQA 开发矩阵；缺失 train 输入的条件仍显式保留 |
| `results/mmlu_clean/` | clean MMLU 条件结果与汇总 |
| `results/experimental_matrix/` | 输入审计、WikiText clean 结果及早期 clean 汇总 |
| `results/clean_benchmark_submissions/` | 完整 clean 基线、失败保留及重试计划 |
| `results/clean_evaluation/` | 输入与执行准备的证据 |
| `results/campaigns/` | 冻结计划注册表、执行队列、阶段证据、后续评价与独立资源轮次 |
| `results/nohup_launches/` | 运行启动脚本、登记和日志；日志不是完成结果 |
| `results/migration/` | 旧新路径映射、逐文件 SHA256、暂停记录和迁移核验 |

各 run 的 `plan.json` 登记协议；`source/` 或 `sources/` 保存当时的源码；`artifacts/` 保存搜索产物；`eval/` 保存独立评价；存在的 `summary.json` 保存汇总。
目录或计划存在不表示实验完成；判断以注册项覆盖和有效报告为准。

## 保留边界与历史版本

- 本包于 2026-11-11 从 `layout_alignment_attack` 仓库分离。内核、路径配置和 VQA 评分已收进包内；内核只保留 in-range 一条数值规则，其余定义逐字未改。
- 那个仓库研究的是另一个假设（可翻单元图与受限轨道）。它的代码、文档与编号都不属于本包，本包不导入也不引用。
- `results/` 里的 run 都产生于分离之前，绑定当时的源码身份和旧路径。它们的 `plan.json`、源码快照、哈希、输入、日志和指标原样保留。
- 旧计划不能续跑，也不能通过重写哈希继续使用；新代码须生成新计划。
- 保留 run 的历史源码快照可能带有当时的包外依赖；这些快照作为证据原字节保留，不作为当前实验入口。
- `results/migration/` 记录的是更早一次（2026-11-10）结果搬迁与单 bit 枚举结果删除，与本次分离无关。
- 最终 test 访问追加到 `results/test_access.jsonl`。

## 验证与阅读顺序

先读 `AGENTS.md`，再读 `harness/core.py`、`harness/region.py`、所用 Runtime、`algorithm/search.py` 和对应 experiment；实现细节以各模块 docstring 为准。

在包的上一级目录运行 CPU 回归：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m unittest discover -s regionalign/tests -t . -v
```

操作要求与证据约束以本目录 `AGENTS.md` 为准。
