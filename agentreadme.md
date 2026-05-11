# Agent 框架全览

## 公共基础设施

| 组件 | 文件 | 作用 |
|------|------|------|
| `CtrlHGenAdapter` | `ctrlhgen_adapter.py` | 加载 GPT2 checkpoint；提供 `build_model_input`（shifted token序列）和 `generate`（beam search）接口 |
| `format_conversion_tool` | `tools.py` | 观测实体名称 + 条件列表 → tokenized `source_text`（shifted）；条件类型见下表 |
| `generate_hypothesis_tool` | `tools.py` | `source_text` → 模型推理 → unshifted action string + `query_nl` |
| `compute_metrics` | `tools.py` | 在 KG 上执行假设，返回 `jaccard/dice/overlap`，`pred_answers/label_answers` |
| `execute_and_diagnose_tool` | `tools.py` | 执行假设，计算 TP/FP/FN，诊断：`too_broad / too_narrow / wrong_predicates / good` |

### 条件类型（`format_conversion` 的 `conditions` 参数）

```json
[
  {"type": "unconditional",  "value": null},
  {"type": "pattern",        "value": "i p e p e"},
  {"type": "relation",       "value": "GG"},
  {"type": "entity",         "value": "pex19"},
  {"type": "relationnumber", "value": 2},
  {"type": "entitynumber",   "value": 3}
]
```

---

## 13 种逻辑模式

| 模式 | 结构 | 分解策略 |
|------|------|----------|
| 1p | `(p,(e))` | 直接验证单路径 |
| 2p | `(p,(p,(e)))` | 内层 `(p,(e))` 先执行，结果作为中间实体集，再执行外层 p；用 `two_hop_candidates` 找好的 `(hop1_rel, hop2_rel, anchor)` |
| 2i | `(i,(p,(e)),(p,(e)))` | 两个分支独立验证，取交集 |
| 3i | `(i,(p,(e)),(p,(e)),(p,(e)))` | 三个分支独立验证，取交集 |
| ip | `(p,(i,(p,(e)),(p,(e))))` | 内层 2i 先执行，外层 p 作用于交集结果；用 `two_hop_candidates` 找外层关系 |
| pi | `(i,(p,(e)),(p,(p,(e))))` | 一个 1p 分支 + 一个 2p 链分支；用 `two_hop_candidates` 找链分支 |
| 2u | `(u,(p,(e)),(p,(e)))` | 两个分支独立验证，取并集；引入额外 OR 分支可扩大覆盖而不破坏正确性 |
| up | `(p,(u,(p,(e)),(p,(e))))` | 内层 2u 先执行，外层 p 作用于并集结果 |
| 2in | `(i,(n,(p,(e))),(p,(e)))` | 正分支决定主覆盖，否定分支仅用于收窄范围 |
| 3in | `(i,(i,(n,(p,(e))),(p,(e))),(p,(e)))` | 内层 2in + 外层正分支 |
| inp | `(p,(i,(n,(p,(e))),(p,(e))))` | 内层 2in 结果再经外层 p 链 |
| pni | `(i,(n,(p,(p,(e)))),(p,(e)))` | 否定分支是 2p 链，正分支是 1p |
| pin | `(i,(n,(p,(e))),(p,(p,(e))))` | 否定分支是 1p，正分支是 2p 链；用 `two_hop_candidates` 找正分支 |

### 修复 Insight

1. **交集/并集子逻辑（2i/3i/ip/pi/2u/up）**：每个分支都关键，优先修复 overlap 最低的分支。union 模式可通过引入额外 OR 分支扩大覆盖而不破坏正确性。
2. **含有效 1p/2p 分支时**：必须做邻域搜索（`incoming_edge_intersection` + `neighborhood_candidates`）。
3. **否定模式**：先修非否定部分最大化正覆盖，否定部分只用于收窄范围（减少 FP）。

---

## 工具详解

### 假设生成/修改

**`GenerateHypothesisLLMTool`** (`generate_hypothesis_llm`) — 三种模式：
- `mode='raw'`：直接传入任意格式的 raw action string（1p/2p/2i/3i/2u/pi/ip/up 等）
- `mode='conditions'`：从 `conditions_json` 构建；每个 dict 含 `entity_id`、`relation_id`，可选 `op='u'`（union）
- `mode='modify'`：传入 `history_raw`（历史假设）+ `modification_instruction`（自然语言修改指令），由 LLM 改写

**`FormatConversionTool`** (`format_conversion`) — 接受 `answer_nl`（逗号分隔实体名）+ `conditions_json`（JSON 字符串），返回 `source_text`

**`GenerateHypothesisTool`** (`generate_hypothesis`) — 接受 `source_text`，调用 CtrlHGen 模型，返回 `raw_output` + `query_nl`

### 条件发现（从 KG 中提出新条件）

**`IncomingEdgeIntersectionTool`** (`incoming_edge_intersection`)
- 输入：`answer_entity_ids`（逗号分隔 ID）、`split`、`top_k`
- 输出：
  - `flat_candidates`：1-hop `(head_entity, relation)` 对，按 Jaccard 排序；每项代表 `p(relation, e(head))`
  - `two_hop_candidates`：2-hop 路径 `(hop1_rel, hop2_rel, anchor_entity)`，代表 `p(hop1, p(hop2, e(anchor)))`；用于 2p/ip/pi 模式
  - `hints`：按公共入边头实体分组的详细信息

**`IntersectionCandidatesTool`** (`intersection_candidates`)
- 输入：`flat_candidates_json`（直接传 `incoming_edge_intersection` 返回的 JSON 字符串，**不要 json.loads**）、`observation_ids`、`mode`、`top_k`
- `mode='2i'`：枚举 top-20 中的 C(20,2)=190 对，执行 2i 查询，按 Jaccard 排序
- `mode='3i'`：枚举 top-10 中的 C(10,3)=120 三元组，执行 3i 查询
- `mode='2u'`：枚举 top-20 中的 C(20,2)=190 对，执行 2u（union）查询

**`NeighborhoodCandidatesTool`** (`neighborhood_candidates`)
- 输入：`entity_ids`（观测实体或 FN 实体）、`fp_ids`（FP 实体，用于惩罚）、`split`、`top_k`
- 评分：`score = obs_coverage - fp_coverage`
- 输出：
  - `relation_candidates`：1-hop 关系候选
  - `entity_candidates`：1-hop 锚实体候选
  - `two_hop_candidates`：2-hop 路径 `(rel1, rel2, anchor)`，结构为 `p(rel1, p(rel2, e(anchor)))`

### 验证与诊断

**`GraphValidationTool`** (`graph_validation`) — 执行完整查询并分解顶层 i/u 子查询，每个子查询返回 `answer_count`、`overlap_count`、`relation_to_label`（exact_match/partial_overlap/disjoint 等）

**`ValidateHypothesisTool`** (`validate_hypothesis`) — 验证假设是否满足条件约束，返回 `valid`、`scores`（validity/specific/smatch）

**`ExecuteAndDiagnoseTool`** (`execute_and_diagnose`) — 返回 TP/FP/FN + Precision/Recall/F1 + `diagnosis`

**`MetricTool`** (`compute_metrics`) — 返回 Jaccard/Dice/Overlap

**`QueryTranslationTool`** (`query_translation`) — raw token list → 自然语言描述

---

## single.py — 单轮有条件生成

```
输入: case["answers_nl"] + case["followup_question"]
  │
  ▼
[CodeAgent: FormatConversionTool]
  LLM 解析 followup_question → conditions_json
  (valid types: unconditional/pattern/relation/entity/relationnumber/entitynumber)
  调用 format_conversion(answer_nl, conditions_json) → {"source_text": "..."}
  │
  ▼ (直接调用)
generate_hypothesis_tool(adapter, source_text)
  → raw_output (unshifted action string) + query_nl
  │
  ▼ (直接调用)
compute_metrics(raw_output, case["answers"], kg.graph_samplers)
  searching_split="test"
  → jaccard / dice / overlap + pred_answers
  │
  ▼ (直接调用)
execute_and_diagnose_tool(raw_output, case["answers"], graph_samplers["train"])
  → TP/FP/FN + precision/recall/f1 + diagnosis

输出: 单次假设结果及诊断
```

---

## loop.py — 多轮反馈优化生成

```
输入: case + max_rounds + jaccard_threshold(默认0.8)
  │
  ▼ ══════════════ 循环 (最多 max_rounds 轮) ══════════════
  │
  ├─ [Step 1] parse_conditions_from_question(llm_model, current_followup)
  │           LLM 将自然语言条件解析为 [{type, value}, ...] 列表
  │
  ├─ [Step 2] format_conversion_tool(adapter, answer_nl, conditions)
  │           → source_text (shifted token 序列)
  │
  ├─ [Step 3] generate_hypothesis_tool(adapter, source_text)
  │           → raw_output + hypothesis_nl
  │
  ├─ [Step 4] compute_metrics → jaccard
  │           jaccard >= threshold 或达到 max_rounds → 停止
  │
  └─ [Step 5] 分析阶段 ─ CodeAgent
       工具: GraphValidationTool, IncomingEdgeIntersectionTool,
             IntersectionCandidatesTool, GenerateHypothesisLLMTool
       │
       ├─ graph_validation(tokens, label_answers, split='test')
       │   → 各子查询的 overlap_count；找 overlap 最高的子查询作为最佳构建块
       │   → 从子查询 token 中提取 entity_id / relation_id（正数）
       │
       ├─ incoming_edge_intersection(answer_entity_ids, split='test', top_k=10)
       │   → flat_candidates（1-hop）+ two_hop_candidates（2-hop，用于 2p/ip/pi）
       │
       ├─ intersection_candidates(flat_candidates_json, obs_ids, mode='2i'/'3i'/'2u')
       │   → 按 Jaccard 排序的组合候选
       │   注意：flat_candidates_json 直接传 incoming_edge_intersection 返回值，不要 json.loads
       │
       └─ 返回 3 个 candidates：
            Candidate 1 (keep):     保持原条件不变，hypothesis_raw=null
            Candidate 2 (update):   基于子查询分析更新条件，hypothesis_raw=null
            Candidate 3 (generate): 调用 generate_hypothesis_llm 直接生成，hypothesis_raw=<raw>
       │
       对每个 candidate 评估：
         若含 hypothesis_raw → 直接用
         否则 → parse_conditions + format_conversion + generate_hypothesis
         compute_metrics → 选 jaccard 最高的 candidate
       │
       current_followup = best_candidate.new_condition → 进入下一轮
  │
  ▼
输出: history 列表（每轮含条件/假设/指标）；打印最优轮次结果
```

### 分析 Agent 的关键约束
- 关系 ID 必须为**正整数**，只能用 `Available relations` 表中的 ID（工具内部自动取负）
- 新假设最多 3 个关系、3 个实体
- 当前模式含 2p/ip/pi 时，**必须**检查 `two_hop_candidates`

---

## uncondition.py — 无条件生成 + 条件建议

```
输入: case["answers_nl"] + case["answers"]
  │
  ▼
[Step 1] format_conversion_tool(conditions=[])  ← 无条件
         generate_hypothesis_tool → raw_output + hypothesis_nl
         compute_metrics → jaccard / dice / overlap
  │
  ▼
[Step 2] CodeAgent 分析（工具预算各 1 次）
         graph_validation(tokens, label_answers, split='train')
           → 各子查询 overlap_count；推断 entitynumber / relationnumber
         incoming_edge_intersection(answer_entity_ids, split='train', top_k=10)
           → flat_candidates（1-hop）+ two_hop_candidates（2-hop）
  │
  ▼
[Step 3] 返回 3 类条件方案（final_answer JSON）
         structural: {entitynumber, relationnumber, pattern} 中至少一项
         semantic:   {relation, entity} 中至少一项（来自 flat_candidates top-1）
         hybrid:     structural + semantic 各至少一项
  │
  ▼
[Step 4] 评估 4 个 candidate（unconditional / structural / semantic / hybrid）
         各自 format_conversion + generate_hypothesis + compute_metrics
         取 jaccard 最高者为 best
  │
  ▼
输出: {unconditional, generated_conditions, candidates, best}
```

---

## multi-turn.py — 多轮对话生成

```
输入: case["answers_nl"] + case["answers"] + case["turns"]（每轮含 followup_question）
  │
  ▼ ══════════════ 遍历每个 turn ══════════════
  │
  ├─ _generate_conditions_from_history(llm_model, history, followup)
  │   LLM 结合历史轮次上下文，将当前 followup 解析为 [{type, value}, ...] 列表
  │
  ├─ format_conversion_tool + generate_hypothesis_tool + compute_metrics
  │   → raw_output + jaccard
  │
  └─ （可选，--analysis 模式）若 jaccard < threshold：
       调用 run_loop(max_rounds=2) 进一步优化
       取 loop 最优结果作为本轮 round_best
  │
  ▼
输出: history 列表（每轮含 condition / parsed_conditions / round_best）

运行:
  python -m akgr.agent.multi-turn --mode case --dataname BioKG [--analysis]
  python -m akgr.agent.multi-turn --mode run  --dataname BioKG --limit 200 [--analysis]
```

---

## judge.py — 单轮结果评估

评估 `singleturn_<model>.jsonl` 日志，输出 Baseline（Round 1）与 Best 两组指标。

**指标**
- Retrieval: Jaccard / Dice / Overlap（mean ± std）
- Condition Accuracy（规则匹配）：
  - per type（relation / entity / relationnumber / entitynumber / pattern）
  - per-hypothesis avg / joint

**条件匹配规则**
| 类型 | 匹配方式 |
|------|----------|
| relation | raw 中含对应负整数 ID（优先），或 NL 中含 `p(name,` |
| entity | raw 中含对应正整数 ID（优先），或 NL 中含 `e(name)` |
| relationnumber | raw 中负整数个数 == value |
| entitynumber | raw 中正整数个数 == value |
| pattern | raw token 化后 == value |

```bash
python -m akgr.agent.judge --dataname PharmKG8k --modelname DeepSeek-V4-Flash
```

---

## judge_multi.py — 多轮结果评估（LLM-based）

评估 `multiturn_<model>.jsonl` 日志，用 LLM 判断每轮假设是否满足当前条件（结合历史上下文）。

**流程**
1. 加载 KG id→name 映射，解析假设的 relations / entities / pattern
2. 对每轮调用 LLM judge（带缓存，避免重复调用）
3. 输出 per-turn Jaccard/Dice/Overlap + Condition Accuracy（avg / joint）

```bash
python akgr/agent/judge_multi.py --dataname PharmKG8k --modelname DeepSeek-V4-Flash [--analysis]
```

---

## 数据流总览

```
KG 原始数据
  └─ load_kg() → KG 对象（ent_id2name, rel_id2name, graph_samplers）
       └─ CtrlHGenAdapter（加载 checkpoint）
            │
            ├─ format_conversion_tool ← 观测实体名 + 条件
            │       ↓ source_text（shifted）
            ├─ generate_hypothesis_tool
            │       ↓ raw_output（unshifted action string）
            ├─ compute_metrics / execute_and_diagnose
            │       ↓ jaccard / TP/FP/FN / diagnosis
            └─ 分析工具（graph_validation / incoming_edge_intersection /
                         intersection_candidates / neighborhood_candidates）
                    ↓ 新条件 → 下一轮
```
