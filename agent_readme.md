# Agent 执行流程说明

## 整体架构

系统包含两种运行模式：
- **单轮模式** (`single.py`)：给定 observation 和条件，生成一次假设并评估
- **多轮循环模式** (`loop.py`)：在单轮基础上，通过反思 Agent 迭代优化条件，直到 Jaccard ≥ 0.7 或达到最大轮数

---

## 涉及的 Agent

| Agent | 类型 | 工具 | 职责 |
|---|---|---|---|
| `format_conversion_agent` | CodeAgent | `FormatConversionTool` | 将自然语言条件解析为模型输入格式 |
| `hypothesis_generate_agent` | CodeAgent | `GenerateHypothesisTool` | 调用 CtrlHGen 模型生成假设 |
| `metric_agent` | CodeAgent | `MetricTool` | 在 KG 上执行假设并计算评估指标 |
| `reflection_agent` | CodeAgent | 无工具（纯推理） | 分析历史轮次不足，提出新条件 |

---

## 单轮执行流程（`run_single_turn`）

```
输入: answer_nl（观测实体名列表）, followup（条件问题）, label_answers（真实答案 ID 列表）

Step 1: format_conversion_agent
  - 输入: followup 问题字符串 + answer_nl 实体列表
  - 内部: 将 followup 解析为 conditions_json（类型+值的数组）
  - 调用工具: FormatConversionTool
    - 将实体名映射为 KG 中的 ID（通过 KGNameMapper）
    - 将条件编码为 source token（如 relation -> 负数 ID，entity -> shifted ID）
    - 拼接成 source_text，例如 "2464 2579 SEP -8 2308"
  - 输出: source_text（模型输入字符串）

Step 2: hypothesis_generate_agent
  - 输入: source_text
  - 调用工具: GenerateHypothesisTool
    - 将 source_text tokenize 后送入 GPT2 模型
    - 使用 generate_with_constraints 采样生成 action string
    - 将 shifted action string 还原为 unshifted 形式
    - 解析 action tree -> query tokens -> 自然语言描述
  - 输出: raw_output（unshifted action string）, query_nl（自然语言假设）

Step 3: metric_agent
  - 输入: raw_output + label_answers（真实答案 ID）
  - 调用工具: MetricTool
    - 将 raw_output 解析为 query word list
    - 在 KG graph_samplers 上执行查询，得到预测答案集合
    - 与真实答案集合计算 Jaccard / Dice / Overlap
  - 输出: jaccard, dice, overlap, pred_answers, label_answers

返回: 本轮的条件、假设、指标等完整记录
```

---

## 多轮循环流程（`run_loop`）

```
输入: case（包含 answers_nl, answers, followup_question）
      max_rounds（最大轮数，默认 5）
      jaccard_threshold（停止阈值，默认 0.7）

初始化:
  - current_followup = case["followup_question"]
  - history = []

FOR round = 1 to max_rounds:

  ┌─────────────────────────────────────────────┐
  │  执行单轮（run_single_turn）                  │
  │  使用 current_followup 作为当前条件           │
  └─────────────────────────────────────────────┘
          │
          ▼
  记录本轮结果到 history（条件、假设、Jaccard 等）

  检查停止条件:
    ├── Jaccard >= jaccard_threshold → 停止（成功）
    └── round == max_rounds         → 停止（达到上限）

  若继续:
  ┌─────────────────────────────────────────────────────────────────┐
  │  reflection_agent（纯推理，无工具）                               │
  │                                                                 │
  │  输入 prompt 包含:                                               │
  │    - 观测实体列表（answer_nl）                                    │
  │    - 真实答案数量                                                 │
  │    - 所有历史轮次摘要（轮次、条件、假设NL、假设raw、Jaccard、答案数）│
  │    - KG 中所有可用 relation 名称                                  │
  │    - KG 中前 50 个 entity 名称（供参考）                          │
  │    - 可用条件类型说明（pattern/relation/entity/数量约束）           │
  │                                                                 │
  │  推理任务:                                                        │
  │    1. 分析历史假设 Jaccard 低的原因                                │
  │       - 预测答案过多（条件太宽松）？                                │
  │       - 预测答案过少（条件太严格）？                                │
  │       - 结构模式不匹配？                                           │
  │       - 关系/实体选择不当？                                        │
  │    2. 提出与之前不同的新条件策略                                    │
  │                                                                 │
  │  输出: JSON {"analysis": "...", "new_condition": "..."}          │
  └─────────────────────────────────────────────────────────────────┘
          │
          ▼
  current_followup = new_condition（进入下一轮）

END FOR

输出最终摘要:
  - 每轮的 Jaccard 和条件
  - 最优轮次（最高 Jaccard）的假设
  - 返回完整 history 列表
```

---

## 历史信息格式（传给 reflection_agent）

每轮历史条目包含：

```
Round N:
  Condition: <当前轮使用的条件问题>
  Hypothesis (NL): <生成假设的自然语言描述>
  Hypothesis (raw): <生成假设的 action string>
  Jaccard: <0.xxxx>
  Predicted answers: <预测答案数量>, Label answers: <真实答案数量>
```

---

## 工具详细说明

### FormatConversionTool
- **输入**: `answer_nl`（逗号分隔实体名）, `conditions_json`（JSON 数组）
- **处理**:
  - 实体名 → KG entity ID（支持模糊匹配）
  - 对每个条件类型分别编码：
    - `entity` → shifted entity ID 字符串
    - `relation` → `-(abs(rel_id)+1)` 字符串
    - `entitynumber` / `relationnumber` → `"N e"` / `"N p"` 格式
    - `pattern` → 规范化 pattern 字符串
  - 拼接：`"<obs_ids> SEP <cond_tokens>"`
- **输出**: `source_text`, `observation_entity_ids`, `conditions`

### GenerateHypothesisTool
- **输入**: `source_text`
- **处理**:
  - Tokenize → GPT2 自回归生成 → decode → unshift
  - 解析 action string → action tree → query tokens → NL
  - 统计 entity/relation 数量
- **输出**: `raw_output`, `query_nl`, `entitynumber`, `relationnumber`

### MetricTool
- **输入**: `raw_output`（unshifted action string）, `label_answers`（逗号分隔 ID）
- **处理**:
  - 解析 action string → query word list
  - 在 KG train split 上执行图查询，得到预测答案集合
  - 计算集合交并比
- **输出**: `jaccard`, `dice`, `overlap`, `pred_answers`, `label_answers`
