# akgr/agent — 文件说明

## 入口

- **cli_chat.py** — 交互式命令行对话循环。接收用户输入的实体和条件，驱动完整流程：解析 → 构建模型输入 → 生成 → 判断 → 语言化 → 追问。

## 核心流程组件

- **ctrlhgen_adapter.py** — 封装训练好的 CtrlHGen 模型。负责加载 checkpoint、tokenizer 和模型；根据解析后的控制信息构建 source 文本；执行约束生成；将 action string 解码为 query token 和自然语言。

- **llm_parser.py** — `LocalQwenParser`：使用本地 Qwen LLM 将用户自由文本条件解析为结构化 JSON `{conditions: [{type, value}]}`。下游验证失败时也负责修复解析结果。

- **llm_verbalizer.py** — `LocalQwenVerbalizer`：使用本地 Qwen LLM 完成三项任务：(1) 将假设语言化为自然语言，(2) 判断生成的 query 是否满足用户条件，(3) 提出追问澄清问题。

## 数据 / 映射工具

- **kg_mapper.py** — `KGNameMapper`：将实体/关系名称映射为 KG 整数 ID，支持精确匹配、规范化匹配、词集合匹配和模糊匹配四级回退。

- **action_to_nl.py** — 将模型输出的 action string（如 `i -9 5531 -22 1137`）解析为树结构，并将树转换为自然语言描述。

- **getsomesampleFromDB.py** — 将结构化 query token 列表转换为自然语言，通过解析 query 树并将实体/关系 ID 解码为名称实现。同时提供 CLI 工具批量转换 `.jsonl` 文件。

- **source_builder.py** — 根据观测实体 ID 和条件类型/值构建模型 source 字符串（旧版辅助函数，主要逻辑已迁移至 `ctrlhgen_adapter.py`）。

## 会话管理

- **chat_session.py** — `ChatSession` 和 `TurnRecord` 数据类。跨轮次追踪对话历史、最近解析的控制信息及观测/条件记忆。`reset_context()` 用于清空状态以开始新问题。

## 其他 / 开发脚本

- **test.py** — 冒烟测试脚本：使用 `smolagents` `CodeAgent` 和 `EchoTool` 对接 DeepInfra OpenAI 兼容接口。

- **sample_frosample.py** — 从 KG 中采样/查看数据的开发工具。

- **printdata.py** — 打印数据集记录的开发工具。

- **to_see_name.py** — 查看实体/关系名称映射的开发工具。

- **DBpedia_question_example.txt** — DBpedia50 数据集的示例问题，用于手动测试。
