# HypoAgent

This is the official code repository for **HypoAgent: An Agentic Framework for Interactive Abductive Hypothesis Generation over Knowledge Graphs**.

---

## 1. Environment Setup

```bash
conda create -n hypoagent python=3.10
conda activate hypoagent
pip install -r requirements.txt
```

> **Note:** PyTorch is installed with CUDA 11.8 support. If you use a different CUDA version, install the matching PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/).

### LLM API Configuration

Copy the API key template and fill in your own credentials:

```bash
cp akgr/configs/api_keys.yml.example akgr/configs/api_keys.yml
# Edit akgr/configs/api_keys.yml with your API keys
```

The agent modules use [smolagents](https://github.com/huggingface/smolagents) `OpenAIServerModel` to call LLMs. You can configure any OpenAI-compatible API provider in `akgr/configs/api_keys.yml`.

---

## 2. Data Sampling

Sample training data from the knowledge graph:

```bash
python -m akgr.sampling.sample_parallel -s="full" -a=32 -p=16
```

This generates sampled data under `./sampled_data/` for all configured datasets.

---

## 3. Training the Hypothesis Generation Model

### 3.1 Unconditional Training

Train a lightweight hypothesis generation model without any condition control:

```bash
CUDA_VISIBLE_DEVICES=0 python -m akgr.abduction_model.main \
    --condition='unconditional'
    --modelname='GPT2_6_act_nt' \
    --data_root='./sampled_data/' -d='BioKG' --scale='full' -a=32 \
    --checkpoint_root='checkpoints/' \
    --result_root='./results/' \
    --save_frequency 5 \
    --mode='training'
```

### 3.2 Multi-Condition Training

Train with multiple types of control signals (pattern, entity, relation, etc.) simultaneously:

```bash
CUDA_VISIBLE_DEVICES=0 python -m akgr.abduction_model.main \
    --condition='multi' \
    --multi_conditions='pattern,entity,entitynumber,relation,relationnumber' \
    --random_multi \
    --seed 42 \
    --modelname GPT2_6_act_nt \
    --accelerate \
    --data_root ./sampled_data/ \
    -d PharmKG8k \
    --scale full \
    -a 32 \
    --checkpoint_root ./results/ \
    --result_root ./results/multi-pharmkg8k/ \
    --save_frequency 5 \
    --mode training
```

---

## 4. Running the Agent

All agent scripts require a trained multi-condition checkpoint and an LLM API configured in `akgr/configs/api_keys.yml`.

### 4.1 Single-Turn Agent (`loop.py`)

Single-turn hypothesis generation with iterative self-refinement. The agent generates an initial hypothesis, analyzes its quality via hypothesis fragment diagnosis and neighborhood search, then refines it over multiple rounds.

```bash
# Run on batch data
CUDA_VISIBLE_DEVICES=0 python -m akgr.agent.loop \
    --mode run \
    --dataname BioKG \
    --checkpoint checkpoints/BioKG-full-32-multi.pth \
    --data_root ./data/ \
    --max_rounds 3 \
    --jaccard_threshold 0.95

# Run on a single demo case
CUDA_VISIBLE_DEVICES=0 python -m akgr.agent.loop --mode case
```

### 4.2 Multi-Turn Agent (`multi-turn.py`)

Multi-turn dialogue-based hypothesis generation. The user provides a sequence of follow-up questions across multiple turns, and the agent progressively refines the hypothesis.

```bash
# Run on batch data
CUDA_VISIBLE_DEVICES=0 python -m akgr.agent.multi-turn \
    --mode run \
    --dataname BioKG \
    --checkpoint checkpoints/BioKG-full-32-multi.pth \
    --data_root ./data/ \
    --analysis

# Run on a single demo case
CUDA_VISIBLE_DEVICES=0 python -m akgr.agent.multi-turn --mode case --analysis
```

The `--analysis` flag enables the RCA-Agent analysis for quality refinement when Jaccard is below threshold.

### 4.3 Unconditional Agent (`uncondition.py`)

Starts from an unconditional hypothesis (no user conditions), then uses the agent to automatically analyze the KG structure and derive structural/semantic/hybrid conditions for refinement.

```bash
# Run on batch data
CUDA_VISIBLE_DEVICES=0 python -m akgr.agent.uncondition \
    --mode run \
    --dataname DBpedia50 \
    --checkpoint checkpoints/DBpedia50-full-32-multi.pth \
    --data_root ./data/

# Run on a single demo case
CUDA_VISIBLE_DEVICES=0 python -m akgr.agent.uncondition --mode case
```

### 4.4 Evaluate Agent Results

```bash
# Evaluate single-turn results
python -m akgr.agent.judge --dataname BioKG --modelname DeepSeek-V4-Flash

# Evaluate multi-turn results
python -m akgr.agent.judge_multi --dataname BioKG --modelname DeepSeek-V4-Flash --analysis

# Evaluate unconditional results
python -m akgr.agent.judge_uncondition --dataname DBpedia50 --modelname gpt-5.4-mini --analysis
```


