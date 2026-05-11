CUDA_VISIBLE_DEVICES=0 python -m akgr.agent.single
python -m akgr.agent.test
CUDA_VISIBLE_DEVICES=2 python -m akgr.agent.loop
CUDA_VISIBLE_DEVICES=1 python -m akgr.agent.uncondition --mode run
CUDA_VISIBLE_DEVICES=1 python -m akgr.agent.multi-turn --mode run --analysis

python akgr/agent/judge.py --dataname PharmKG8k --modelname DeepSeek-V4-Flash
python akgr/agent/judge.py --dataname PharmKG8k --modelname Qwen3-235B-A22B-Instruct-2507

python -m akgr.agent.judge_multi --dataname PharmKG8k --modelname DeepSeek-V4-Flash --analysis
python -m akgr.agent.judge_multi --dataname PharmKG8k --modelname Qwen3-235B-A22B-Instruct-2507