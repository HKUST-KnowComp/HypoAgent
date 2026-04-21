CUDA_VISIBLE_DEVICES=0 python -m akgr.agent.single
python -m akgr.agent.test
CUDA_VISIBLE_DEVICES=0 python -m akgr.agent.loop
CUDA_VISIBLE_DEVICES=0 python -m akgr.agent.uncondition