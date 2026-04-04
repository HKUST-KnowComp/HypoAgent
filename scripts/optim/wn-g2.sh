CUDA_VISIBLE_DEVICES=1 python -m akgr.abduction_model.main \
    --modelname='GPT2_6_act_nt'\
    --data_root='./sampled_data/' -d WN18RR --scale='full' -a=32  \
    --checkpoint_root='checkpoints/' -r=50 \
    --condition='pattern' \
    --result_root='./results/'\
    --save_frequency 10\
    --mode='optimizing' \
    --overwrite_batchsize=32\
    --rl_lr=1e-5\
    --rl_smatch_factor=0\
    --rl_factor='[1.0, 1.0, 0.5, 0.0]'\
    --rl_init_kl_coef=0.05\
    --rl_cliprange=0.2\
    --rl_epochs=1\
 