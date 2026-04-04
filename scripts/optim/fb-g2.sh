
CUDA_VISIBLE_DEVICES=0 python -m akgr.abduction_model.main \
    --modelname='GPT2_6_act_nt'\
    --data_root='./sampled_data/' -d 'FB15k-237' --scale='full' -a=32  \
    --checkpoint_root='checkpoints/' -r=0 \
    --condition='pattern' \
    --result_root='./results/'\
    --save_frequency 10\
    --mode='optimizing' \
    --overwrite_batchsize=16\
    --rl_lr=1e-5\
    --rl_smatch_factor=0\
    --rl_factor='[1.0, 1.0, 0.5, 0.0]'\
    --rl_init_kl_coef=0.1\
    --rl_cliprange=0.2\
    --rl_epochs=8\
    --rl_resume_epoch=1\
 
