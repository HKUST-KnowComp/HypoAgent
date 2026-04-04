accelerate launch --num_processes=4 -m akgr.abduction_model.main \
    --modelname='GPT2_6_act_nt'\
    --data_root='./sampled_data/' -d DBpedia50 --scale='full' -a=32  \
    --checkpoint_root='checkpoints/' -r=0 \
    --condition='pattern' \
    --result_root='./results/'\
    --save_frequency 10\
    --mode='testing'\
    --overwrite_batchsize=128\
    --rl_lr=0.5e-5\
    --rl_smatch_factor=0\
    --rl_factor='[1.0, 0.5, 0.0, 1.0]'\
    --rl_init_kl_coef=0.2\
    --rl_cliprange=0.2\
    --rl_resume_epoch=2\
 