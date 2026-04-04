accelerate launch --num_processes=4 -m akgr.abduction_model.main \
    --modelname='GPT2_6_act_nt'\
    --data_root='./sampled_data/' -d DBpedia50 --scale='full' -a=32  \
    --checkpoint_root='checkpoints/' -r=100 \
    --condition='pattern' \
    --result_root='./results/'\
    --save_frequency 10\
    --mode='optimizing' \
    --overwrite_batchsize=16\
    --rl_lr=1e-5\
    --rl_smatch_factor=0\
    --rl_factor='[0.75, 0.5, 0.0, 1.25]'\
    --rl_init_kl_coef=0.1\
    --rl_cliprange=0.2\
    --rl_epochs=3\
 
