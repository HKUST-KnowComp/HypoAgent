CUDA_VISIBLE_DEVICES=0 python -m akgr.abduction_model.main \
    --condition='entity' \
    --modelname='GPT2_6_act_nt'\
    --data_root='./sampled_data/' -d='WN18RR' --scale='full' -a=32 -r 50 \
    --checkpoint_root='checkpoints/'\
    --result_root='./results/'\
    --save_frequency 5\
    --mode='training'