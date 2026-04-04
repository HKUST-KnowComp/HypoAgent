CUDA_VISIBLE_DEVICES=7 python -m akgr.abduction_model.main \
    --modelname='GPT2_6_act_nt'\
    --condition='relationnumber' \
    --data_root='./sampled_data/' -d='FB15k-237' --scale='full' -a=32 \
    --checkpoint_root='checkpoints/' -r 90\
    --result_root='./results/'\
    --save_frequency 10\
    --mode='training'