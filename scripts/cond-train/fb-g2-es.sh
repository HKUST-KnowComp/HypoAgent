CUDA_VISIBLE_DEVICES=2 python -m akgr.abduction_model.main \
    --modelname='GPT2_6_act_nt'\
    --condition='entity' \
    --data_root='./sampled_data/' -d='FB15k-237' --scale='full' -a=32 \
    --checkpoint_root='checkpoints/' -r 90\
    --result_root='./results/'\
    --save_frequency 5\
    --mode='training'