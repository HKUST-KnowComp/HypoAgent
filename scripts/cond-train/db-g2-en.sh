CUDA_VISIBLE_DEVICES=0 python -m akgr.abduction_model.main_reverse \
    --condition='multi' \
    --multi_conditions='entitynumber,pattern,relation' \
    --modelname='GPT2_6_act_nt'\
    --data_root='./sampled_data/' -d='DBpedia50' --scale='full' -a=32 \
    --checkpoint_root='./results/'\
    --result_root='./test-results/'\
    --save_frequency 20\
    --mode='training'\