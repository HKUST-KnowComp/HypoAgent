# CUDA_VISIBLE_DEVICES=3 python -m akgr.abduction_model.main \

#     --modelname='GPT2_6_act_nt' \
#     --data_root='./sampled_data/' -d='DBpedia50' --scale='full' -a=32 -r 380 \
#     --checkpoint_root='checkpoints/' \
#     --result_root='./results/' \
#     --save_frequency 20 \
#     --mode='training'

result_root=./results/

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --main_process_port 41013 \
    --num_processes 1 \
    -m akgr.abduction_model.main_reverse \
    --condition='multi' \
    --multi_condit  ions='pattern,entity,entitynumber,relation,relationnumber' \
    --random_multi \
    --seed 42\
    --modelname GPT2_6_act_nt \
    --accelerate \
    --data_root ./sampled_data/ \
    -d PharmKG8k \
    --scale full \
    -a 32 \
    -r 100\
    --checkpoint_root  ${result_root} \
    --result_root ${result_root}/multi-pharmkg8k/ \
    --save_frequency 5 \
    --mode training

CUDA_VISIBLE_DEVICES=3 python -m akgr.abduction_model.main_reverse \
    --modelname='GPT2_6_act_nt' \
    --data_root='./data/' -d='PharmKG8k' --scale='full' -a=32 -r 380 \
    --condition='multi' \
    --checkpoint_root='checkpoints/' \
    --result_root='./results/' \
    --save_frequency 20 \
    --test_proportion=1\
    --overwrite_batchsize=10\
    --mode='testing'\
    --test_top_k=0\
    --test_count0\
    --vs