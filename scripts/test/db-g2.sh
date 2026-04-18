# CUDA_VISIBLE_DEVICES=1 python -m akgr.abduction_model.main \
#     --modelname='GPT2_6_act_nt' --condition='relationnumber'\
#     --data_root='./sampled_data/' -d='DBpedia50' --scale='full' -a=32  \
#     --checkpoint_root='checkpoints/' -r=100\
#     --result_root='./results/'\
#     --save_frequency 5\
#     --test_proportion=1\
#     --overwrite_batchsize=256\
#     --mode='testing'\
#     --test_top_k=0\
#     --test_count0


result_root=./results/

CUDA_VISIBLE_DEVICES=1 python -m akgr.abduction_model.main_reverse \
    --condition='multi' \
    --multi_conditions='pattern,entity,relationnumber,entitynumber,relation' \
    --random_multi \
    --modelname GPT2_6_act_nt \
    --data_root ./data/ \
    -d DBpedia50 \
    --scale full \
    -a 32 \
    -r 430\
    --tuning\
    --checkpoint_root  ${result_root} \
    --result_root ${result_root} \
    --save_frequency 5 \
    --test_proportion=1\
    --overwrite_batchsize=1\
    --mode='testing'\
    --test_top_k=0\
    --test_count0\
    --vs
        # --reverse_edges_flag\
    # --tuning\
    # --multi_conditions='entitynumber,pattern,relation' \