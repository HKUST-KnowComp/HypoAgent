result_root=./results/pharm/False

CUDA_VISIBLE_DEVICES=1 accelerate launch \
    --main_process_port 41012 \
    --num_processes 2 \
    -m akgr.abduction_model.main_reverse \
    --modelname GPT2_6_act_nt \
    --accelerate \
    --data_root ./sampled_data/ \
    -d PharmKG8k \
    --scale my_sample_Pharm \
    -a 32 \
    --checkpoint_root  ${result_root} \
    --result_root ${result_root} \
    --save_frequency 10 \
    --mode training
    # --reverse_edges_flag