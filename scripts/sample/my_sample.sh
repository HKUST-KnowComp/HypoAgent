export CUDA_VISIBLE_DEVICES=0,1
python -m akgr.sampling.sample_parallel -s="full" -a=32 -p=16

# python -m akgr.sampling.sample_parallel -s="my_sample_Bio" -a=32 -p=16 -r

# python -m akgr.sampling.sample_parallel -s="my_sample_Pharm" -a=32 -p=16

# python -m akgr.sampling.sample_parallel -s="my_sample_Pharm" -a=32 -p=16 -r