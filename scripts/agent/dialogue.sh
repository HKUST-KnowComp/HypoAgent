CUDA_VISIBLE_DEVICES=1 \
python -m akgr.agent.cli_chat\
  --checkpoint_path /home/ycaicr/CtrlHGen/results/GPT2_6_act_nt/DBpedia50-full-32-380-unconditional.pth \
  --data_root /home/ycaicr/CtrlHGen/sampled_data \
  --dataname DBpedia50\
  --constrained \
  --temperature 1.0 \
  --fallback_query '"(", "p", "(", -659, ")", "(", "i", "(", "n", "(", "p", "(", -658, ")", "(", "e", "(", 8271, ")", ")", ")", ")", "(", "p", "(", -587, ")", "(", "e", "(", 269, ")", ")", ")", ")", ")"'
  # --vs

#You can use the following entities to test the agent:
# Leandro_Rinaudo, Abderrazzak_Jadid, Luca_Simeoni, Devis_Nossa, César_Cervo_Luca




