from akgr.dataloader import new_create_dataset
import pandas as pd

pattern_filtered = pd.read_csv("akgr/metadata/pattern_filtered.csv", index_col="id")

dataset_dict, nentity, nrelation = new_create_dataset(
    dataname="PharmKG8k",
    scale="my_sample_Pharm",   # 你实际用的 scale 按你的来
    answer_size=32,
    pattern_filtered=pattern_filtered,
    data_root="./sampled_data",
    splits=["train"],
    is_act=True,
)

print(dataset_dict["train"][0])