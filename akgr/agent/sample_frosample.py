from akgr.kgdata import load_kg

kg = load_kg(
    dataroot="/home/ycaicr/CtrlHGen/sampled_data",
    dataname="PharmKG8k",
    reverse_edges_flag=False,
)

print("num entities:", len(kg.ent_id2name))
for i, (eid, name) in enumerate(kg.ent_id2name.items()):
    print(eid, name)
    if i >= 49:
        break