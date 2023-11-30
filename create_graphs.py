from constants import HF_DS_PATH
from datasets import load_dataset
import matplotlib.pyplot as plt

#Load dataset from huggingface
dataset = load_dataset(HF_DS_PATH)

d = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
for split in ["train", "test"]:
    for entry in dataset[split]:
        d[entry["label"]] += 1
    
updated_counts = {"normal": d[4], "murmur": d[3], "exstrastole": d[0], "artifact": d[2], "extrahls": d[1]}

plt.bar(range(len(updated_counts)), list(updated_counts.values()), align='center')
plt.xticks(range(len(updated_counts)), list(updated_counts.keys()))
plt.xlabel("Heartbeat classification")
plt.ylabel("Count")
plt.savefig("classdist_barplot.png")
