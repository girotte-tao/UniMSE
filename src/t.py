import os
import torch
import pickle
from pathlib import Path
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained('./t5-base')
project_dir = Path(__file__).resolve().parent.parent
DataPath = os.path.join(project_dir, "datasets/IEMOCAP/train.pkl")
with open(DataPath, "rb") as f:
    data = pickle.load(f)
outputs_seq = []
for i in range(100):
    sample = data[i]
    score = sample[1][-1]
    print(score)
    break



