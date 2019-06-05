import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:1")
print(torch.cuda.current_device())  # 0
