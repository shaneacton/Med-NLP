import torch

device = torch.device("cuda:0")

print("using device:", device, "cuda avail:", torch.cuda.is_available())





