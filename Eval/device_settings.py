import torch

device = torch.device("cuda:0")
# device = torch.device("cpu")


print("using device:", device, "cuda avail:", torch.cuda.is_available())





