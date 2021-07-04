import torch
from torch import nn


class Demo(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(6, 12)
        self.linear2 = nn.Linear(12, 6)
        self.linear3 = nn.Linear(6, 1)

    def forward(self, x: torch.Tensor):
        print("network", self, "got input:", x.size())
        out1 = self.linear1(x)
        print("out1:", out1.size())
        out2 = self.linear2(out1)
        print("out2:", out2.size())
        out3 = self.linear3(out2)
        print("out3:", out3.size())
        return out3


if __name__ == "__main__":
    demo = Demo()

    x = torch.Tensor([1,2,3,4,5,6]).float()
    print("x:", x.size(), x)
    output = demo(x)
