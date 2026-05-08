import torch

def softmax(x):
    x_exp = torch.exp(x - torch.max(x, dim=-1, keepdim=True).values)
    return x_exp / torch.sum(x_exp, dim=-1, keepdim=True)

if __name__ == '__main__':
    input_tensor = torch.randn(10, 5)
    ouput_custom = softmax(input_tensor)

