import torch

x = torch.tensor([1, 2, 3])
print(f"原始形状: {x.shape}")

y = x.unsqueeze(0)
print(f"形状: {y.shape}")      # 输出: torch.Size([1, 3])
print(f"内容:\n{y}")

