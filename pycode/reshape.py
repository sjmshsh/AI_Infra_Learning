import torch

batch_size = 1
seq_len = 6
d_model = 8
num_heads = 2 # 把注意力分成2个头
head_dim = 4 # 每个头分到4个特征

q = torch.arange(1, 49).float().view(batch_size, seq_len, d_model)

print("🔴 原始数据 q 的形状:", q.shape) 

q_reshaped = q.reshape(batch_size, seq_len, num_heads, head_dim).contiguous()

print("\n🟢 reshape 后的形状:", q_reshaped.shape)


