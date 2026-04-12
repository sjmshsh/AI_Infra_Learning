import torch

# 创建一些示例张量
tensor_a = torch.tensor([1.0, 2.0])
tensor_b = torch.tensor([3.0, 4.0])
tensor_c = torch.tensor([5.0, 6.0])
tensor_d = torch.tensor([7.0, 8.0])
tensor_e = torch.tensor([9.0, 10.0])

# 创建两个张量列表
list_1 = [tensor_b, tensor_c]
list_2 = [tensor_d, tensor_e]

# 构建最终的嵌套数据结构
data = (tensor_a, (list_1, list_2))

print(data)