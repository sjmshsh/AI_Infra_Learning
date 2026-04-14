
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '62115'

def example(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 每个进程生成一个[5, 2]的张量，值为rank*10 + 列索引（便于观察）
    cols_per_rank = 2
    tensor = torch.zeros(5, cols_per_rank, device=rank, dtype=torch.float)
    for j in range(cols_per_rank):
        tensor[:, j] = rank * 10 + j
    print(f"Rank {rank} local tensor shape {tensor.shape}:\n{tensor.cpu().numpy()}")

    # 准备接收列表：每个元素形状与本地张量相同
    gather_list = [torch.empty_like(tensor) for _ in range(world_size)]

    # 执行all_gather（收集到列表）
    dist.all_gather(gather_list, tensor)

    # 沿列维度（dim=1）拼接所有张量
    gathered_tensor = torch.cat(gather_list, dim=1)  # 形状[5, world_size * cols_per_rank] = [5, 8]
    print(f"Rank {rank} after column-wise all_gather, shape {gathered_tensor.shape}:\n{gathered_tensor.cpu().numpy()}")

def main():
    world_size = 4
    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

