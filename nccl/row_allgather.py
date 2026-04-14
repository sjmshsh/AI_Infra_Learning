import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '62115'

def example(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 本地张量形状[1, 5]，放在当前GPU
    tensor_shard = torch.ones(1, 5, device=rank) * rank

    # 创建接收列表：每个元素形状为[1, 5]
    tensor_gather_list = [torch.empty_like(tensor_shard) for _ in range(world_size)]

    # 执行 all_gather
    dist.all_gather(tensor_gather_list, tensor_shard)

    # 方法1：使用torch.cat沿dim=0拼接，得到 [world_size, 5]
    gathered_tensor = torch.cat(tensor_gather_list, dim=0)  # 形状 (world_size, 5)
    print(f"rank {rank} after cat: {gathered_tensor.shape}")

    # 方法2：如果使用torch.stack，会得到[world_size, 1, 5]
    # stacked = torch.stack(tensor_gather_list, dim=0)  # (world_size, 1, 5)
    # gathered_tensor = stacked.squeeze(1)         # (world_size, 5)

    time.sleep(1)
    # 验证结果：rank 0 打印聚合后的张量
    print(f"Rank {rank} gathered tensor:\n", gathered_tensor)

def main():
    world_size = 4  # 假设 4 个进程
    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

