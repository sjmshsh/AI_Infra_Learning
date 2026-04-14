
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '62115'

def example(rank, world_size):
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 本地张量：形状[1, 5]，数值为 rank
    tensor = torch.ones(1, 5, device=rank) * rank

    print(f"Rank {rank} before all_reduce: {tensor.cpu().tolist()}")

    # 执行all_reduce操作，求和并广播到所有进程
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"Rank {rank} after all_reduce (sum): {tensor.cpu().tolist()}")

def main():
    world_size = 4
    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

