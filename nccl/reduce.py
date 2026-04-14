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

    # 本地张量：形状[1, 5]，数值为rank
    tensor = torch.ones(1, 5, device=rank) * rank 

    print(f"Rank {rank} before reduce: {tensor.cpu().tolist()}")

    # 执行reduce操作，将所有进程的tensor求和，结果存储到rank 0的tensor中
    # 注意：reduce 后，非目标进程的tensor内容可能不再有效
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)

    # 在 rank 0 上打印规约结果
    if rank == 0:
        print(f"Rank {rank} after reduce (sum): {tensor.cpu().tolist()}")
    else:
        # 非目标进程的tensor内容未定义，但为了演示，打印其当前值
        print(f"Rank {rank} after reduce (tensor content undefined): {tensor.cpu().tolist()}")

def main():
    world_size = 4
    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
    