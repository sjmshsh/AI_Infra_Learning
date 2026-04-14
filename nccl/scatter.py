
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

    # 接收缓冲区：每个进程得到一行，形状[5]
    recv_tensor = torch.empty(5, device=rank, dtype=torch.float)

    if rank == 0:
        # 源进程：创建一个形状[world_size, 5]的大张量，每一行填充行号
        data_to_scatter = torch.arange(world_size, device=0).float().unsqueeze(1).repeat(1, 5)
        print(f"Rank {rank}: Original data to scatter:\n{data_to_scatter.cpu().numpy()}")

        # 将大张量拆分为列表，每个元素是形状[5]的张量（对应每一行）
        scatter_list = [data_to_scatter[i] for i in range(world_size)]  # 每个元素形状[5]
        print(f"Rank {rank}: Scatter list shapes: {[t.shape for t in scatter_list]}")
    else:
        scatter_list = None

    # 执行scatter操作
    dist.scatter(recv_tensor, scatter_list=scatter_list, src=0)

    print(f"Rank {rank} received tensor of shape {recv_tensor.shape}:\n{recv_tensor.cpu().numpy()}")

def main():
    world_size = 4
    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
