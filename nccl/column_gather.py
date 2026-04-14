import os 
import torch
import torch.distributed as dist 
import torch.multiprocessing as mp

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '62115'

def example(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # 每个进程生成自己的本地张量：形状[5, 2]，值 = rank*10 + 列索引
    cols_per_rank = 2 
    local_tensor = torch.zeros(5, cols_per_rank, device=rank, dtype=torch.float)
    for col in range(cols_per_rank):
        local_tensor[:, col] = rank * 10 + col  # 第0列全是 rank*10，第1列全是 rank*10+1

    print(f"Rank {rank} local tensor (shape {local_tensor.shape}):\n{local_tensor.cpu().numpy()}")

   # 只在目标进程（rank 0）上准备接收列表
    if rank == 0:
        # 接收列表包含 world_size 个空张量，每个形状与 local_tensor 相同，位于 rank 0 的设备上
        gather_list = [torch.empty_like(local_tensor, device=0) for _ in range(world_size)]
    else:
        gather_list = None  # 非目标进程不需要接收列表

    # 执行 gather 操作：所有进程将local_tensor发送给rank 0
    dist.gather(local_tensor, gather_list=gather_list, dst=0)

    # 在rank 0上处理收集到的数据
    if rank == 0:
        # 沿列维度（dim=1）拼接，得到形状[5, world_size * 2]的大张量
        gathered = torch.cat(gather_list, dim=1)
        print(f"\nRank {0} final gathered tensor (shape {gathered.shape}):\n{gathered.cpu().numpy()}")
    else:
        print(f"Rank {rank} has sent its data, no local copy of gathered tensor.\n")



def main():
    world_size = 4
    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()