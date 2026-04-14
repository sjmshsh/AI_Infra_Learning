import os 
import torch
import torch.distributed as dist 
import torch.multiprocessing as mp

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '62115'

def example(rank, world_size):
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # 让当前进程加入群组，并建议与其他进程的通信连
    torch.cuda.set_device(rank) # 这是一个常见的约定。让第 rank 号进程专门使用第 rank 号显卡（例如进程 0 用 GPU 0，进程 1 用 GPU 1），避免冲突。

    # 本地张量：形状[1, 5]，数值为rank
    # device=rank 指定了张量存放在对应的 GPU 上
    tensor = torch.ones(1, 5, device=rank) * rank

    # 只在目标进程（rank 0）创建接收列表
    if rank == 0:
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
    else:
        gather_list = None # 非目标进程不需要接收列表

    # 执行 gather操作，所有进程将tensor发送给rank 0
    dist.gather(tensor, gather_list=gather_list, dst=0)

    # 在rank 0上处理收集到的数据
    if rank == 0:
        # 将列表沿dim=0拼接，得到[world_size, 5]
        gathered_tensor = torch.cat(gather_list, dim=0)
        print(f"Rank {rank} gathered tensor shape: {gathered_tensor.shape}")
        print("Gathered tensor:\n", gathered_tensor)
    else:
        print(f"Rank {rank} has sent its data, no local copy of gathered tensor.")

def main():
    world_size = 4
    mp.spawn(example, args=(world_size, ), nprocs=world_size, join=True) # 启动多进程，复制当前程序4次，分别赋予他们不同的rank（编号0，1，2，3）

if __name__ == "__main__":
    main()
