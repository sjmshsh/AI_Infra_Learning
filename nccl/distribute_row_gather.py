import os
import torch
import torch.distributed as dist

def example(rank, world_size):
    try:
        # 【关键改动】
        # 在真正的分布式环境中，通常不传参，直接读取环境变量
        # 环境变量由 torchrun 自动注入：RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
        dist.init_process_group(backend="nccl")
        
        # 设置当前进程使用的 GPU
        # 在分布式环境中，通常一个进程对应一张卡
        torch.cuda.set_device(rank)

        # --- 业务逻辑保持不变 ---
        
        # 本地张量：形状[1, 5]，数值为rank
        tensor = torch.ones(1, 5, device=rank) * rank

        # 准备接收列表
        if rank == 0:
            gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
        else:
            gather_list = None

        # 执行 gather 操作
        dist.gather(tensor, gather_list=gather_list, dst=0)

        # 打印结果
        if rank == 0:
            gathered_tensor = torch.cat(gather_list, dim=0)
            print(f"\n[Rank 0] 成功收集数据！")
            print(f"[Rank 0] 最终张量形状: {gathered_tensor.shape}")
            print(f"[Rank 0] 内容:\n{gathered_tensor}")
        else:
            print(f"[Rank {rank}] 数据已发送。")

    finally:
        # 销毁进程组，释放资源
        dist.destroy_process_group()

if __name__ == "__main__":
    # 【关键改动】
    # 不再使用 mp.spawn
    # 这里的 rank 和 world_size 由 torchrun 启动时自动分配
    
    # 获取当前进程的 rank 和 world_size (由环境变量提供)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    print(f"进程启动: Rank {rank}, World Size {world_size}")
    example(rank, world_size)