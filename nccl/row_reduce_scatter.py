
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '62115'

def example(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 原始数据：形状[world_size, 5]，第i行的值为rank*10 + i
    data = torch.zeros(world_size, 5, device=rank, dtype=torch.float)
    for i in range(world_size):
        data[i] = rank * 10 + i

    print(f"Rank {rank} original data (shape {data.shape}):\n{data.cpu().numpy()}")

    # 将data按第一维拆分成列表，每个元素保持二维形状[1, 5]（即每个分片是一行并保留维度）
    input_list = [data[i].unsqueeze(0) for i in range(world_size)]  # 列表长度=world_size，每个元素形状[1, 5]

    # 输出张量也设为二维[1, 5]，用于接收规约后属于当前 rank 的分片
    output = torch.empty(1, 5, device=rank, dtype=torch.float)

    # 执行 reduce_scatter：所有进程的input_list中对应本进程的分片会被规约（求和）后存入output
    dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)

    print(f"Rank {rank} after reduce_scatter (sum), output shape {output.shape}:\n{output.cpu().numpy()}")

def main():
    world_size = 4
    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()