import torch
import triton
import triton.language as tl

@triton.jit
def _fused_linear_kernel_fwd (
    x_ptr, # 输入数据矩阵首元素指针
    w_ptr, # 权重矩阵首元素指针
    z_ptr, # 输出结果地址
    M, N, K, # matrix dimensions
    BLOCK_SIZE_M: tl.constexpr = 128, # 块大小
    BLOCK_SIZE_N: tl.constexpr = 128,
    BLOCK_SIZE_K: tl.constexpr = 64,
):
    # 对于每个triton block的二维坐标
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # 一个triton block的处理范围（在M，N轴上）
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None] # 当前block负责的行号（列向量）
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]  # 形状为 (1, BLOCK_SIZE_N)。

    z = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # 在K轴上进行一个分约规约
    for k in range(0, K, BLOCK_SIZE_K):
        x_k = tl.arange(0, BLOCK_SIZE_K)[None, :] + k # K维当前处理的列号

        x = tl.load(
            x_ptr + offs_m * K + x_k, # [BM, BK]个地址，一次性全部读取
            mask=(offs_m < M) & (x_k < K), # [BM, BK]的布尔掩码
            other=0.0 # 越界位置填0
        )
        x = x.to(tl.float16) # 把数据转换成float16精度

        w_k = tl.arange(0, BLOCK_SIZE_K)[:, None] + k
        # tl.load加载的是(w_k,offs_n)
        w = tl.load(w_ptr + w_k * N + offs_n, mask=(w_k < K) & (offs_n < N), other=0.0)
        w = w.to(tl.float16)
        # z += x@w
        z = tl.dot(x, w, acc=z)        
    # 一个triton block计算的结果大小是block_m×block_n
    z_offset = offs_m * N + offs_n
    z_mask = (offs_m < M) & (offs_n < N)

    tl.store(z_ptr + z_offset, z, mask=z_mask)

