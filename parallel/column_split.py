import numpy as np 

# 1. 定义输入矩阵（M, N）和（N, K）
M, N, K = 3, 4, 6
A = np.random.randint(0, 10, size=(M, N)) # 随机整数矩阵 [0, 10)
B = np.random.randint(0, 10, size=(N, K))
print("A:\n", A, "\nshape:", A.shape)
print("\nB:\n", B, "\nshape:", B.shape)

# 2. 对B进行切分（均分）
num_splits = 3 
B_splits = np.split(B, num_splits, axis=1) # 沿列切分
print("\nB 分块结果:")
for i, B_i in enumerate(B_splits):
    print(f"B_{i}:\n", B_i, "\nshape:", B_i.shape)

# 3. 模拟并行计算：每个进程计算 A @ B_i
local_results = [A @ B_i for B_i in B_splits] # 在实际并行系统中，这些计算可以在不同处理器上同时进行
print("\n局部乘积结果:")
for i, C_i in enumerate(local_results):
    print(f"C_{i} (A @ B_{i}):\n", C_i, "\nshape:", C_i.shape)

# 4. 模拟 allgather：拼接所有局部结果
C_final = np.concatenate(local_results, axis=1)
print("\n合并后的 C_final:\n", C_final, "\nshape:", C_final.shape)

# 5. 验证结果与直接乘法的等价性
C_ground_truth = A @ B
print("\n标准乘法结果 (A @ B):\n", C_ground_truth)
print("\n验证一致性:", np.array_equal(C_final, C_ground_truth))

