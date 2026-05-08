import time
import numpy as np 
import matplotlib.pyplot as plt 

def matrix_multiply(A, B):
    A_shape = A.shape
    B_shape = B.shape
    rows_A = A_shape[0]
    cols_A = A_shape[1]
    rows_B = B_shape[0]
    cols_B = B_shape[1]
    assert cols_A == rows_B
    C = np.zeros((rows_A, cols_B))
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(rows_B):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matrix_multiply_blocked(A, B, BLOCK_SIZE):
    M, K = A.shape
    K, N = B.shape

    C = np.zeros((M, N), dtype=np.float32)
    # 在m轴上遍历块
    for m in range(0, M, BLOCK_SIZE):
         # 在n轴上遍历块
        for n in range(0, N, BLOCK_SIZE):
            acc = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)
            # 在k轴上遍历块
            for k in range(0, K, BLOCK_SIZE):
                a = A[m: m + BLOCK_SIZE, k: k + BLOCK_SIZE]
                b = B[k: k + BLOCK_SIZE, n: n + BLOCK_SIZE]

                acc += np.dot(a, b)

            C[m: m + BLOCK_SIZE, n: n + BLOCK_SIZE] = acc

    return C

