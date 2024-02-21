# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 10:37:53 2023

@author: q999s
"""
import math
import numpy as np
from scipy.linalg import qr
import matplotlib.pyplot as plt

def is_unitary(matrix):
    # 计算矩阵和其共轭转置的乘积
    product = np.dot(matrix, np.conj(matrix.T))
    # 比较乘积是否接近单位矩阵
    return np.allclose(product, np.identity(matrix.shape[0]))

def is_hermitian(matrix):
    # 检查矩阵是否为二维
    if len(matrix.shape) != 2:
        return False

    # 检查矩阵是否为方阵
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # 检查共轭转置是否等于矩阵自身的元素
    return np.allclose(matrix, np.conj(matrix).T)

def random_hermitian(n):
    # 生成一个随机化的初始n阶hermitian矩阵以作为权重系数矩阵
    random_matrix = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    
    # 让矩阵成为Hermitian矩阵（共轭转置等于自身）
    hermitian_matrix = (random_matrix + random_matrix.conj().T) / 2
    
    return hermitian_matrix

def random_unitary_matrix(n):
    # 生成一个随机矩阵
    random_matrix = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    
    # 对矩阵进行QR分解，得到酉矩阵
    q, r = qr(random_matrix, mode='full')
    
    return q

def Result_presentation(A):             
    # 遍历A中的每个矩阵
    for i, matrix in enumerate(A):
    
        # 创建一个三维坐标轴
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
        # 绘制三维直方图
        xpos, ypos = np.meshgrid(np.arange(4), np.arange(4))
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros(16)
        dx = dy = np.ones(16)
        dz = matrix.ravel()
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b')
    
        # 保存图形到本地，按照矩阵所在列表的序号来命名
        plt.savefig(f'matrix_{i}.png')
        
def count_leftmost_bits_percentage(counts):
    max_bit_length = 0
    total_count = 0

    for key, value in counts.items():
        leftmost_bits = key.split()[0]
        max_bit_length = max(max_bit_length, len(leftmost_bits))
        total_count += value

    max_decimal = 2**max_bit_length - 1
    result = [0] * (max_decimal + 1)

    for key, value in counts.items():
        leftmost_bits = key.split()[0]
        decimal_value = int(leftmost_bits, 2)
        result[decimal_value] += value / total_count
    
    # 注意！在这里我们返回的是一个量子态对应状态向量
    result = [np.abs(math.sqrt(item)) for item in result]
    # 根据测量结果输出
    return result

def partial_trace(rho, keep, dims, optimize=False):
    """Calculate the partial trace

    ρ_a = Tr_b(ρ)

    Parameters
    ----------
    ρ : 2D array
        Matrix to trace
    keep : array
        An array of indices of the spaces to keep after
        being traced. For instance, if the space is
        A x B x C x D and we want to trace out B and D,
        keep = [0,2]
    dims : array
        An array of the dimensions of each space.
        For instance, if the space is A x B x C x D,
        dims = [dim_A, dim_B, dim_C, dim_D]

    Returns
    -------
    ρ_a : 2D array
        Traced matrix
    """
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(Ndim)]
    idx2 = [Ndim+i if i in keep else i for i in range(Ndim)]
    rho_a = rho.reshape(np.tile(dims,2))
    rho_a = np.einsum(rho_a, idx1+idx2, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)



