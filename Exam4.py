# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:02:10 2024

@author: q999s
"""

'''
Exam4:
    后端：aer_simulator模拟
    噪声：退极化噪声模型(包括门及测量)
    电路：
        1. (13-qubit 纠错码制备及纠错过程) + ([4,1,4]有噪声训练 DQNN辅助态制备) 
             + (已训练的RealAmplitudes regular term)
        2. (13-qubit 纠错码制备及纠错过程) + 有纠错过程的cat state制备电路
    目的：相比传统带纠正模块的cat state，体现DQNN辅助态的优势（保真度，逻辑错误率）
    输出：实验图片
    结果分析：在这种噪声环境下，给出的结果分析具有明显差异性
    备注：13量子比特纠错码编码电路来自于Stim软件生成
'''
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer
from qiskit.quantum_info import Statevector
from UserDefined_noise_model import depo_err_model
from qiskit import execute
from tools import count_leftmost_bits_percentage
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from pathlib import Path

def Encoding():
    # Build a Encoding-circuit
    q_in = QuantumRegister(13)
    circ = QuantumCircuit(q_in, name='Encoding')
    
    circ.h(q_in[0])
    circ.h(q_in[1])
    circ.h(q_in[2])
    circ.h(q_in[3])
    circ.h(q_in[4])
    circ.h(q_in[5])
    
    circ.cx(q_in[0],q_in[1])
    circ.cx(q_in[0],q_in[2])    
    circ.cx(q_in[0],q_in[3])
    circ.cx(q_in[0],q_in[4])
    circ.cx(q_in[0],q_in[5])
    circ.cx(q_in[0],q_in[6])
    circ.cx(q_in[0],q_in[12])
    
    circ.cx(q_in[1],q_in[3])
    circ.cx(q_in[1],q_in[5])
    circ.cx(q_in[1],q_in[7])
    circ.cx(q_in[1],q_in[12])
        
    circ.cx(q_in[8],q_in[2])
    circ.cx(q_in[2],q_in[8])
    circ.cx(q_in[8],q_in[2])
    circ.cx(q_in[11],q_in[2])
    circ.cx(q_in[12],q_in[2])
    
    circ.cx(q_in[8],q_in[3])
    circ.cx(q_in[3],q_in[8])
    circ.cx(q_in[8],q_in[3])
    circ.cx(q_in[3],q_in[4])
    circ.cx(q_in[3],q_in[6])
    circ.cx(q_in[3],q_in[7])
    
    circ.cx(q_in[8],q_in[4])
    circ.cx(q_in[4],q_in[8])
    circ.cx(q_in[8],q_in[4])
    circ.cx(q_in[4],q_in[5])
    circ.cx(q_in[4],q_in[7])
    circ.cx(q_in[4],q_in[12])
    
    circ.cx(q_in[6],q_in[5])
    circ.cx(q_in[5],q_in[6])
    circ.cx(q_in[6],q_in[5])
    circ.cx(q_in[5],q_in[9])

    circ.cx(q_in[7],q_in[6])
    circ.cx(q_in[6],q_in[7])
    circ.cx(q_in[7],q_in[6])    
    circ.cx(q_in[6],q_in[10]) 
    
    circ.cx(q_in[11],q_in[7])
    circ.cx(q_in[7],q_in[11])
    circ.cx(q_in[11],q_in[7])    
    circ.cx(q_in[12],q_in[7])   
    
    circ.cx(q_in[8],q_in[9])    
    circ.cx(q_in[8],q_in[10])  
    
    circ.cx(q_in[11],q_in[9])
    circ.cx(q_in[9],q_in[11])
    circ.cx(q_in[11],q_in[9]) 
    circ.cx(q_in[9],q_in[10])
    circ.cx(q_in[9],q_in[12])
    
    circ.cx(q_in[11],q_in[10])
    circ.cx(q_in[10],q_in[11])
    circ.cx(q_in[11],q_in[10]) 
    
    qc_instru = circ.to_instruction()
    
    return qc_instru

def DQNN_cat_state():
    # 需要配备5个量子比特
    matrix_list = np.load("Noise_training_network.npz")
    # load DQNN model
    weight_mat = list(matrix_list.values())
    q_in = QuantumRegister(4)
    q_hidden = QuantumRegister(1)
    c_in = ClassicalRegister(4)
    c_hidden = ClassicalRegister(1)
    
    circ = QuantumCircuit(q_in, q_hidden, c_in, c_hidden,name='DQNN_cat')
    # 第一层计算
    # 直接制备cat state
    circ.h(q_in[0])
    circ.cx(q_in[0],q_in[1])
    circ.cx(q_in[0],q_in[2])
    circ.cx(q_in[0],q_in[3])  
    # 施加当前权重矩阵
    circ.unitary(weight_mat[0],[q_hidden[0], q_in[0],q_in[1],q_in[2],q_in[3]])
    # 测量去除前一层    
    circ.measure(q_in,c_in)
    circ.reset(q_in)   
    # 第二层计算
    circ.unitary(weight_mat[1], [q_in[0],q_in[1],q_in[2],q_in[3], q_hidden[0]])
    circ.unitary(weight_mat[2], [q_in[0],q_in[1],q_in[2],q_in[3], q_hidden[0]])
    circ.unitary(weight_mat[3], [q_in[0],q_in[1],q_in[2],q_in[3], q_hidden[0]])
    circ.unitary(weight_mat[4], [q_in[0],q_in[1],q_in[2],q_in[3], q_hidden[0]])

    # 测量去除前一层
    circ.measure(q_hidden[0],c_hidden)
    
    circ_instru = circ.to_instruction()
    
    return circ_instru

def cat_state():
    q_in = QuantumRegister(7)
    syn = ClassicalRegister(3)
    circ = QuantumCircuit(q_in, syn, name='cat')
    # 直接制备cat state
    circ.h(q_in[0])
    circ.cx(q_in[0],q_in[1])
    circ.cx(q_in[0],q_in[2])
    circ.cx(q_in[0],q_in[3]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(q_in[0],q_in[4])
    circ.cx(q_in[1],q_in[4])
    
    circ.cx(q_in[0],q_in[5])
    circ.cx(q_in[2],q_in[5])
    
    circ.cx(q_in[0],q_in[6])
    circ.cx(q_in[3],q_in[6])
    
    circ.measure([q_in[4],q_in[5],q_in[6]], syn)
    for i in [4,5,6]:
        circ.reset(q_in[i])
        
    with circ.switch(syn) as case:
        # X-error
        with case(7):
            circ.x(q_in[0])
        with case(1):
            circ.x(q_in[1])
        with case(2):
            circ.x(q_in[2])
        with case(4):
            circ.x(q_in[3]) 
            
    circ_instru = circ.to_instruction()
    
    return circ_instru

def cat_state_measure():
    q_in = QuantumRegister(4)
    syn = ClassicalRegister(1)
    circ = QuantumCircuit(q_in,syn,name='cat')
    # 还原cat state
    circ.cx(q_in[0],q_in[3]) 
    circ.cx(q_in[0],q_in[2])
    circ.cx(q_in[0],q_in[1])
    circ.h(q_in[0])
    
    circ.measure(q_in[0], syn)
    for i in range(4):
        circ.reset(q_in[i])
    
    circ_instru = circ.to_instruction()
    
    return circ_instru

def ideal_output():
    s_in = Statevector.from_label('0000000000000')
    Encoding_circ = Encoding()
    
    q_in = QuantumRegister(13)
    circ = QuantumCircuit(q_in)
    circ.append(Encoding_circ, q_in) 
    
    s_out = s_in.evolve(circ).probabilities()
    target_s = [np.abs(math.sqrt(item)) for item in s_out]
    
    return target_s
      
def syndrome_measure_cat(err_list):
    
    q_in = QuantumRegister(13)
    anci = QuantumRegister(7)
    temp1 = ClassicalRegister(3)
    temp2 = ClassicalRegister(3)
    temp3 = ClassicalRegister(3)
    temp4 = ClassicalRegister(3)
    temp5 = ClassicalRegister(3)
    temp6 = ClassicalRegister(3)
    temp7 = ClassicalRegister(3)
    temp8 = ClassicalRegister(3)
    temp9 = ClassicalRegister(3)
    temp10 = ClassicalRegister(3)
    temp11 = ClassicalRegister(3)
    temp12 = ClassicalRegister(3)
    
    syn_x = ClassicalRegister(6)
    syn_z = ClassicalRegister(6)
    final = ClassicalRegister(13)
    circ = QuantumCircuit(q_in, anci, temp1, temp2, temp3, temp4, temp5, temp6,
                          temp7, temp8, temp9, temp10, temp11, temp12,
                          syn_x, syn_z, final, name='syndrome_cat')
    Encoding_circ = Encoding()
    syn_measure = cat_state_measure()
    # 编码过程
    circ.append(Encoding_circ,q_in)
    
    # 稳定子测量
    # X(0,1,3)
    circ.h(anci[0])
    circ.cx(anci[0],anci[1])
    circ.cx(anci[0],anci[2])
    circ.cx(anci[0],anci[3]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(anci[0],anci[4])
    circ.cx(anci[1],anci[4])
    
    circ.cx(anci[0],anci[5])
    circ.cx(anci[2],anci[5])
    
    circ.cx(anci[0],anci[6])
    circ.cx(anci[3],anci[6])
    # 验证辅助态数据
    circ.measure([anci[4],anci[5],anci[6]], temp1)
    for i in [4,5,6]:
        circ.reset(anci[i])
    # 纠正辅助态 
    with circ.switch(temp1) as case:
        # X-error
        with case(7):
            circ.x(anci[0])
        with case(1):
            circ.x(anci[1])
        with case(2):
            circ.x(anci[2])
        with case(4):
            circ.x(anci[3]) 
    # 进行稳定子测量      
    circ.cx(anci[0], q_in[0])
    circ.cx(anci[1], q_in[1])
    circ.cx(anci[2], q_in[3])
    circ.append(syn_measure,anci[0:4],[syn_z[0]])
    
    # X(1,2,4)
    circ.h(anci[0])
    circ.cx(anci[0],anci[1])
    circ.cx(anci[0],anci[2])
    circ.cx(anci[0],anci[3]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(anci[0],anci[4])
    circ.cx(anci[1],anci[4])
    
    circ.cx(anci[0],anci[5])
    circ.cx(anci[2],anci[5])
    
    circ.cx(anci[0],anci[6])
    circ.cx(anci[3],anci[6])
    # 验证辅助态数据
    circ.measure([anci[4],anci[5],anci[6]], temp2)
    for i in [4,5,6]:
        circ.reset(anci[i])
    # 纠正辅助态 
    with circ.switch(temp2) as case:
        # X-error
        with case(7):
            circ.x(anci[0])
        with case(1):
            circ.x(anci[1])
        with case(2):
            circ.x(anci[2])
        with case(4):
            circ.x(anci[3]) 
    # 进行稳定子测量      
    circ.cx(anci[0], q_in[1])
    circ.cx(anci[1], q_in[2])
    circ.cx(anci[2], q_in[4])
    circ.append(syn_measure,anci[0:4],[syn_z[1]])    
    
    # X(3,5,6,8)
    circ.h(anci[0])
    circ.cx(anci[0],anci[1])
    circ.cx(anci[0],anci[2])
    circ.cx(anci[0],anci[3]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(anci[0],anci[4])
    circ.cx(anci[1],anci[4])
    
    circ.cx(anci[0],anci[5])
    circ.cx(anci[2],anci[5])
    
    circ.cx(anci[0],anci[6])
    circ.cx(anci[3],anci[6])
    # 验证辅助态数据
    circ.measure([anci[4],anci[5],anci[6]], temp3)
    for i in [4,5,6]:
        circ.reset(anci[i])
    # 纠正辅助态 
    with circ.switch(temp3) as case:
        # X-error
        with case(7):
            circ.x(anci[0])
        with case(1):
            circ.x(anci[1])
        with case(2):
            circ.x(anci[2])
        with case(4):
            circ.x(anci[3]) 
    # 进行稳定子测量      
    circ.cx(anci[0], q_in[3])
    circ.cx(anci[1], q_in[5])
    circ.cx(anci[2], q_in[6])
    circ.cx(anci[3], q_in[8])
    circ.append(syn_measure,anci[0:4],[syn_z[2]])        
    
    # X(4,6,7,9)
    circ.h(anci[0])
    circ.cx(anci[0],anci[1])
    circ.cx(anci[0],anci[2])
    circ.cx(anci[0],anci[3]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(anci[0],anci[4])
    circ.cx(anci[1],anci[4])
    
    circ.cx(anci[0],anci[5])
    circ.cx(anci[2],anci[5])
    
    circ.cx(anci[0],anci[6])
    circ.cx(anci[3],anci[6])
    # 验证辅助态数据
    circ.measure([anci[4],anci[5],anci[6]], temp4)
    for i in [4,5,6]:
        circ.reset(anci[i])
    # 纠正辅助态 
    with circ.switch(temp4) as case:
        # X-error
        with case(7):
            circ.x(anci[0])
        with case(1):
            circ.x(anci[1])
        with case(2):
            circ.x(anci[2])
        with case(4):
            circ.x(anci[3]) 
    # 进行稳定子测量      
    circ.cx(anci[0], q_in[4])
    circ.cx(anci[1], q_in[6])
    circ.cx(anci[2], q_in[7])
    circ.cx(anci[3], q_in[9])
    circ.append(syn_measure,anci[0:4],[syn_z[3]])    
    
    # X(8,10,11)
    circ.h(anci[0])
    circ.cx(anci[0],anci[1])
    circ.cx(anci[0],anci[2])
    circ.cx(anci[0],anci[3]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(anci[0],anci[4])
    circ.cx(anci[1],anci[4])
    
    circ.cx(anci[0],anci[5])
    circ.cx(anci[2],anci[5])
    
    circ.cx(anci[0],anci[6])
    circ.cx(anci[3],anci[6])
    # 验证辅助态数据
    circ.measure([anci[4],anci[5],anci[6]], temp5)
    for i in [4,5,6]:
        circ.reset(anci[i])
    # 纠正辅助态 
    with circ.switch(temp5) as case:
        # X-error
        with case(7):
            circ.x(anci[0])
        with case(1):
            circ.x(anci[1])
        with case(2):
            circ.x(anci[2])
        with case(4):
            circ.x(anci[3]) 
    # 进行稳定子测量      
    circ.cx(anci[0], q_in[8])
    circ.cx(anci[1], q_in[10])
    circ.cx(anci[2], q_in[11])
    circ.append(syn_measure,anci[0:4],[syn_z[4]])        
    
    # X(9,11,12)
    circ.h(anci[0])
    circ.cx(anci[0],anci[1])
    circ.cx(anci[0],anci[2])
    circ.cx(anci[0],anci[3]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(anci[0],anci[4])
    circ.cx(anci[1],anci[4])
    
    circ.cx(anci[0],anci[5])
    circ.cx(anci[2],anci[5])
    
    circ.cx(anci[0],anci[6])
    circ.cx(anci[3],anci[6])
    # 验证辅助态数据
    circ.measure([anci[4],anci[5],anci[6]], temp6)
    for i in [4,5,6]:
        circ.reset(anci[i])
    # 纠正辅助态 
    with circ.switch(temp6) as case:
        # X-error
        with case(7):
            circ.x(anci[0])
        with case(1):
            circ.x(anci[1])
        with case(2):
            circ.x(anci[2])
        with case(4):
            circ.x(anci[3]) 
    # 进行稳定子测量      
    circ.cx(anci[0], q_in[9])
    circ.cx(anci[1], q_in[11])
    circ.cx(anci[2], q_in[12])
    circ.append(syn_measure,anci[0:4],[syn_z[5]])     
    #-----------------------------------------------------------
    # Z(0,3,5)
    circ.h(anci[0])
    circ.cx(anci[0],anci[1])
    circ.cx(anci[0],anci[2])
    circ.cx(anci[0],anci[3]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(anci[0],anci[4])
    circ.cx(anci[1],anci[4])
    
    circ.cx(anci[0],anci[5])
    circ.cx(anci[2],anci[5])
    
    circ.cx(anci[0],anci[6])
    circ.cx(anci[3],anci[6])
    # 验证辅助态数据
    circ.measure([anci[4],anci[5],anci[6]], temp7)
    for i in [4,5,6]:
        circ.reset(anci[i])
    # 纠正辅助态 
    with circ.switch(temp7) as case:
        # X-error
        with case(7):
            circ.x(anci[0])
        with case(1):
            circ.x(anci[1])
        with case(2):
            circ.x(anci[2])
        with case(4):
            circ.x(anci[3]) 
    # 进行稳定子测量      
    circ.cz(anci[0], q_in[0])
    circ.cz(anci[1], q_in[3])
    circ.cz(anci[2], q_in[5])
    circ.append(syn_measure,anci[0:4],[syn_x[0]])   
    
    # Z(1,3,4,6)
    circ.h(anci[0])
    circ.cx(anci[0],anci[1])
    circ.cx(anci[0],anci[2])
    circ.cx(anci[0],anci[3]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(anci[0],anci[4])
    circ.cx(anci[1],anci[4])
    
    circ.cx(anci[0],anci[5])
    circ.cx(anci[2],anci[5])
    
    circ.cx(anci[0],anci[6])
    circ.cx(anci[3],anci[6])
    # 验证辅助态数据
    circ.measure([anci[4],anci[5],anci[6]], temp8)
    for i in [4,5,6]:
        circ.reset(anci[i])
    # 纠正辅助态 
    with circ.switch(temp8) as case:
        # X-error
        with case(7):
            circ.x(anci[0])
        with case(1):
            circ.x(anci[1])
        with case(2):
            circ.x(anci[2])
        with case(4):
            circ.x(anci[3]) 
    # 进行稳定子测量      
    circ.cz(anci[0], q_in[1])
    circ.cz(anci[1], q_in[3])
    circ.cz(anci[2], q_in[4])
    circ.cz(anci[2], q_in[6])
    circ.append(syn_measure,anci[0:4],[syn_x[1]])   

    # Z(2,4,7)
    circ.h(anci[0])
    circ.cx(anci[0],anci[1])
    circ.cx(anci[0],anci[2])
    circ.cx(anci[0],anci[3]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(anci[0],anci[4])
    circ.cx(anci[1],anci[4])
    
    circ.cx(anci[0],anci[5])
    circ.cx(anci[2],anci[5])
    
    circ.cx(anci[0],anci[6])
    circ.cx(anci[3],anci[6])
    # 验证辅助态数据
    circ.measure([anci[4],anci[5],anci[6]], temp9)
    for i in [4,5,6]:
        circ.reset(anci[i])
    # 纠正辅助态 
    with circ.switch(temp9) as case:
        # X-error
        with case(7):
            circ.x(anci[0])
        with case(1):
            circ.x(anci[1])
        with case(2):
            circ.x(anci[2])
        with case(4):
            circ.x(anci[3]) 
    # 进行稳定子测量      
    circ.cz(anci[0], q_in[2])
    circ.cz(anci[1], q_in[4])
    circ.cz(anci[2], q_in[7])
    circ.append(syn_measure,anci[0:4],[syn_x[2]])   

    # Z(5,8,10)
    circ.h(anci[0])
    circ.cx(anci[0],anci[1])
    circ.cx(anci[0],anci[2])
    circ.cx(anci[0],anci[3]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(anci[0],anci[4])
    circ.cx(anci[1],anci[4])
    
    circ.cx(anci[0],anci[5])
    circ.cx(anci[2],anci[5])
    
    circ.cx(anci[0],anci[6])
    circ.cx(anci[3],anci[6])
    # 验证辅助态数据
    circ.measure([anci[4],anci[5],anci[6]], temp10)
    for i in [4,5,6]:
        circ.reset(anci[i])
    # 纠正辅助态 
    with circ.switch(temp10) as case:
        # X-error
        with case(7):
            circ.x(anci[0])
        with case(1):
            circ.x(anci[1])
        with case(2):
            circ.x(anci[2])
        with case(4):
            circ.x(anci[3]) 
    # 进行稳定子测量      
    circ.cz(anci[0], q_in[5])
    circ.cz(anci[1], q_in[8])
    circ.cz(anci[2], q_in[10])
    circ.append(syn_measure,anci[0:4],[syn_x[3]])   

    # Z(6,8,9,11)
    circ.h(anci[0])
    circ.cx(anci[0],anci[1])
    circ.cx(anci[0],anci[2])
    circ.cx(anci[0],anci[3]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(anci[0],anci[4])
    circ.cx(anci[1],anci[4])
    
    circ.cx(anci[0],anci[5])
    circ.cx(anci[2],anci[5])
    
    circ.cx(anci[0],anci[6])
    circ.cx(anci[3],anci[6])
    # 验证辅助态数据
    circ.measure([anci[4],anci[5],anci[6]], temp11)
    for i in [4,5,6]:
        circ.reset(anci[i])
    # 纠正辅助态 
    with circ.switch(temp11) as case:
        # X-error
        with case(7):
            circ.x(anci[0])
        with case(1):
            circ.x(anci[1])
        with case(2):
            circ.x(anci[2])
        with case(4):
            circ.x(anci[3]) 
    # 进行稳定子测量      
    circ.cz(anci[0], q_in[6])
    circ.cz(anci[1], q_in[8])
    circ.cz(anci[2], q_in[9])
    circ.cz(anci[2], q_in[11])
    circ.append(syn_measure,anci[0:4],[syn_x[4]])   

    # Z(7,9,12)
    circ.h(anci[0])
    circ.cx(anci[0],anci[1])
    circ.cx(anci[0],anci[2])
    circ.cx(anci[0],anci[3]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(anci[0],anci[4])
    circ.cx(anci[1],anci[4])
    
    circ.cx(anci[0],anci[5])
    circ.cx(anci[2],anci[5])
    
    circ.cx(anci[0],anci[6])
    circ.cx(anci[3],anci[6])
    # 验证辅助态数据
    circ.measure([anci[4],anci[5],anci[6]], temp12)
    for i in [4,5,6]:
        circ.reset(anci[i])
    # 纠正辅助态 
    with circ.switch(temp12) as case:
        # X-error
        with case(7):
            circ.x(anci[0])
        with case(1):
            circ.x(anci[1])
        with case(2):
            circ.x(anci[2])
        with case(4):
            circ.x(anci[3]) 
    # 进行稳定子测量      
    circ.cz(anci[0], q_in[7])
    circ.cz(anci[1], q_in[9])
    circ.cz(anci[2], q_in[12])
    circ.append(syn_measure,anci[0:4],[syn_x[5]])       
    
    # Minimum weight Match Decoding process
    with circ.switch(syn_x) as case:
        # X-error
        with case(1):
            circ.x(q_in[0])
        with case(2):
            circ.x(q_in[1])
        with case(4):
            circ.x(q_in[2])
        with case(3):
            circ.x(q_in[3])            
        with case(6):
            circ.x(q_in[4])
        with case(9):
            circ.x(q_in[5])
        with case(18):
            circ.x(q_in[6])
        with case(36):
            circ.x(q_in[7])
        with case(24):
            circ.x(q_in[8])
        with case(48):
            circ.x(q_in[9])
        with case(8):
            circ.x(q_in[10])            
        with case(16):
            circ.x(q_in[11])
        with case(32):
            circ.x(q_in[12])          
        with case(case.DEFAULT):
            circ.id(q_in[0])   

    with circ.switch(syn_z) as case:
        # Z-error
        with case(1):
            circ.z(q_in[0])
        with case(3):
            circ.z(q_in[1])
        with case(2):
            circ.z(q_in[2])
        with case(5):
            circ.z(q_in[3])            
        with case(10):
            circ.z(q_in[4])
        with case(4):
            circ.z(q_in[5])
        with case(12):
            circ.z(q_in[6])
        with case(8):
            circ.z(q_in[7])
        with case(20):
            circ.z(q_in[8])
        with case(40):
            circ.z(q_in[9])
        with case(16):
            circ.z(q_in[10])            
        with case(48):
            circ.z(q_in[11])
        with case(32):
            circ.z(q_in[12])          
        with case(case.DEFAULT):
            circ.id(q_in[0])                   
      
    circ.measure(q_in, final)
    
    fidelity = []
    # 单轮纠错，开始执行电路
    for err in err_list:
        noise_model, _ = depo_err_model(err,1)
        counts = execute(circ, Aer.get_backend('aer_simulator'), 
                      noise_model=noise_model,shots = 4096).result().get_counts()
        # 
        output_s = count_leftmost_bits_percentage(counts) 
        ideal_s = ideal_output()
        res = np.inner(output_s, ideal_s)**2 
        fidelity.append(res)
        
    return fidelity

def syndrome_measure_DQNN_cat(err_list):
    
    q_in = QuantumRegister(13)
    anci = QuantumRegister(6) 
    temp = ClassicalRegister(60)
    syn1 = ClassicalRegister(4)
    syn2 = ClassicalRegister(4)
    syn3 = ClassicalRegister(4)
    syn4 = ClassicalRegister(4)
    syn5 = ClassicalRegister(4)
    syn6 = ClassicalRegister(4)
    syn7 = ClassicalRegister(4)
    syn8 = ClassicalRegister(4)
    syn9 = ClassicalRegister(4)
    syn10 = ClassicalRegister(4)
    syn11 = ClassicalRegister(4)
    syn12 = ClassicalRegister(4)
    syn_x = ClassicalRegister(6)
    syn_z = ClassicalRegister(6)
    final = ClassicalRegister(13)
    circ = QuantumCircuit(q_in, anci, temp, syn1, syn2, syn3, syn4, syn5, syn6, 
                          syn7, syn8, syn9, syn10, syn11, syn12, 
                          syn_x, syn_z, final, name='syndrome_cat')
    Encoding_circ = Encoding()
    anci_prep = DQNN_cat_state()
    
    # 编码过程
    circ.append(Encoding_circ,q_in)
    
    # 稳定子测量
    # X(0,1,3)
    circ.append(anci_prep,anci[0:5],temp[0:5])
    circ.cx(anci[0], q_in[0])
    circ.cx(anci[1], q_in[1])
    circ.cx(anci[2], q_in[3])
    circ.measure(anci[0:4], syn1)
    for i in list(range(5)):
        circ.reset(anci[i])
    
    # X(1,2,4)
    circ.append(anci_prep,anci[0:5],temp[5:10])
    circ.cx(anci[0], q_in[1])
    circ.cx(anci[1], q_in[2])
    circ.cx(anci[2], q_in[4])
    circ.measure(anci[0:4], syn2)
    for i in list(range(5)):
        circ.reset(anci[i])
    
    # X(3,5,6,8)
    circ.append(anci_prep,anci[0:5],temp[10:15])
    circ.cx(anci[0], q_in[3])
    circ.cx(anci[1], q_in[5])
    circ.cx(anci[2], q_in[6])
    circ.cx(anci[3], q_in[8])
    circ.measure(anci[0:4], syn3)
    for i in list(range(5)):
        circ.reset(anci[i])

    # X(4,6,7,9)
    circ.append(anci_prep,anci[0:5],temp[15:20])
    circ.cx(anci[0], q_in[4])
    circ.cx(anci[1], q_in[6])
    circ.cx(anci[2], q_in[7])
    circ.cx(anci[3], q_in[9])
    circ.measure(anci[0:4], syn4)
    for i in list(range(5)):
        circ.reset(anci[i])

    # X(8,10,11)
    circ.append(anci_prep,anci[0:5],temp[20:25])
    circ.cx(anci[0], q_in[8])
    circ.cx(anci[1], q_in[10])
    circ.cx(anci[2], q_in[11])
    circ.measure(anci[0:4], syn5)
    for i in list(range(5)):
        circ.reset(anci[i])
    
    # X(9,11,12)
    circ.append(anci_prep,anci[0:5],temp[25:30])
    circ.cx(anci[0], q_in[9])
    circ.cx(anci[1], q_in[11])
    circ.cx(anci[2], q_in[12])
    circ.measure(anci[0:4], syn6)
    for i in list(range(5)):
        circ.reset(anci[i])   
    #--------------------------------------------------------
    # Z(0,3,5)
    circ.append(anci_prep,anci[0:5],temp[30:35])
    circ.cz(anci[0], q_in[0])
    circ.cz(anci[1], q_in[3])
    circ.cz(anci[2], q_in[5])
    circ.measure(anci[0:4], syn7)
    for i in list(range(5)):
        circ.reset(anci[i])      
        
    # Z(1,3,4,6)
    circ.append(anci_prep,anci[0:5],temp[35:40])
    circ.cz(anci[0], q_in[1])
    circ.cz(anci[1], q_in[3])
    circ.cz(anci[2], q_in[4])
    circ.cz(anci[3], q_in[6])
    circ.measure(anci[0:4], syn8)
    for i in list(range(5)):
        circ.reset(anci[i])  
        
    # Z(2,4,7)
    circ.append(anci_prep,anci[0:5],temp[40:45])
    circ.cz(anci[0], q_in[2])
    circ.cz(anci[1], q_in[4])
    circ.cz(anci[2], q_in[7])
    circ.measure(anci[0:4], syn9)
    for i in list(range(5)):
        circ.reset(anci[i])  

    # Z(5,8,10)
    circ.append(anci_prep,anci[0:5],temp[45:50])
    circ.cz(anci[0], q_in[5])
    circ.cz(anci[1], q_in[8])
    circ.cz(anci[2], q_in[10])
    circ.measure(anci[0:4], syn10)
    for i in list(range(5)):
        circ.reset(anci[i])  
        
    # Z(6,8,9,11)
    circ.append(anci_prep,anci[0:5],temp[50:55])
    circ.cz(anci[0], q_in[6])
    circ.cz(anci[1], q_in[8])
    circ.cz(anci[2], q_in[9])
    circ.cz(anci[3], q_in[11])
    circ.measure(anci[0:4], syn11)
    for i in list(range(5)):
        circ.reset(anci[i])  
        
    # Z(7,9,12)
    circ.append(anci_prep,anci[0:5],temp[55:60])
    circ.cz(anci[0], q_in[7])
    circ.cz(anci[1], q_in[9])
    circ.cz(anci[2], q_in[12])
    circ.measure(anci[0:4], syn12)
    for i in list(range(5)):
        circ.reset(anci[i])  
    # ----------------------------------------------------------
    # 记录测量结果
    # 此处是通过辅助态的奇偶特征，判定稳定子测量的结果是否为非平凡的
    record1 = [syn1, syn2, syn3, syn4, syn5, syn6]
    record2 = [syn7, syn8, syn9, syn10, syn11, syn12]
    
    for i in list(range(6)):   
        with circ.switch(record1[i]) as case:
            with case(1,2,4,8):
                circ.x(anci[i])
            with case(7,11,13,14):
                circ.x(anci[i])
            with case(case.DEFAULT):
                circ.id(anci[i])        
    circ.measure(anci,  syn_z)
    
    for i in list(range(6)):  
        circ.reset(anci[i]) 
    
    for i in list(range(6)):   
        with circ.switch(record2[i]) as case:
            with case(1,2,4,8):
                circ.x(anci[i])
            with case(7,11,13,14):
                circ.x(anci[i])
            with case(case.DEFAULT):
                circ.id(anci[i])        
    circ.measure(anci,  syn_x)
    
    for i in list(range(6)):  
        circ.reset(anci[i]) 
    
    # Minimum weight Match Decoding process
    with circ.switch(syn_x) as case:
        # X-error
        with case(1):
            circ.x(q_in[0])
        with case(2):
            circ.x(q_in[1])
        with case(4):
            circ.x(q_in[2])
        with case(3):
            circ.x(q_in[3])            
        with case(6):
            circ.x(q_in[4])
        with case(9):
            circ.x(q_in[5])
        with case(18):
            circ.x(q_in[6])
        with case(36):
            circ.x(q_in[7])
        with case(24):
            circ.x(q_in[8])
        with case(48):
            circ.x(q_in[9])
        with case(8):
            circ.x(q_in[10])            
        with case(16):
            circ.x(q_in[11])
        with case(32):
            circ.x(q_in[12])          
        with case(case.DEFAULT):
            circ.id(q_in[0])   

    with circ.switch(syn_z) as case:
        # Z-error
        with case(1):
            circ.z(q_in[0])
        with case(3):
            circ.z(q_in[1])
        with case(2):
            circ.z(q_in[2])
        with case(5):
            circ.z(q_in[3])            
        with case(10):
            circ.z(q_in[4])
        with case(4):
            circ.z(q_in[5])
        with case(12):
            circ.z(q_in[6])
        with case(8):
            circ.z(q_in[7])
        with case(20):
            circ.z(q_in[8])
        with case(40):
            circ.z(q_in[9])
        with case(16):
            circ.z(q_in[10])            
        with case(48):
            circ.z(q_in[11])
        with case(32):
            circ.z(q_in[12])          
        with case(case.DEFAULT):
            circ.id(q_in[0])                   
      
    circ.measure(q_in, final)
    
    fidelity = []
    # 单轮纠错，开始执行电路
    N = len(err_list)
    k = 0
    for err in err_list:
        noise_model, _ = depo_err_model(err,1)
        counts = execute(circ, Aer.get_backend('aer_simulator'), 
                      noise_model=noise_model,shots = 4096).result().get_counts()
        # 
        output_s = count_leftmost_bits_percentage(counts) 
        ideal_s = ideal_output()
        res = np.inner(output_s, ideal_s)**2 
        fidelity.append(res)
        k = k + 1
        progress = k/N
        print("{:.2%}".format(progress))  
        
    return fidelity
   

# err_list = np.linspace(0.0001, 0.001, 10)

# fidelity1 = syndrome_measure_cat(err_list)
# fidelity2 = syndrome_measure_DQNN_cat(err_list) 

# 初始化参数
err_list = np.linspace(0.0001, 0.001, 8)
num_experiments = 5  # 实验次数
results = []

# 重复收集数据
for exp_idx in range(num_experiments):
    print(f"Running experiment {exp_idx+1}/{num_experiments}")
    
    # 生成单组数据
    fidelity1 = syndrome_measure_cat(err_list)
    fidelity2 = syndrome_measure_DQNN_cat(err_list)
    print("the current exp_num is:",exp_idx)
    # 存储结构化数据
    for i, err in enumerate(err_list):
        results.append({
            "Experiment": exp_idx + 1,
            "Error Rate": err,
            "Fidelity1": fidelity1[i],
            "Fidelity2": fidelity2[i]
        })

# 转换为DataFrame并保存
df = pd.DataFrame(results)

# 按实验编号和错误率排序
df = df.sort_values(by=["Experiment", "Error Rate"])

# 保存到CSV文件
output_path = Path.cwd() / "fidelity_comparison.csv"
df.to_csv(output_path, index=False)

print(f"数据已保存至: {output_path}")


#fidelity1 = [0.9920231367164541, 0.9813029290097871, 0.9724227514952116, 0.9625930076675666, 0.9567934956137183, 0.9447582285884104, 0.9328360555756695, 0.9252380124217345]
#fidelity2 = [0.9933093144005001, 0.9869183353518768, 0.9808149109670157, 0.9762987990142382, 0.9710230040846868, 0.9668332806746024, 0.9594199976146188, 0.9535291801546748]
# plt.plot(err_list, fidelity1, color="red", linestyle="dashed", linewidth=2, marker="o", label="normal")

# plt.plot(err_list, fidelity2, color="blue", linestyle="solid", linewidth=2, marker="o", label="regular")

# # 显示legend，位置为右上角
# plt.legend(loc=3)

# plt.xlabel("physical noise rate")
# plt.ylabel("fidelity")
# plt.title("7-qubit basic encoding state prep")
# plt.show()