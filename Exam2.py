# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:02:10 2024

@author: q999s
"""

'''
Exam2:
    后端：aer_simulator模拟
    噪声：退极化噪声模型(包括门及测量)
    电路：
        1. (7-qubit 纠错码制备及纠错过程) + ([4,1,4]有噪声训练 DQNN辅助态制备) 
             + (已训练的RealAmplitudes regular term)
        2. (7-qubit 纠错码制备及纠错过程) + 有纠错过程的cat state制备电路
    目的：相比传统带纠正模块的cat state，体现DQNN辅助态的优势（保真度，逻辑错误率）
    输出：实验图片
    结果分析：在这种噪声环境下，给出的结果分析具有明显差异性
    备注：七量子比特纠错码编码电路来自于Stac软件包生成(github:https://abdullahkhalid.com/qecft/introduction/stac/)
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

def Encoding():
    # Build a Encoding-circuit
    q_in = QuantumRegister(7)
    circ = QuantumCircuit(q_in, name='Encoding')
    
    circ.h(q_in[0])
    
    circ.cx(q_in[6],q_in[4])
    circ.cx(q_in[6],q_in[5])
    
    circ.cx(q_in[0],q_in[3])
    circ.cx(q_in[0],q_in[5])
    circ.cx(q_in[0],q_in[6])
    
    circ.h(q_in[1])
    circ.cx(q_in[1],q_in[3])
    circ.cx(q_in[1],q_in[4])
    circ.cx(q_in[1],q_in[6])
    
    circ.h(q_in[2])
    circ.cx(q_in[2],q_in[3])
    circ.cx(q_in[2],q_in[4])
    circ.cx(q_in[2],q_in[5])
    
    circ.z(q_in[2])
    
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
    s_in = Statevector.from_label('0000000')
    Encoding_circ = Encoding()
    
    q_in = QuantumRegister(7)
    circ = QuantumCircuit(q_in)
    circ.append(Encoding_circ, q_in) 
    
    s_out = s_in.evolve(circ).probabilities()
    target_s = [np.abs(math.sqrt(item)) for item in s_out]
    
    return target_s
      
def syndrome_measure_cat(err_list):
    
    q_in = QuantumRegister(14)
    temp1 = ClassicalRegister(3)
    temp2 = ClassicalRegister(3)
    temp3 = ClassicalRegister(3)
    temp4 = ClassicalRegister(3)
    temp5 = ClassicalRegister(3)
    temp6 = ClassicalRegister(3)
    
    syn_x = ClassicalRegister(3)
    syn_z = ClassicalRegister(3)
    final = ClassicalRegister(7)
    circ = QuantumCircuit(q_in, temp1, temp2, temp3, temp4, temp5, temp6,
                          syn_x, syn_z, final, name='syndrome_cat')
    Encoding_circ = Encoding()
    syn_measure = cat_state_measure()
    # 编码过程
    circ.append(Encoding_circ,q_in[0:7])
    
    # 稳定子测量
    # Z(0,1,2,3)
    circ.h(q_in[7])
    circ.cx(q_in[7],q_in[8])
    circ.cx(q_in[7],q_in[9])
    circ.cx(q_in[7],q_in[10]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(q_in[7],q_in[11])
    circ.cx(q_in[8],q_in[11])
    
    circ.cx(q_in[7],q_in[12])
    circ.cx(q_in[9],q_in[12])
    
    circ.cx(q_in[7],q_in[13])
    circ.cx(q_in[10],q_in[13])
    
    circ.measure([q_in[11],q_in[12],q_in[13]], temp1)
    for i in [11,12,13]:
        circ.reset(q_in[i])
        
    with circ.switch(temp1) as case:
        # X-error
        with case(7):
            circ.x(q_in[7])
        with case(1):
            circ.x(q_in[8])
        with case(2):
            circ.x(q_in[9])
        with case(4):
            circ.x(q_in[10]) 
            
    circ.cz(q_in[7], q_in[0])
    circ.cz(q_in[8], q_in[1])
    circ.cz(q_in[9], q_in[2])
    circ.cz(q_in[10], q_in[3])
    circ.append(syn_measure,q_in[7:11],[syn_z[0]])
    
    # Z(0,1,4,5)
    circ.h(q_in[7])
    circ.cx(q_in[7],q_in[8])
    circ.cx(q_in[7],q_in[9])
    circ.cx(q_in[7],q_in[10]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(q_in[7],q_in[11])
    circ.cx(q_in[8],q_in[11])
    
    circ.cx(q_in[7],q_in[12])
    circ.cx(q_in[9],q_in[12])
    
    circ.cx(q_in[7],q_in[13])
    circ.cx(q_in[10],q_in[13])
    
    circ.measure([q_in[11],q_in[12],q_in[13]], temp2)
    for i in [11,12,13]:
        circ.reset(q_in[i])
        
    with circ.switch(temp2) as case:
        # X-error
        with case(7):
            circ.x(q_in[7])
        with case(1):
            circ.x(q_in[8])
        with case(2):
            circ.x(q_in[9])
        with case(4):
            circ.x(q_in[10]) 
            
    circ.cz(q_in[7], q_in[0])
    circ.cz(q_in[8], q_in[1])
    circ.cz(q_in[9], q_in[4])
    circ.cz(q_in[10], q_in[5])
    circ.append(syn_measure,q_in[7:11],[syn_z[1]])
        
    # Z(0,2,4,6)
    circ.h(q_in[7])
    circ.cx(q_in[7],q_in[8])
    circ.cx(q_in[7],q_in[9])
    circ.cx(q_in[7],q_in[10]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(q_in[7],q_in[11])
    circ.cx(q_in[8],q_in[11])
    
    circ.cx(q_in[7],q_in[12])
    circ.cx(q_in[9],q_in[12])
    
    circ.cx(q_in[7],q_in[13])
    circ.cx(q_in[10],q_in[13])
    
    circ.measure([q_in[11],q_in[12],q_in[13]], temp3)
    for i in [11,12,13]:
        circ.reset(q_in[i])
        
    with circ.switch(temp3) as case:
        # X-error
        with case(7):
            circ.x(q_in[7])
        with case(1):
            circ.x(q_in[8])
        with case(2):
            circ.x(q_in[9])
        with case(4):
            circ.x(q_in[10]) 
            
    circ.cz(q_in[7], q_in[0])
    circ.cz(q_in[8], q_in[2])
    circ.cz(q_in[9], q_in[4])
    circ.cz(q_in[10], q_in[6])
    circ.append(syn_measure,q_in[7:11],[syn_z[2]])
        
    # X(0,1,2,3)
    circ.h(q_in[7])
    circ.cx(q_in[7],q_in[8])
    circ.cx(q_in[7],q_in[9])
    circ.cx(q_in[7],q_in[10]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(q_in[7],q_in[11])
    circ.cx(q_in[8],q_in[11])
    
    circ.cx(q_in[7],q_in[12])
    circ.cx(q_in[9],q_in[12])
    
    circ.cx(q_in[7],q_in[13])
    circ.cx(q_in[10],q_in[13])
    
    circ.measure([q_in[11],q_in[12],q_in[13]], temp4)
    for i in [11,12,13]:
        circ.reset(q_in[i])
        
    with circ.switch(temp4) as case:
        # X-error
        with case(7):
            circ.x(q_in[7])
        with case(1):
            circ.x(q_in[8])
        with case(2):
            circ.x(q_in[9])
        with case(4):
            circ.x(q_in[10]) 
            
    circ.cx(q_in[7], q_in[0])
    circ.cx(q_in[8], q_in[1])
    circ.cx(q_in[9], q_in[2])
    circ.cx(q_in[10], q_in[3])
    circ.append(syn_measure,q_in[7:11],[syn_x[0]])
    
    # X(0,1,4,5)
    circ.h(q_in[7])
    circ.cx(q_in[7],q_in[8])
    circ.cx(q_in[7],q_in[9])
    circ.cx(q_in[7],q_in[10]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(q_in[7],q_in[11])
    circ.cx(q_in[8],q_in[11])
    
    circ.cx(q_in[7],q_in[12])
    circ.cx(q_in[9],q_in[12])
    
    circ.cx(q_in[7],q_in[13])
    circ.cx(q_in[10],q_in[13])
    
    circ.measure([q_in[11],q_in[12],q_in[13]], temp5)
    for i in [11,12,13]:
        circ.reset(q_in[i])
        
    with circ.switch(temp5) as case:
        # X-error
        with case(7):
            circ.x(q_in[7])
        with case(1):
            circ.x(q_in[8])
        with case(2):
            circ.x(q_in[9])
        with case(4):
            circ.x(q_in[10]) 
            
    circ.cx(q_in[7], q_in[0])
    circ.cx(q_in[8], q_in[1])
    circ.cx(q_in[9], q_in[4])
    circ.cx(q_in[10], q_in[5])
    circ.append(syn_measure,q_in[7:11],[syn_x[1]])
        
    # X(0,2,4,6)
    circ.h(q_in[7])
    circ.cx(q_in[7],q_in[8])
    circ.cx(q_in[7],q_in[9])
    circ.cx(q_in[7],q_in[10]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(q_in[7],q_in[11])
    circ.cx(q_in[8],q_in[11])
    
    circ.cx(q_in[7],q_in[12])
    circ.cx(q_in[9],q_in[12])
    
    circ.cx(q_in[7],q_in[13])
    circ.cx(q_in[10],q_in[13])
    
    circ.measure([q_in[11],q_in[12],q_in[13]], temp6)
    for i in [11,12,13]:
        circ.reset(q_in[i])
        
    with circ.switch(temp6) as case:
        # X-error
        with case(7):
            circ.x(q_in[7])
        with case(1):
            circ.x(q_in[8])
        with case(2):
            circ.x(q_in[9])
        with case(4):
            circ.x(q_in[10]) 
            
    circ.cx(q_in[7], q_in[0])
    circ.cx(q_in[8], q_in[2])
    circ.cx(q_in[9], q_in[4])
    circ.cx(q_in[10], q_in[6])
    circ.append(syn_measure,q_in[7:11],[syn_x[2]])
    
    # Match Decoding process
    with circ.switch(syn_z) as case:
        # X-error
        with case(7):
            circ.x(q_in[0])
        with case(3):
            circ.x(q_in[1])
        with case(5):
            circ.x(q_in[2])
        with case(1):
            circ.x(q_in[3])            
        with case(6):
            circ.x(q_in[4])
        with case(2):
            circ.x(q_in[5])
        with case(4):
            circ.x(q_in[6])
        with case(case.DEFAULT):
            circ.id(q_in[0])   

    with circ.switch(syn_x) as case:
        # Z-error
        with case(7):
            circ.z(q_in[0])
        with case(3):
            circ.z(q_in[1])
        with case(5):
            circ.z(q_in[2])
        with case(1):
            circ.z(q_in[3])            
        with case(6):
            circ.z(q_in[4])
        with case(2):
            circ.z(q_in[5])
        with case(4):
            circ.z(q_in[6])
        with case(case.DEFAULT):
            circ.id(q_in[0])                   
      
    circ.measure(q_in[0:7], final)
    
    fidelity = []
    # 单轮纠错，开始执行电路
    for err in err_list:
        noise_model, _ = depo_err_model(err,1)
        counts = execute(circ, Aer.get_backend('aer_simulator'), 
                      noise_model=noise_model,shots = 8192).result().get_counts()
        # 
        output_s = count_leftmost_bits_percentage(counts) 
        ideal_s = ideal_output()
        res = np.inner(output_s, ideal_s)**2 
        fidelity.append(res)
        
    return fidelity

def syndrome_measure_DQNN_cat(err_list):
    
    q_in = QuantumRegister(13)
    temp = ClassicalRegister(30)
    syn1 = ClassicalRegister(4)
    syn2 = ClassicalRegister(4)
    syn3 = ClassicalRegister(4)
    syn4 = ClassicalRegister(4)
    syn5 = ClassicalRegister(4)
    syn6 = ClassicalRegister(4)
    syn_x = ClassicalRegister(3)
    syn_z = ClassicalRegister(3)
    final = ClassicalRegister(7)
    circ = QuantumCircuit(q_in, temp, syn1, syn2, syn3, syn4, syn5, syn6, 
                          syn_x, syn_z, final, name='syndrome_cat')
    Encoding_circ = Encoding()
    anci_prep = DQNN_cat_state()
    
    # 编码过程
    circ.append(Encoding_circ,q_in[0:7])
    
    # 稳定子测量
    # Z(0,1,2,3)
    circ.append(anci_prep,q_in[7:12],temp[0:5])
    circ.cz(q_in[7], q_in[0])
    circ.cz(q_in[8], q_in[1])
    circ.cz(q_in[9], q_in[2])
    circ.cz(q_in[10], q_in[3])
    circ.measure(q_in[7:11], syn1)
    for i in [7,8,9,10,11]:
        circ.reset(q_in[i])
    
    # Z(0,1,4,5)
    circ.append(anci_prep,q_in[7:12],temp[5:10])
    circ.cz(q_in[7], q_in[0])
    circ.cz(q_in[8], q_in[1])
    circ.cz(q_in[9], q_in[4])
    circ.cz(q_in[10], q_in[5])
    circ.measure(q_in[7:11], syn2)
    for i in [7,8,9,10,11]:
        circ.reset(q_in[i])
        
    # Z(0,2,4,6)
    circ.append(anci_prep,q_in[7:12],temp[10:15])
    circ.cz(q_in[7], q_in[0])
    circ.cz(q_in[8], q_in[2])
    circ.cz(q_in[9], q_in[4])
    circ.cz(q_in[10], q_in[6])
    circ.measure(q_in[7:11], syn3)
    for i in [7,8,9,10,11]:
        circ.reset(q_in[i])
        
    # X(0,1,2,3)
    circ.append(anci_prep,q_in[7:12],temp[15:20])
    circ.cx(q_in[7], q_in[0])
    circ.cx(q_in[8], q_in[1])
    circ.cx(q_in[9], q_in[2])
    circ.cx(q_in[10], q_in[3])
    circ.measure(q_in[7:11], syn4)
    for i in [7,8,9,10,11]:
        circ.reset(q_in[i])
    
    # X(0,1,4,5)
    circ.append(anci_prep,q_in[7:12],temp[20:25])
    circ.cx(q_in[7], q_in[0])
    circ.cx(q_in[8], q_in[1])
    circ.cx(q_in[9], q_in[4])
    circ.cx(q_in[10], q_in[5])
    circ.measure(q_in[7:11], syn5)
    for i in [7,8,9,10,11]:
        circ.reset(q_in[i])
        
    # X(0,2,4,6)
    circ.append(anci_prep,q_in[7:12],temp[25:30])
    circ.cx(q_in[7], q_in[0])
    circ.cx(q_in[8], q_in[2])
    circ.cx(q_in[9], q_in[4])
    circ.cx(q_in[10], q_in[6])
    circ.measure(q_in[7:11], syn6)
    for i in [7,8,9,10,11]:
        circ.reset(q_in[i])
   
    # 记录测量结果
    with circ.switch(syn1) as case:
        with case(1,2,4,8):
            circ.x(q_in[7])
        with case(7,11,13,14):
            circ.x(q_in[7])
        with case(case.DEFAULT):
            circ.id(q_in[7])

    with circ.switch(syn2) as case:
        with case(1,2,4,8):
            circ.x(q_in[8])
        with case(7,11,13,14):
            circ.x(q_in[8])
        with case(case.DEFAULT):
            circ.id(q_in[8])

    with circ.switch(syn3) as case:
        with case(1,2,4,8):
            circ.x(q_in[9])
        with case(7,11,13,14):
            circ.x(q_in[9])
        with case(case.DEFAULT):
            circ.id(q_in[9])

    with circ.switch(syn4) as case:
        with case(1,2,4,8):
            circ.x(q_in[10])
        with case(7,11,13,14):
            circ.x(q_in[10])
        with case(case.DEFAULT):
            circ.id(q_in[10])  

    with circ.switch(syn5) as case:
        with case(1,2,4,8):
            circ.x(q_in[11])
        with case(7,11,13,14):
            circ.x(q_in[11])
        with case(case.DEFAULT):
            circ.id(q_in[11]) 

    with circ.switch(syn6) as case:
        with case(1,2,4,8):
            circ.x(q_in[12])
        with case(7,11,13,14):
            circ.x(q_in[12])
        with case(case.DEFAULT):
            circ.id(q_in[12]) 
            
    circ.measure([q_in[7],q_in[8],q_in[9]], syn_z)
    circ.measure([q_in[10],q_in[11],q_in[12]], syn_x)
    
    # Match Decoding process
    with circ.switch(syn_z) as case:
        # X-error
        with case(7):
            circ.x(q_in[0])
        with case(3):
            circ.x(q_in[1])
        with case(5):
            circ.x(q_in[2])
        with case(1):
            circ.x(q_in[3])            
        with case(6):
            circ.x(q_in[4])
        with case(2):
            circ.x(q_in[5])
        with case(4):
            circ.x(q_in[6])
        with case(case.DEFAULT):
            circ.id(q_in[0])   

    with circ.switch(syn_x) as case:
        # Z-error
        with case(7):
            circ.z(q_in[0])
        with case(3):
            circ.z(q_in[1])
        with case(5):
            circ.z(q_in[2])
        with case(1):
            circ.z(q_in[3])            
        with case(6):
            circ.z(q_in[4])
        with case(2):
            circ.z(q_in[5])
        with case(4):
            circ.z(q_in[6])
        with case(case.DEFAULT):
            circ.id(q_in[0])                    
      
    circ.measure([q_in[0],q_in[1],q_in[2],q_in[3],q_in[4],q_in[5],q_in[6]], final)
    
    fidelity = []
    # 单轮纠错，开始执行电路
    for err in err_list:
        noise_model, _ = depo_err_model(err,1)
        counts = execute(circ, Aer.get_backend('aer_simulator'), 
                      noise_model=noise_model,shots = 8192).result().get_counts()
        # 
        output_s = count_leftmost_bits_percentage(counts) 
        ideal_s = ideal_output()
        res = np.inner(output_s, ideal_s)**2 
        fidelity.append(res)
    return fidelity
   

err_list = np.linspace(0.0001, 0.001, 8)
'''
fidelity1 = syndrome_measure_cat(err_list)
fidelity2 = syndrome_measure_DQNN_cat(err_list) 
'''
fidelity1 = [0.9920231367164541, 0.9813029290097871, 0.9724227514952116, 0.9625930076675666, 0.9567934956137183, 0.9447582285884104, 0.9328360555756695, 0.9252380124217345]
fidelity2 = [0.9933093144005001, 0.9869183353518768, 0.9808149109670157, 0.9762987990142382, 0.9710230040846868, 0.9668332806746024, 0.9594199976146188, 0.9535291801546748]
plt.plot(err_list, fidelity1, color="red", linestyle="dashed", linewidth=2, marker="o", label="normal")

plt.plot(err_list, fidelity2, color="blue", linestyle="solid", linewidth=2, marker="o", label="regular")

# 显示legend，位置为右上角
plt.legend(loc=3)

plt.xlabel("physical noise rate")
plt.ylabel("fidelity")
plt.title("7-qubit basic encoding state prep")
plt.show()