# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:55:03 2024

@author: q999s
"""

'''
Exam11:
    后端：aer_simulator模拟
    噪声：退极化噪声模型(包括门及测量)
    电路：
        1. (5-qubit 纠错码制备及纠错过程) + ([4,1,4]有噪声训练 DQNN辅助态制备) 
             + (已训练的RealAmplitudes regular term)
        2. (5-qubit 纠错码制备及纠错过程) + 有纠错cat state制备电路
    目的：相比传统带纠正意义的cat state，体现DQNN辅助态的优势（保真度，逻辑错误率）
    输出：实验图片
    结果分析：在这种公平环境下，给出的结果分析具有明显差异性
    备注：五量子比特纠错码编码电路及纠错码来自于Stac软件包生成
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
    q_in = QuantumRegister(5)
    circ = QuantumCircuit(q_in, name='Encoding')
    
    circ.h(q_in[0])
    circ.cz(q_in[0],q_in[1])
    circ.cz(q_in[0],q_in[3])
    circ.cx(q_in[0],q_in[4])
    circ.cz(q_in[0],q_in[4])
    
    circ.h(q_in[1])
    circ.cz(q_in[1],q_in[2])
    circ.cz(q_in[1],q_in[3])
    circ.cx(q_in[1],q_in[4])    
    
    circ.h(q_in[2])
    circ.cz(q_in[2],q_in[0])
    circ.cz(q_in[2],q_in[1])
    circ.cx(q_in[2],q_in[4]) 
    
    circ.h(q_in[3])
    circ.cz(q_in[3],q_in[0])
    circ.cz(q_in[3],q_in[2])
    circ.cx(q_in[3],q_in[4])  
    circ.cz(q_in[3],q_in[4])
    
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
    s_in = Statevector.from_label('00000')
    Encoding_circ = Encoding()
    
    q_in = QuantumRegister(5)
    circ = QuantumCircuit(q_in)
    circ.append(Encoding_circ, q_in) 
    
    s_out = s_in.evolve(circ).probabilities()
    target_s = [np.abs(math.sqrt(item)) for item in s_out]
    
    return target_s
      
def syndrome_measure_cat(err_list):
    
    q_in = QuantumRegister(12)
    temp1 = ClassicalRegister(3)
    temp2 = ClassicalRegister(3)
    temp3 = ClassicalRegister(3)
    temp4 = ClassicalRegister(3)
    
    syn = ClassicalRegister(4)
    
    final = ClassicalRegister(5)
    circ = QuantumCircuit(q_in, temp1, temp2, temp3, temp4,
                          syn, final, name='syndrome_cat')
    Encoding_circ = Encoding()
    syn_measure = cat_state_measure()
    # 编码过程
    circ.append(Encoding_circ,q_in[0:5])
    
    # 稳定子测量
    # X0Z1Z2X3
    circ.h(q_in[5])
    circ.cx(q_in[5],q_in[6])
    circ.cx(q_in[5],q_in[7])
    circ.cx(q_in[5],q_in[8]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(q_in[5],q_in[9])
    circ.cx(q_in[6],q_in[9])
    
    circ.cx(q_in[5],q_in[10])
    circ.cx(q_in[7],q_in[10])
    
    circ.cx(q_in[5],q_in[11])
    circ.cx(q_in[8],q_in[11])
    
    circ.measure([q_in[9],q_in[10],q_in[11]], temp1)
    for i in [9,10,11]:
        circ.reset(q_in[i])
        
    with circ.switch(temp1) as case:
        # X-error
        with case(7):
            circ.x(q_in[5])
        with case(1):
            circ.x(q_in[6])
        with case(2):
            circ.x(q_in[7])
        with case(4):
            circ.x(q_in[8]) 
            
    circ.cx(q_in[5], q_in[0])
    circ.cz(q_in[6], q_in[1])
    circ.cz(q_in[7], q_in[2])
    circ.cx(q_in[8], q_in[3])
    circ.append(syn_measure,q_in[5:9],[syn[0]])
    
    # X1Z2Z3X4
    circ.h(q_in[5])
    circ.cx(q_in[5],q_in[6])
    circ.cx(q_in[5],q_in[7])
    circ.cx(q_in[5],q_in[8]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(q_in[5],q_in[9])
    circ.cx(q_in[6],q_in[9])
    
    circ.cx(q_in[5],q_in[10])
    circ.cx(q_in[7],q_in[10])
    
    circ.cx(q_in[5],q_in[11])
    circ.cx(q_in[8],q_in[11])
    
    circ.measure([q_in[9],q_in[10],q_in[11]], temp2)
    for i in [9,10,11]:
        circ.reset(q_in[i])
        
    with circ.switch(temp1) as case:
        # X-error
        with case(7):
            circ.x(q_in[5])
        with case(1):
            circ.x(q_in[6])
        with case(2):
            circ.x(q_in[7])
        with case(4):
            circ.x(q_in[8]) 
            
    circ.cx(q_in[5], q_in[1])
    circ.cz(q_in[6], q_in[2])
    circ.cz(q_in[7], q_in[3])
    circ.cx(q_in[8], q_in[4])
    circ.append(syn_measure,q_in[5:9],[syn[1]])

    # X0X2Z3Z4
    circ.h(q_in[5])
    circ.cx(q_in[5],q_in[6])
    circ.cx(q_in[5],q_in[7])
    circ.cx(q_in[5],q_in[8]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(q_in[5],q_in[9])
    circ.cx(q_in[6],q_in[9])
    
    circ.cx(q_in[5],q_in[10])
    circ.cx(q_in[7],q_in[10])
    
    circ.cx(q_in[5],q_in[11])
    circ.cx(q_in[8],q_in[11])
    
    circ.measure([q_in[9],q_in[10],q_in[11]], temp3)
    for i in [9,10,11]:
        circ.reset(q_in[i])
        
    with circ.switch(temp1) as case:
        # X-error
        with case(7):
            circ.x(q_in[5])
        with case(1):
            circ.x(q_in[6])
        with case(2):
            circ.x(q_in[7])
        with case(4):
            circ.x(q_in[8]) 
            
    circ.cx(q_in[5], q_in[0])
    circ.cx(q_in[6], q_in[2])
    circ.cz(q_in[7], q_in[3])
    circ.cz(q_in[8], q_in[4])
    circ.append(syn_measure,q_in[5:9],[syn[2]])
    
    # Z0X1X3Z4
    circ.h(q_in[5])
    circ.cx(q_in[5],q_in[6])
    circ.cx(q_in[5],q_in[7])
    circ.cx(q_in[5],q_in[8]) 
    
    #对该cat使用非容错稳定子测量
    circ.cx(q_in[5],q_in[9])
    circ.cx(q_in[6],q_in[9])
    
    circ.cx(q_in[5],q_in[10])
    circ.cx(q_in[7],q_in[10])
    
    circ.cx(q_in[5],q_in[11])
    circ.cx(q_in[8],q_in[11])
    
    circ.measure([q_in[9],q_in[10],q_in[11]], temp4)
    for i in [9,10,11]:
        circ.reset(q_in[i])
        
    with circ.switch(temp1) as case:
        # X-error
        with case(7):
            circ.x(q_in[5])
        with case(1):
            circ.x(q_in[6])
        with case(2):
            circ.x(q_in[7])
        with case(4):
            circ.x(q_in[8]) 
            
    circ.cz(q_in[5], q_in[0])
    circ.cx(q_in[6], q_in[1])
    circ.cx(q_in[7], q_in[3])
    circ.cz(q_in[8], q_in[4])
    circ.append(syn_measure,q_in[5:9],[syn[3]])
    
    # Match Decoding process
    with circ.switch(syn) as case:
        # X-error
        with case(8):
            circ.x(q_in[0])
        with case(1):
            circ.x(q_in[1])
        with case(3):
            circ.x(q_in[2])
        with case(6):
            circ.x(q_in[3])            
        with case(12):
            circ.x(q_in[4])
        # Z-error
        with case(5):
            circ.z(q_in[0])
        with case(10):
            circ.z(q_in[1])
        with case(4):
            circ.z(q_in[2])
        with case(9):
            circ.z(q_in[3])            
        with case(2):
            circ.z(q_in[4])
        # Y-error
        with case(13):
            circ.z(q_in[0])
        with case(11):
            circ.z(q_in[1])
        with case(7):
            circ.z(q_in[2])
        with case(15):
            circ.z(q_in[3])            
        with case(14):
            circ.z(q_in[4])
        with case(case.DEFAULT):
            circ.id(q_in[0])   
                      
    circ.measure(q_in[0:5], final)
    
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
    
    q_in = QuantumRegister(11)
    temp = ClassicalRegister(20)
    syn1 = ClassicalRegister(4)
    syn2 = ClassicalRegister(4)
    syn3 = ClassicalRegister(4)
    syn4 = ClassicalRegister(4)
    syn = ClassicalRegister(4)
    final = ClassicalRegister(5)
    circ = QuantumCircuit(q_in, temp, syn1, syn2, syn3, syn4, 
                          syn, final, name='syndrome_cat')
    Encoding_circ = Encoding()
    anci_prep = DQNN_cat_state()
    
    # 编码过程
    circ.append(Encoding_circ,q_in[0:5])
    
    # 稳定子测量
    # X0Z1Z2X3
    circ.append(anci_prep,q_in[5:10],temp[0:5])
    circ.cx(q_in[5], q_in[0])
    circ.cz(q_in[6], q_in[1])
    circ.cz(q_in[7], q_in[2])
    circ.cx(q_in[8], q_in[3])
    circ.measure(q_in[5:9], syn1)
    for i in [5,6,7,8,9]:
        circ.reset(q_in[i])
    
    # X1Z2Z3X4
    circ.append(anci_prep,q_in[5:10],temp[5:10])
    circ.cx(q_in[5], q_in[1])
    circ.cz(q_in[6], q_in[2])
    circ.cz(q_in[7], q_in[3])
    circ.cx(q_in[8], q_in[4])
    circ.measure(q_in[5:9], syn1)
    for i in [5,6,7,8,9]:
        circ.reset(q_in[i])
        
    # X0X2Z3Z4
    circ.append(anci_prep,q_in[5:10],temp[10:15])
    circ.cx(q_in[5], q_in[0])
    circ.cx(q_in[6], q_in[2])
    circ.cz(q_in[7], q_in[3])
    circ.cz(q_in[8], q_in[4])
    circ.measure(q_in[5:9], syn1)
    for i in [5,6,7,8,9]:
        circ.reset(q_in[i])
        
    # Z0X1X3Z4
    circ.append(anci_prep,q_in[5:10],temp[15:20])
    circ.cz(q_in[5], q_in[0])
    circ.cx(q_in[6], q_in[1])
    circ.cx(q_in[7], q_in[3])
    circ.cz(q_in[8], q_in[4])
    circ.measure(q_in[5:9], syn1)
    for i in [5,6,7,8,9]:
        circ.reset(q_in[i])
   
    # 记录测量结果
    with circ.switch(syn1) as case:
        with case(1,2,4,8):
            circ.x(q_in[5])
        with case(7,11,13,14):
            circ.x(q_in[5])
        with case(case.DEFAULT):
            circ.id(q_in[5])

    with circ.switch(syn2) as case:
        with case(1,2,4,8):
            circ.x(q_in[6])
        with case(7,11,13,14):
            circ.x(q_in[6])
        with case(case.DEFAULT):
            circ.id(q_in[6])

    with circ.switch(syn3) as case:
        with case(1,2,4,8):
            circ.x(q_in[7])
        with case(7,11,13,14):
            circ.x(q_in[7])
        with case(case.DEFAULT):
            circ.id(q_in[7])

    with circ.switch(syn4) as case:
        with case(1,2,4,8):
            circ.x(q_in[8])
        with case(7,11,13,14):
            circ.x(q_in[8])
        with case(case.DEFAULT):
            circ.id(q_in[8])  
            
    circ.measure([q_in[5], q_in[6],q_in[7],q_in[8]], syn)
    
    # Match Decoding process
    with circ.switch(syn) as case:
        # X-error
        with case(8):
            circ.x(q_in[0])
        with case(1):
            circ.x(q_in[1])
        with case(3):
            circ.x(q_in[2])
        with case(6):
            circ.x(q_in[3])            
        with case(12):
            circ.x(q_in[4])
        # Z-error
        with case(5):
            circ.z(q_in[0])
        with case(10):
            circ.z(q_in[1])
        with case(4):
            circ.z(q_in[2])
        with case(9):
            circ.z(q_in[3])            
        with case(2):
            circ.z(q_in[4])
        # Y-error
        with case(13):
            circ.z(q_in[0])
        with case(11):
            circ.z(q_in[1])
        with case(7):
            circ.z(q_in[2])
        with case(15):
            circ.z(q_in[3])            
        with case(14):
            circ.z(q_in[4])
        with case(case.DEFAULT):
            circ.id(q_in[0])                    
      
    circ.measure(q_in[0:5], final)
    
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

fidelity1 = [0.9945789712572635, 0.9896294125671314, 0.9808754464307068, 0.9797856781728047, 0.9721488702425779, 0.9694909129897481, 0.9582583146742873, 0.9566149278003392]
fidelity2 = [0.9943761493205417, 0.992936258554547, 0.9896347447845483, 0.9844258881435289, 0.9829249801643681, 0.9810715170575233, 0.9796483083452208, 0.9755075577364722]

plt.plot(err_list, fidelity1, color="red", linestyle="dashed", linewidth=2, marker="o", label="normal")

plt.plot(err_list, fidelity2, color="blue", linestyle="solid", linewidth=2, marker="o", label="DQNN")

# 显示legend，位置为右上角
plt.legend(loc=3)

plt.xlabel("physical noise rate")
plt.ylabel("fidelity")
plt.title("5-qubit code basic encoding state prep")
plt.show()