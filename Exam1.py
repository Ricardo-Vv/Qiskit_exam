# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 17:51:05 2023

@author: q999s
"""
'''
Exam6:
    后端：QASM模拟
    噪声：退极化噪声模型
    电路：
    1. normal cat state prep with normal error correction
    2. (cat state prep) + ([4,1,4]有噪声训练 DQNN)
    3. (cat state prep) + ([4,1,4]有噪声训练 DQNN) 
        + (RealAmplitudes regular term)
    目的：制备4-qubit cat state
    输出：在该 DQNN 作为纠错过程计算保真度在多组噪声率下的变化，
            并对比两种不同策略
    结果分析：在限定必须具备纠错环节的条件下，DQNN显示出了对于噪声的抗性
            且明显优于 normal cat state prep with normal error correction
    备注：对于正则项的训练还有待提高
'''
import math
import numpy as np
from tools import count_leftmost_bits_percentage
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer
from qiskit.quantum_info import Statevector
from UserDefined_noise_model import depo_err_model
import matplotlib.pyplot as plt
from qiskit import execute
from qiskit.circuit.library import RealAmplitudes

def posteriori_error_correction(A):
    def bin_to_int(bin_str):
        return int(bin_str, 2)
    
    B = {}
    
    # 遍历字典A的每一个键值对
    for key, value in A.items():
        # 将键分割为左右两部分
        left, right = key.split()
        new_left = format(bin_to_int(left), '04b')
        if right == '001':
            # 翻转第二位比特
            temp = list(map(int, left))
            temp[2] = (temp[2]+1)%2
            new_left =  "".join(map(str, temp))
        elif right == '010':
            # 翻转第三位比特
            temp = list(map(int, left))
            temp[1] = (temp[1]+1)%2
            new_left =  "".join(map(str, temp))
        elif right == '100':
            # 翻转第四位比特
            temp = list(map(int, left))
            temp[0] = (temp[0]+1)%2
            new_left =  "".join(map(str, temp))
        elif right == '111':
            # 翻转第一位比特
            temp = list(map(int, left))
            temp[3] = (temp[3]+1)%2
            new_left =  "".join(map(str, temp))
        # 将修改后的左右两部分重新拼接为一个新的键
        new_key = new_left + ' ' + right
        B[new_key] = value
        
    return B
    
def Normal_cost(err_rate):
    s_in = Statevector.from_label('0000')
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0,1)
    qc.cx(0,2)
    qc.cx(0,3)   
    
    s_out = s_in.evolve(qc).probabilities()
    target_s = [np.abs(math.sqrt(item)) for item in s_out]
    
    # Noise simulation
    rounds = 100
    fidelity = []
    for err in err_rate:
        objective = []
        noise_model, _ = depo_err_model(err,1)
        for i in range(rounds):
            q_in = QuantumRegister(4)
            anci = QuantumRegister(3)
            c_syn = ClassicalRegister(3)
            c_out = ClassicalRegister(4)
            circ = QuantumCircuit(q_in,anci,c_syn,c_out)

            # 直接制备cat state
            circ.h(q_in[0])
            circ.cx(q_in[0],q_in[1])
            circ.cx(q_in[0],q_in[2])
            circ.cx(q_in[0],q_in[3])  
            
            circ.cx(q_in[0],anci[0])  
            circ.cx(q_in[1],anci[0])  
            
            circ.cx(q_in[0],anci[1])  
            circ.cx(q_in[2],anci[1]) 
            
            circ.cx(q_in[0],anci[2])  
            circ.cx(q_in[3],anci[2]) 
            
            circ.measure(anci, c_syn)
                
            circ.measure(q_in, c_out)
            
            counts = execute(circ, Aer.get_backend('qasm_simulator'), 
                             noise_model=noise_model, shots = 1024).result().get_counts()
            counts = posteriori_error_correction(counts)
            output_s = count_leftmost_bits_percentage(counts)  
            res = np.inner(output_s, target_s)**2 
            objective.append(res)
    
        fidelity.append(np.abs(sum(objective)/rounds))
        
    return fidelity

def Simulation_cost(weight_mat, err_rate, regular_add = False, params = None):
    s_in = Statevector.from_label('0000')
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0,1)
    qc.cx(0,2)
    qc.cx(0,3)     
    s_out = s_in.evolve(qc).probabilities()
    target_s = [np.abs(math.sqrt(item)) for item in s_out]
    
    if regular_add:
        # 给定正则项变分电路
        regular_term = RealAmplitudes(4, reps=1)
        parameters = {}
        for j, p in enumerate(regular_term.ordered_parameters):
            parameters[p] = params[j]
        regular_term.assign_parameters(parameters)
    # Noise simulation
    rounds = 100
    fidelity = []
    for err in err_rate:
        objective = []
        noise_model, _ = depo_err_model(err,1)
        for i in range(rounds):
            q_in = QuantumRegister(4)
            c_in = ClassicalRegister(4)
            c_hidden = ClassicalRegister(1)
            c_out = ClassicalRegister(4)
            circ = QuantumCircuit(q_in, c_in, c_hidden, c_out)
            # 第一层计算
            # 直接制备cat state
            circ.h(q_in[0])
            circ.cx(q_in[0],q_in[1])
            circ.cx(q_in[0],q_in[2])
            circ.cx(q_in[0],q_in[3])  
            # 初始化第一个隐藏层
            q_hidden = QuantumRegister(1)
            circ.add_register(q_hidden)
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
            if regular_add:
                # 施加变分线路
                circ.compose(regular_term, q_in)
            # 测量去除前一层
            circ.measure(q_hidden[0],c_hidden)
            circ.measure(q_in, c_out)
            
            counts = execute(circ, Aer.get_backend('qasm_simulator'), 
                             noise_model=noise_model, shots = 1024).result().get_counts()
        
            output_s = count_leftmost_bits_percentage(counts)  
            res = np.inner(output_s, target_s)**2 
            objective.append(res)
    
        fidelity.append(np.abs(sum(objective)/rounds))
        
    return fidelity

def repeat_prep(err_rate):
    s_in = Statevector.from_label('000000000000')
    qc = QuantumCircuit(12)
    qc.h(0)
    qc.cx(0,1)
    qc.cx(0,2)
    qc.cx(0,3)   
    
    qc.h(4)
    qc.cx(4,5)
    qc.cx(4,6)
    qc.cx(4,7)  
    
    qc.h(8)
    qc.cx(8,9)
    qc.cx(8,10)
    qc.cx(8,11)  
    
    s_out = s_in.evolve(qc).probabilities()
    target_s = [np.abs(math.sqrt(item)) for item in s_out]
    
    # Noise simulation
    rounds = 100
    fidelity = []
    for err in err_rate:
        objective = []
        noise_model, _ = depo_err_model(err,1)
        for i in range(rounds):
            q_in = QuantumRegister(12)
            c_out = ClassicalRegister(12)
            circ = QuantumCircuit(q_in,c_out)

            # 直接制备cat state
            circ.h(q_in[0])
            circ.cx(q_in[0],q_in[1])
            circ.cx(q_in[0],q_in[2])
            circ.cx(q_in[0],q_in[3])  
            
            circ.h(q_in[4])
            circ.cx(q_in[4],q_in[5])
            circ.cx(q_in[4],q_in[6])
            circ.cx(q_in[4],q_in[7]) 
            
            circ.h(q_in[8])
            circ.cx(q_in[8],q_in[9])
            circ.cx(q_in[8],q_in[10])
            circ.cx(q_in[8],q_in[11]) 
                
            circ.measure(q_in, c_out)
            
            counts = execute(circ, Aer.get_backend('qasm_simulator'), 
                             noise_model=noise_model, shots = 1024).result().get_counts()
            output_s = count_leftmost_bits_percentage(counts)  
            res = np.inner(output_s, target_s)**2 
            objective.append(res)
    
        fidelity.append(np.abs(sum(objective)/rounds))
        
    return fidelity
        
        
matrix_list = np.load("Noise_training_network.npz")
# load DQNN model
weight_mat = list(matrix_list.values())

err_rate = np.linspace(0.0001, 0.001, 8)
ansatz_params = [4.12117599,2.97884861,6.26545179,4.78701417,2.35295404,5.77750245,4.81764718,1.69858263]

fidelity1 = Normal_cost(err_rate)
fidelity2 = Simulation_cost(weight_mat, err_rate, regular_add = False, params = None)
fidelity3 = repeat_prep(err_rate)
'''
fidelity1 = [0.9988352405788781, 0.997487246065508, 0.9966392845501126, 0.9952450758976817, 0.9936292961427172, 0.9927679018588303, 0.9913707054264538, 0.9897156594388088]
fidelity2 = [0.9990193381775362, 0.9988236510364935, 0.9984714136093286, 0.9981498530810687, 0.9977543311514129, 0.9976211216664218, 0.997438421657968, 0.9968086024688467]
fidelity3 = [0.996925108892912, 0.9949185977926822, 0.9931961357002327, 0.9907025409701764, 0.9892634611726134, 0.9876137135263204, 0.9855178456294766, 0.9828696611977253]
'''  
plt.plot(err_rate, fidelity1, color="red", linestyle="dashed", linewidth=2, marker="o", label="normal")

plt.plot(err_rate, fidelity2, color="blue", linestyle="solid", linewidth=2, marker="o", label="DQNN")

plt.plot(err_rate, fidelity3, color="green", linestyle="solid", linewidth=2, marker="o", label="repeat_prep")

# 显示legend，位置为右上角
plt.legend(loc=3)

plt.xlabel("physical noise rate")
plt.ylabel("fidelity")
plt.title("4-qubit cat state prep")
plt.show()



