# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:44:20 2023

@author: q999s
"""
import math
import pickle
import random
import numpy as np
from qutip import basis, tensor
from qutip import Qobj
from qutip_qip.circuit import QubitCircuit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit import transpile
from tools import count_leftmost_bits_percentage

def Verify_Data():
    # 生成一组可以供验证的数据集
    input_s = []
    for i in range(4):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0,1)
        qc.cx(0,2)
        qc.cx(0,3) 
        qc.x(i)
        
        psi = Statevector.from_label('0000')
        temp4 = psi.evolve(qc)
        probs = temp4.probabilities()
        s = [np.abs(math.sqrt(item)) for item in probs]
        input_s.append(s)
        
    data=[]
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0,1)
    qc.cx(0,2)
    qc.cx(0,3)  
    
    temp1 = Statevector.from_label('0000')
    temp2 = temp1.evolve(qc)
    temp3 = temp2.probabilities()
    # 这里必须返回的是状态向量
    output_s = [np.abs(math.sqrt(item)) for item in temp3]
    
    for item in input_s: 
        data.append([item, output_s])
        
    psi = Statevector.from_label('0000')
    probs = psi.probabilities()
    s = [np.abs(math.sqrt(item)) for item in probs]
    data.append([s, output_s])
    # 返回输入数据状态向量与输出数据状态向量构成的组合
    return data 

def noise_train_data_generation():
    # 针对[4,1,4]网络，生成4-qubit cat state有噪声训练数据
    matrix_list = np.load("matrix_list.npz")
    U = list(matrix_list.values())
        
    size = 32
    Initial_input_data = []
    
    Final_output = []
    
    for i in range(5):
        q_in = QuantumRegister(4)
        
        qc = QuantumCircuit(q_in)
        qc.h(0)
        qc.cx(0,1)
        qc.cx(0,2)
        qc.cx(0,3) 
        if i<4:  
            qc.x(i)
        
        psi = Statevector.from_label('0000')
        psi_out = psi.evolve(qc)
        psi = [np.abs(math.sqrt(item)) for item in psi_out.probabilities()]
        Initial_input_data.append(psi)
        
    error = ['x','z','y']
    for item in Initial_input_data:
        for i in range(len(U)):
            temp_u = np.identity(size)
            for j in range(i):
                temp_u = U[j]@temp_u
            for k in range(4):
                for e in error:
                    q = QuantumRegister(4)
                    q_hidden = QuantumRegister(1)
                    c1 = ClassicalRegister(1)
                    c2 = ClassicalRegister(4)
                    circ = QuantumCircuit(q, c1, c2)
                    circ.initialize(item, q)
                    circ.add_register(q_hidden)
                    circ.unitary(temp_u, [q_hidden[0],q[0],q[1],q[2],q[3]])
                    if e == 'x':
                        circ.x(q[k])
                    elif e == 'z':
                        circ.z(q[k])
                    else:
                        circ.y(q[k])
                    circ.unitary(np.conjugate(temp_u).T,[q_hidden[0], q[0],q[1],q[2],q[3]])
                    circ.measure(q_hidden[0], c1)
                    circ.measure(q, c2)
                    
                    backend = QasmSimulatorPy()
                    # backend = StatevectorSimulator()
                    job = backend.run(transpile(circ, backend),shots = 1024)
                    counts = job.result().get_counts()
                    output_data = count_leftmost_bits_percentage(counts)
                    Final_output.append(output_data)
    
    s = tensor(basis(2, 0), basis(2, 0), basis(2, 0), basis(2, 0))    
    for i in range(4): 
        qc = QubitCircuit(4)
        qc.add_gate("H", 0)
        qc.add_gate("CNOT", 1, 0)
        qc.add_gate("CNOT", 2, 0)
        qc.add_gate("CNOT", 3, 0)          
        
        model_out = qc.run(s)
    
    trainingData=[]
    for item in Final_output:
        temp = [[x] for x in item]
        trainingData.append([Qobj(temp, dims=[[2, 2, 2, 2], [1, 1, 1, 1]]),model_out])
        
    with open("Noise_Training_Data.pkl", "wb") as f:
        pickle.dump(trainingData, f)
        
def Three_qubit_code_traing_data1():
    trainingData=[]
    s_0 = tensor(basis(2, 0), basis(2, 0), basis(2, 0), basis(2, 0), basis(2, 0))
    l1 = []
    l1out = []
    for i in range(3): 
        qc = QubitCircuit(5)
        qc.add_gate("CNOT", 1, 0)
        qc.add_gate("CNOT", 2, 0)
        
        qc.add_gate("X", i)
        
        qc.add_gate("H", 3)
        qc.add_gate("CZ", 0, 3)
        qc.add_gate("CZ", 1, 3) 
        qc.add_gate("H", 3)
        
        qc.add_gate("H", 4)
        qc.add_gate("CZ", 1, 4)
        qc.add_gate("CZ", 2, 4)
        qc.add_gate("H", 4)
        
        input_s = qc.run(s_0)
        l1.append(input_s)
    
    for i in range(3): 
        qc = QubitCircuit(5)
        qc.add_gate("CNOT", 1, 0)
        qc.add_gate("CNOT", 2, 0)
        
        qc.add_gate("H", 3)
        qc.add_gate("CZ", 0, 3)
        qc.add_gate("CZ", 1, 3) 
        qc.add_gate("H", 3)
        
        qc.add_gate("H", 4)
        qc.add_gate("CZ", 1, 4)
        qc.add_gate("CZ", 2, 4)
        qc.add_gate("H", 4)
        
        input_s = qc.run(s_0)
        l1out.append(input_s)
    
    for i in range(3): 
        trainingData.append([l1[i],l1out[i]])
        
    s_1 = tensor(basis(2, 1), basis(2, 0), basis(2, 0), basis(2, 0), basis(2, 0))
    l2 = []
    l2out = []
    for i in range(3): 
        qc = QubitCircuit(5)
        qc.add_gate("CNOT", 1, 0)
        qc.add_gate("CNOT", 2, 0)
        
        qc.add_gate("X", i)
        
        qc.add_gate("H", 3)
        qc.add_gate("CZ", 0, 3)
        qc.add_gate("CZ", 1, 3) 
        qc.add_gate("H", 3)
        
        qc.add_gate("H", 4)
        qc.add_gate("CZ", 1, 4)
        qc.add_gate("CZ", 2, 4)
        qc.add_gate("H", 4)
        
        input_s = qc.run(s_1)
        l2.append(input_s)
    
    for i in range(3): 
        qc = QubitCircuit(5)
        qc.add_gate("CNOT", 1, 0)
        qc.add_gate("CNOT", 2, 0)
        
        qc.add_gate("H", 3)
        qc.add_gate("CZ", 0, 3)
        qc.add_gate("CZ", 1, 3) 
        qc.add_gate("H", 3)
        
        qc.add_gate("H", 4)
        qc.add_gate("CZ", 1, 4)
        qc.add_gate("CZ", 2, 4)
        qc.add_gate("H", 4)
        
        input_s = qc.run(s_1)
        l2out.append(input_s)
    
    for i in range(3): 
        trainingData.append([l2[i],l2out[i]])
        
    return trainingData 

def Three_qubit_code_traing_data2():
    trainingData=[]
    s_0 = tensor(basis(2, 0), basis(2, 0), basis(2, 0))
    l1 = []
    l1out = []
    for i in range(3): 
        qc = QubitCircuit(3)
        qc.add_gate("CNOT", 1, 0)
        qc.add_gate("CNOT", 2, 0)
        
        qc.add_gate("X", i)
        
        input_s = qc.run(s_0)
        l1.append(input_s)
    
    for i in range(3): 
        qc = QubitCircuit(3)
        qc.add_gate("CNOT", 1, 0)
        qc.add_gate("CNOT", 2, 0)           
        
        input_s = qc.run(s_0)
        l1out.append(input_s)
    
    for i in range(3): 
        trainingData.append([l1[i],l1out[i]])
        
    s_1 = tensor(basis(2, 1), basis(2, 0), basis(2, 0))
    l2 = []
    l2out = []
    for i in range(3): 
        qc = QubitCircuit(3)
        qc.add_gate("CNOT", 1, 0)
        qc.add_gate("CNOT", 2, 0)
        
        qc.add_gate("X", i)
        
        input_s = qc.run(s_1)
        l2.append(input_s)
    
    for i in range(3): 
        qc = QubitCircuit(3)
        qc.add_gate("CNOT", 1, 0)
        qc.add_gate("CNOT", 2, 0)
        
        input_s = qc.run(s_1)
        l2out.append(input_s)
    
    for i in range(3): 
        trainingData.append([l2[i],l2out[i]])
        
    return trainingData 

def Seven_qubit_code_traing_data1():
    trainingData=[]
    s = tensor(basis(2, 0), basis(2, 0), basis(2, 0),basis(2, 0), basis(2, 0), basis(2, 0),basis(2, 0))
    l0 = []
    for i in range(7): 
        qc = QubitCircuit(7)
        qc.add_gate('H', 0)
        qc.add_gate('H', 1)
        qc.add_gate('H', 3)
            
        qc.add_gate('CNOT', 6,0)
        qc.add_gate('CNOT', 2,1)
        qc.add_gate('CNOT', 5,3)
            
        qc.add_gate('CNOT', 4,0)
        qc.add_gate('CNOT', 5,1)
        qc.add_gate('CNOT', 6,3)
            
        qc.add_gate('CNOT', 2,0)
        qc.add_gate('CNOT', 4,3)
        qc.add_gate('CNOT', 6,1) 
        
        qc.add_gate("X", i)
        
        input_s = qc.run(s)
        l0.append(input_s)
    
    qc1 = QubitCircuit(7)
    qc1.add_gate('H', 0)
    qc1.add_gate('H', 1)
    qc1.add_gate('H', 3)
        
    qc1.add_gate('CNOT', 6,0)
    qc1.add_gate('CNOT', 2,1)
    qc1.add_gate('CNOT', 5,3)
        
    qc1.add_gate('CNOT', 4,0)
    qc1.add_gate('CNOT', 5,1)
    qc1.add_gate('CNOT', 6,3)
        
    qc1.add_gate('CNOT', 2,0)
    qc1.add_gate('CNOT', 4,3)
    qc1.add_gate('CNOT', 6,1) 
    input_s0 = qc1.run(s)

    for i in range(7): 
        trainingData.append([l0[i],input_s0])
        
    l1 = []
    for i in range(7): 
        qc = QubitCircuit(7)
        qc.add_gate('H', 0)
        qc.add_gate('H', 1)
        qc.add_gate('H', 3)
            
        qc.add_gate('CNOT', 6,0)
        qc.add_gate('CNOT', 2,1)
        qc.add_gate('CNOT', 5,3)
            
        qc.add_gate('CNOT', 4,0)
        qc.add_gate('CNOT', 5,1)
        qc.add_gate('CNOT', 6,3)
            
        qc.add_gate('CNOT', 2,0)
        qc.add_gate('CNOT', 4,3)
        qc.add_gate('CNOT', 6,1) 
        for j in range(7):
            qc.add_gate("X", j)
                
        qc.add_gate("X", i)
        
        input_s = qc.run(s)
        l1.append(input_s)
    
    qc2 = QubitCircuit(7)
    qc2.add_gate('H', 0)
    qc2.add_gate('H', 1)
    qc2.add_gate('H', 3)
        
    qc2.add_gate('CNOT', 6,0)
    qc2.add_gate('CNOT', 2,1)
    qc2.add_gate('CNOT', 5,3)
        
    qc2.add_gate('CNOT', 4,0)
    qc2.add_gate('CNOT', 5,1)
    qc2.add_gate('CNOT', 6,3)
        
    qc2.add_gate('CNOT', 2,0)
    qc2.add_gate('CNOT', 4,3)
    qc2.add_gate('CNOT', 6,1) 
    for j in range(7):
        qc2.add_gate("X", j)
        
    input_s1 = qc2.run(s)

    for i in range(7): 
        trainingData.append([l1[i],input_s1])
        
    return trainingData 

def Seven_qubit_code_traing_data2():
    trainingData=[]
    s0 = tensor(basis(2, 0), basis(2, 0), basis(2, 0),basis(2, 0), basis(2, 0), basis(2, 0),basis(2, 0))
    s1 = tensor(basis(2, 1), basis(2, 0), basis(2, 0),basis(2, 0), basis(2, 0), basis(2, 0),basis(2, 0))
    
    qc = QubitCircuit(7)
    qc.add_gate('H', 0)
    qc.add_gate('H', 1)
    qc.add_gate('H', 3)
        
    qc.add_gate('CNOT', 6,0)
    qc.add_gate('CNOT', 2,1)
    qc.add_gate('CNOT', 5,3)
        
    qc.add_gate('CNOT', 4,0)
    qc.add_gate('CNOT', 5,1)
    qc.add_gate('CNOT', 6,3)
        
    qc.add_gate('CNOT', 2,0)
    qc.add_gate('CNOT', 4,3)
    qc.add_gate('CNOT', 6,1) 
    
    input_s = qc.run(s0)
    trainingData.append([s0,input_s])

    qc = QubitCircuit(7)
    qc.add_gate('H', 0)
    qc.add_gate('H', 1)
    qc.add_gate('H', 3)
        
    qc.add_gate('CNOT', 6,0)
    qc.add_gate('CNOT', 2,1)
    qc.add_gate('CNOT', 5,3)
        
    qc.add_gate('CNOT', 4,0)
    qc.add_gate('CNOT', 5,1)
    qc.add_gate('CNOT', 6,3)
        
    qc.add_gate('CNOT', 2,0)
    qc.add_gate('CNOT', 4,3)
    qc.add_gate('CNOT', 6,1) 
    
    for j in range(7):
        qc.add_gate("X", j)

    input_s = qc.run(s0)
    trainingData.append([s1,input_s])            
 
    qc = QubitCircuit(7)
    qc.add_gate('H', 0)
        
    s_plus = qc.run(s0)
    
    qc = QubitCircuit(7)
    qc.add_gate('H', 0)
    qc.add_gate('H', 1)
    qc.add_gate('H', 3)
        
    qc.add_gate('CNOT', 6,0)
    qc.add_gate('CNOT', 2,1)
    qc.add_gate('CNOT', 5,3)
        
    qc.add_gate('CNOT', 4,0)
    qc.add_gate('CNOT', 5,1)
    qc.add_gate('CNOT', 6,3)
        
    qc.add_gate('CNOT', 2,0)
    qc.add_gate('CNOT', 4,3)
    qc.add_gate('CNOT', 6,1) 
    
    for j in range(7):
        qc.add_gate("H", j)
               
    input_s = qc.run(s0)
    trainingData.append([s_plus,input_s])   
        
    return trainingData 

def Cat_state_data1():
    trainingData=[]
    s = tensor(basis(2, 0), basis(2, 0), basis(2, 0), basis(2, 0))
    l = []
    l_out = []
    for i in range(4): 
        qc = QubitCircuit(4)
        qc.add_gate("H", 0)
        qc.add_gate("CNOT", 1, 0)
        qc.add_gate("CNOT", 2, 0)
        qc.add_gate("CNOT", 3, 0)
        qc.add_gate("X", i)
        
        input_s = qc.run(s)
        l.append(input_s)
    
    for i in range(4): 
        qc = QubitCircuit(4)
        qc.add_gate("H", 0)
        qc.add_gate("CNOT", 1, 0)
        qc.add_gate("CNOT", 2, 0)
        qc.add_gate("CNOT", 3, 0)          
        
        input_s = qc.run(s)
        l_out.append(input_s)
    
    for i in range(4): 
        trainingData.append([l[i],l_out[i]])
    
    trainingData.append([s,l_out[i]])# 理想情形下的输入输出
    
    return trainingData 

def Cat_state_data2():

    with open("Noise_Training_Data.pkl", "rb") as f:
        trainingData = pickle.load(f)
    
    '''
    sample_train = []
    
    # 对于每一类，从中随机选择 10 个元素，
    for i in range(5):
        # 计算每一类的起始和结束索引
        start = i * 60
        end = (i + 1) * 60
        # 从 A 的子列表中随机抽取 10 个元素
        sample = random.sample(trainingData[start:end], 10)
        # 将抽取的元素扩展到 B 中
        sample_train.extend(sample)
    '''
    return trainingData
        
def syndrome_data1():
    #Three qubit code
    #3-2-3
    trainingData=[]
    s = tensor(basis(2, 0), basis(2, 0), basis(2, 0))
    l = []
    l_out = []

    # syndrome_input
    qc = QubitCircuit(3)
    qc.add_gate("X", 0)
    input_s = qc.run(s)
    l.append(input_s)
    
    qc = QubitCircuit(3)
    qc.add_gate("X", 0)
    qc.add_gate("X", 1)
    input_s = qc.run(s)
    l.append(input_s)
    
    qc = QubitCircuit(3)
    qc.add_gate("X", 1)
    input_s = qc.run(s)
    l.append(input_s)

    # syndrome_output
    qc = QubitCircuit(3)
    qc.add_gate("X", 2)
    input_s = qc.run(s)
    l_out.append(input_s)
    
    qc = QubitCircuit(3)
    qc.add_gate("X", 1)
    input_s = qc.run(s)
    l_out.append(input_s)
    
    qc = QubitCircuit(3)
    qc.add_gate("X", 0)
    input_s = qc.run(s)
    l_out.append(input_s)

    for i in range(3): 
        trainingData.append([l[i],l_out[i]])
        
    trainingData.append([s,s])
    
    return trainingData 
        
def syndrome_data2():
    #Seven qubit code
    #
    trainingData=[]
    s = tensor(basis(2, 0), basis(2, 0), basis(2, 0),basis(2, 0), basis(2, 0), basis(2, 0), basis(2, 0))
    l = []
    l_out = []

    # syndrome_input
    qc = QubitCircuit(7)
    qc.add_gate("X", 0)
    input_s = qc.run(s)
    l.append(input_s)
    
    qc = QubitCircuit(7)
    qc.add_gate("X", 1)
    input_s = qc.run(s)
    l.append(input_s)
    
    qc = QubitCircuit(7)
    qc.add_gate("X", 0)
    qc.add_gate("X", 1)
    input_s = qc.run(s)
    l.append(input_s)
    
    qc = QubitCircuit(7)
    qc.add_gate("X", 2)
    input_s = qc.run(s)
    l.append(input_s)
    
    qc = QubitCircuit(7)
    qc.add_gate("X", 0)
    qc.add_gate("X", 2)
    input_s = qc.run(s)
    l.append(input_s)

    qc = QubitCircuit(7)
    qc.add_gate("X", 1)
    qc.add_gate("X", 2)
    input_s = qc.run(s)
    l.append(input_s)
    
    qc = QubitCircuit(7)
    qc.add_gate("X", 0)
    qc.add_gate("X", 1)
    qc.add_gate("X", 2)
    input_s = qc.run(s)
    l.append(input_s)
    
    # syndrome_output
    qc = QubitCircuit(7)
    qc.add_gate("X", 0)
    input_s = qc.run(s)
    l_out.append(input_s)
    
    qc = QubitCircuit(7)
    qc.add_gate("X", 1)
    input_s = qc.run(s)
    l_out.append(input_s)
    
    qc = QubitCircuit(7)
    qc.add_gate("X", 2)
    input_s = qc.run(s)
    l_out.append(input_s)
    
    qc = QubitCircuit(7)
    qc.add_gate("X", 3)
    input_s = qc.run(s)
    l_out.append(input_s)
    
    qc = QubitCircuit(7)
    qc.add_gate("X", 4)
    input_s = qc.run(s)
    l_out.append(input_s)
    
    qc = QubitCircuit(7)
    qc.add_gate("X", 5)
    input_s = qc.run(s)
    l_out.append(input_s)
    
    qc = QubitCircuit(7)
    qc.add_gate("X", 6)
    input_s = qc.run(s)
    l_out.append(input_s)
    
    for i in range(7): 
        trainingData.append([l[i],l_out[i]])
    
    trainingData.append([s,s])
    
    return trainingData 