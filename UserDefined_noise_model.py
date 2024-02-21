# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:32:37 2023

@author: q999s
"""
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import depolarizing_error

def depo_err_model(p, qubits=1):
    # This creates the depolarizing error channel,
    # epsilon(P) = (1-P)rho + (P/3)(XrhoX + YrhoY + ZrhoZ).
    depo_err_chan = depolarizing_error(p, qubits)
    two_qubit_err_chan = depo_err_chan.tensor(depo_err_chan)
    # Creating the noise model to be used during execution.
    noise_model = NoiseModel()
    
    noise_model.add_all_qubit_quantum_error(depo_err_chan, ['u1', 'u2', 'u3', 'measure']) # measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(two_qubit_err_chan, ["cx","cz"])
    return noise_model, depo_err_chan

def depo_err_model_measure(p, qubits=1):
    # This creates the depolarizing error channel,
    # epsilon(P) = (1-P)rho + (P/3)(XrhoX + YrhoY + ZrhoZ).
    depo_err_chan = depolarizing_error(p, qubits)
    # Creating the noise model to be used during execution.
    noise_model = NoiseModel()
    
    noise_model.add_all_qubit_quantum_error(depo_err_chan, 'measure') # measurement error is applied to measurements
    return noise_model, depo_err_chan

def test_err_model(p, qubits=1):
    # This creates the depolarizing error channel,
    # epsilon(P) = (1-P)rho + (P/3)(XrhoX + YrhoY + ZrhoZ).
    depo_err_chan = depolarizing_error(p, qubits)
    two_qubit_err_chan = depo_err_chan.tensor(depo_err_chan)
    # Creating the noise model to be used during execution.
    noise_model = NoiseModel()
    
    noise_model.add_all_qubit_quantum_error(depo_err_chan, ['u1', 'u2', 'u3','x']) # measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(two_qubit_err_chan, ["cx"])
    return noise_model, depo_err_chan