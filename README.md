Exam1:
- Backend: Qiskit_QASM simulation
- Noise: Depolarizing noise model
- Circuit:
    1. Normal cat state preparation with normal error correction
    2. (Cat state preparation) + ([4,1,4] noise-trained DQNN)
    3. Repeat preparation
- Purpose: Preparation of a 4-qubit cat state
- Output: The fidelity calculated by this DQNN as an error correction process changes under multiple noise rates, and compares the fidelity changes of three different preparation strategies
- Result Analysis: DQNN shows resistance to noise and is significantly better than normal cat state preparation with normal error correction and repeat preparation
- Note: Training for the regular term still needs to be improved

Exam2:
- Backend: Aer_simulator simulation
- Noise: Depolarizing noise model (including gates and measurements)
- Circuit:
    1. (7-qubit error correction code preparation and error correction process) + ([4,1,4] noise-trained DQNN auxiliary state preparation) + (trained RealAmplitudes regular term)
    2. (7-qubit error correction code preparation and error correction process) + error correction cat state preparation circuit
- Purpose: Compared to the traditional cat state with correction module, it shows the advantages of DQNN auxiliary state (fidelity, logical error rate)
- Output: Experimental images
- Result Analysis: In this noise environment, the given result analysis has obvious differences
- Note: The seven-qubit error correction code encoding circuit comes from the Stac software package generation (github: https://abdullahkhalid.com/qecft/introduction/stac/)

Exam3:
- Backend: Aer_simulator simulation
- Noise: Depolarizing noise model (including gates and measurements)
- Circuit:
    1. (5-qubit error correction code preparation and error correction process) + ([4,1,4] noise-trained DQNN auxiliary state preparation)
    2. (5-qubit error correction code preparation and error correction process) + error correction cat state preparation circuit
- Purpose: Compared to the traditional cat state with correction module, it shows the advantages of DQNN auxiliary state (fidelity, logical error rate)
- Output: Experimental images
- Result Analysis: In this fair environment, the given result analysis has obvious differences
- Note: The five-qubit error correction code encoding circuit and error correction code come from the Stac software package generation (github: https://abdullahkhalid.com/qecft/introduction/stac/)

Note: The simulation platform version is limited to Qiskit 0.46.0.




