import numpy as np
from qiskit import transpile, ClassicalRegister, QuantumCircuit

from functions import QFT_LNN
    
def laplacian_evs_1D_simulator(params, parameters, psi_param_gate, num_qubits, simulator_backend, num_shots, boundary_condition):
    
    if (boundary_condition == "D") or (boundary_condition == "N"):
        evs = 0
        for j in range (num_qubits):
            c = ClassicalRegister(num_qubits - j, 'my_creg')
            qc = QuantumCircuit(num_qubits)
            qc.add_register(c)
            qc.append(psi_param_gate, qc.qubits)
            for i in range (num_qubits - 1 - j):
                qc.cx(num_qubits - 1 - (num_qubits - i - 2), num_qubits - 1 - (num_qubits - i - 1))
            qc.h(num_qubits - 1 - j)
            qc.measure([(index) for index in range (num_qubits - j)], c)
            qc_transpiled = transpile(qc, backend = simulator_backend, optimization_level=1)
            qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
            result = simulator_backend.run(qc_transpiled, shots = num_shots).get_counts()
            
            if j == num_qubits - 1:
                evs += (result.get('0', 0) - result.get('1', 0)) / num_shots
            else:
                evs += (result.get('0' + '1' + '0' * (num_qubits - j - 2), 0) - result.get('1' + '1' + '0' * (num_qubits - j - 2), 0)) / num_shots
        if boundary_condition == 'D':
            return evs - 2
        else:
            qc = QuantumCircuit(num_qubits)
            
            qc.append(psi_param_gate, qc.qubits)
            qc.measure_all()
            qc_transpiled = transpile(qc, backend = simulator_backend, optimization_level = 1)
            qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
            counts = simulator_backend.run(qc_transpiled, shots = num_shots).get_counts()
            
            correction = (counts.get('0' * num_qubits, 0) + counts.get('1' * num_qubits, 0)) / num_shots
            
            return evs - 2 + correction
    
    elif boundary_condition == "P":
        qc = QuantumCircuit(num_qubits+1)
        c = ClassicalRegister(1)
        qc.add_register(c)
        qc.append(psi_param_gate, qc.qubits[1:])
        qc = QFT_LNN(qc, num_qubits, 1)
        qc.h(0)
        for idx in range(num_qubits):
            qc.cp((2 * np.pi / (2**num_qubits)) * (2**idx), 0, num_qubits - idx)
        qc.h(0)
        qc.measure(0, c)
        qc_transpiled = transpile(qc, backend = simulator_backend, optimization_level = 1)
        qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
        counts = simulator_backend.run(qc_transpiled, shots = num_shots).get_counts()
        qevs = 2 * (counts.get('0', 0) - counts.get('1', 0)) / num_shots
        
        return qevs - 2
    
def laplacian_evs_1D_hardware(params, parameters, psi_param_gate, num_qubits, hardware_backend, sampler, num_shots, boundary_condition):
    
    if (boundary_condition == "D") or (boundary_condition == "N"):
        transpile_list = []
        evs = 0
        for j in range (num_qubits):
            c = ClassicalRegister(num_qubits - j, 'my_creg')
            qc = QuantumCircuit(num_qubits)
            qc.add_register(c)
            qc.append(psi_param_gate, qc.qubits)
            for i in range (num_qubits - 1 - j):
                qc.cx(num_qubits - 1 - (num_qubits - i - 2), num_qubits - 1 - (num_qubits - i - 1))
            qc.h(num_qubits - 1 - j)
            qc.measure([(index) for index in range (num_qubits - j)], c)
            qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level=3)
            qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
            
            transpile_list.append(qc_transpiled)
        if boundary_condition == 'D':
            job = sampler.run(transpile_list)
            print(f"Job ID: {job.job_id()}")
            
            result_list = job.result()
            
            for j in range (num_qubits):
                result = result_list[j].data.my_creg.get_counts()
            
                if j == num_qubits - 1:
                    evs += (result.get('0', 0) - result.get('1', 0)) / num_shots
                else:
                    evs += (result.get('0' + '1' + '0' * (num_qubits - j - 2), 0) - result.get('1' + '1' + '0' * (num_qubits - j - 2), 0)) / num_shots
            return evs - 2
        
        else:
            qc = QuantumCircuit(num_qubits)
            c = ClassicalRegister(num_qubits, 'my_creg')
            qc.add_register(c)
            
            qc.append(psi_param_gate, qc.qubits)
            qc.measure([i for i in range (num_qubits)], c)
            qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level = 3)
            qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
            transpile_list.append(qc_transpiled)
            
            job = sampler.run(transpile_list)
            print(f"Job ID: {job.job_id()}")
            
            result_list = job.result()
            
            for j in range (num_qubits + 1):
                result = result_list[j].data.my_creg.get_counts()
            
                if j == num_qubits - 1:
                    evs += (result.get('0', 0) - result.get('1', 0)) / num_shots
                elif j == num_qubits:
                    evs += (result.get('0' * num_qubits, 0) + result.get('1' * num_qubits, 0)) / num_shots
                else:
                    evs += (result.get('0' + '1' + '0' * (num_qubits - j - 2), 0) - result.get('1' + '1' + '0' * (num_qubits - j - 2), 0)) / num_shots
            return evs - 2
        
    elif boundary_condition == "P":
        qc = QuantumCircuit(num_qubits+1)
        c = ClassicalRegister(1, 'my_creg')
        qc.add_register(c)
        qc.append(psi_param_gate, qc.qubits[1:])
        qc = QFT_LNN(qc, num_qubits, 1)
        qc.h(0)
        for idx in range(num_qubits):
            qc.cp((2 * np.pi / (2**num_qubits)) * (2**idx), 0, num_qubits - idx)
        qc.h(0)
        qc.measure(0, c)
        qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level = 3)
        qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
        
        job = sampler.run([qc_transpiled])
        print(f"Job ID: {job.job_id()}")
        
        result = job.result()[0].data.my_creg.get_counts()
        
        qevs = 2 * (result.get('0', 0) - result.get('1', 0)) / num_shots
        
        return qevs - 2
