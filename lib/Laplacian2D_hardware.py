import numpy as np
from qiskit import transpile, ClassicalRegister, QuantumCircuit

from functions import QFT_LNN

def laplacian_evs_2D_hardware_dirichlet(params, parameters, psi_param_gate, num_qubits, hardware_backend, sampler, num_shots, axis):
    evs = 0
    num_qubits1D = np.int64(num_qubits / 2)
    
    if axis == 'x':
        transpile_list = []
        
        for j in range (num_qubits1D):
            
            c = ClassicalRegister(num_qubits1D - j, 'my_creg')
            
            qc = QuantumCircuit(num_qubits)
            qc.add_register(c)
            qc.append(psi_param_gate, qc.qubits)
            
            for i in range (num_qubits1D-j-1):
                qc.cx(num_qubits - i - 2, num_qubits - i - 1)
            
            qc.h(j + num_qubits1D)
            
            qc.measure([(2 * num_qubits1D - 1 - index) for index in range (num_qubits1D - j)], c)
            
            qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level = 3)
            qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
            
            transpile_list.append(qc_transpiled)

        result_list = sampler.run(transpile_list).result()
        
        for j in range (num_qubits1D):
            result = result_list[j].data.my_creg.get_counts()
            
            if j == num_qubits1D - 1:
                evs += (result.get('0', 0) - result.get('1', 0)) / num_shots
            else:
                evs += (result.get('0' + '1' + '0' * (num_qubits1D - j - 2), 0) - result.get('1' + '1' + '0' * (num_qubits1D - j - 2), 0)) / num_shots
            
        return evs - 2
    
    elif axis == 'y':
        transpile_list = []
        for j in range (num_qubits1D):
            
            c = ClassicalRegister(num_qubits1D - j, 'my_creg')
            
            qc = QuantumCircuit(num_qubits)
            qc.add_register(c)
            qc.append(psi_param_gate, qc.qubits)
            
            for i in range (num_qubits1D-j-1):
                qc.cx(num_qubits1D - i - 2, num_qubits1D - i-1)
                
            qc.h(j)
            
            qc.measure([(num_qubits1D - 1 - index) for index in range (num_qubits1D - j)], c)
            
            qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level = 1)
            qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
            
            transpile_list.append(qc_transpiled)
        
        result_list = sampler.run(transpile_list).result()
        
        for j in range (num_qubits1D):
            result = result_list[j].data.my_creg.get_counts()
            
            if j == num_qubits1D - 1:
                evs += (result.get('0', 0) - result.get('1', 0)) / num_shots
            else:
                evs += (result.get('0' + '1' + '0' * (num_qubits1D - j - 2), 0) - result.get('1' + '1' + '0' * (num_qubits1D - j - 2), 0)) / num_shots
            
        return evs - 2
        
    else:
        raise ValueError("Invalid axis: must be either 'x' or 'y'.")
    
def laplacian_evs_2D_hardware_dirichlet(params, parameters, psi_param_gate, num_qubits, hardware_backend, sampler, num_shots, axis):
    
    evs = 0
    num_qubits1D = np.int64(num_qubits / 2)
    
    if axis == 'y':
        transpile_list = []
        for j in range (num_qubits1D):
            
            c = ClassicalRegister(num_qubits1D - j, 'my_creg')
            
            qc = QuantumCircuit(num_qubits)
            qc.add_register(c)
            qc.append(psi_param_gate, qc.qubits)
            
            for i in range (num_qubits1D-j-1):
                # qc.cx(num_qubits - 1  - (num_qubits - i - 2) + num_qubits1D, num_qubits - 1 - (num_qubits - i - 1) + num_qubits1D)
                qc.cx((i + 1) + num_qubits1D, i + num_qubits1D)
            
            # qc.h(num_qubits - 1 - (j + num_qubits1D) + num_qubits1D)
            qc.h(num_qubits - 1 - j)
            
            qc.measure([(index + num_qubits1D) for index in range (num_qubits1D - j)], c)
            
            qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level = 3)
            qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
            
            transpile_list.append(qc_transpiled)
        job = sampler.run(transpile_list)
        
        print(f"Job ID: {job.job_id()}")
        
        result_list = job.result()
        for j in range (num_qubits1D):
            result = result_list[j].data.my_creg.get_counts()

            if j == num_qubits1D - 1:
                evs += (result.get('0', 0) - result.get('1', 0)) / num_shots
            else:
                evs += (result.get('0' + '1' + '0' * (num_qubits1D - j - 2), 0) - result.get('1' + '1' + '0' * (num_qubits1D - j - 2), 0)) / num_shots
            
        return evs - 2
    
    elif axis == 'x':
        transpile_list = []
        for j in range (num_qubits1D):
            
            c = ClassicalRegister(num_qubits1D - j, 'my_creg')
            
            qc = QuantumCircuit(num_qubits)
            qc.add_register(c)
            qc.append(psi_param_gate, qc.qubits)
            
            for i in range (num_qubits1D-j-1):
                qc.cx(num_qubits1D - 1 - (num_qubits1D - i - 2), num_qubits1D - 1 - (num_qubits1D - i-1))
                
            qc.h(num_qubits1D - 1 - j)
            
            qc.measure([(num_qubits1D - 1 - (num_qubits1D - 1 - index)) for index in range (num_qubits1D - j)], c)
            
            qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level = 3)
            qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
            
            transpile_list.append(qc_transpiled)
        job = sampler.run(transpile_list)
        print(f"Job ID: {job.job_id()}")
        result_list = job.result()
        
        for j in range (num_qubits1D):
            result = result_list[j].data.my_creg.get_counts()
            
            if j == num_qubits1D - 1:
                evs += (result.get('0', 0) - result.get('1', 0)) / num_shots
            else:
                evs += (result.get('0' + '1' + '0' * (num_qubits1D - j - 2), 0) - result.get('1' + '1' + '0' * (num_qubits1D - j - 2), 0)) / num_shots
            
        return evs - 2
        
    else:
        raise ValueError("Invalid axis: must be either 'x' or 'y'.")
        
        
def laplacian_evs_2D_hardware_neumann(params, parameters, psi_param_gate, num_qubits, hardware_backend, sampler, num_shots, axis):
    evs = 0
    num_qubits1D = np.int64(num_qubits / 2)
    
    if axis == 'x':
        transpile_list = []
        for j in range (num_qubits1D):
            
            c = ClassicalRegister(num_qubits1D - j, 'my_creg')
            
            qc = QuantumCircuit(num_qubits)
            qc.add_register(c)
            qc.append(psi_param_gate, qc.qubits)
            
            for i in range (num_qubits1D-j-1):
                qc.cx(num_qubits - i - 2, num_qubits - i - 1)
            
            qc.h(j + num_qubits1D)
            
            qc.measure([(2 * num_qubits1D - 1 - index) for index in range (num_qubits1D - j)], c)
            
            qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level = 3)
            qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
            
            transpile_list.append(qc_transpiled)
        
        qc = QuantumCircuit(num_qubits)
        qc.append(psi_param_gate, qc.qubits)
        c = ClassicalRegister(num_qubits1D, 'my_creg')
        
        qc.add_register(c)
        qc.measure([(i + num_qubits1D) for i in range (num_qubits1D)], c)
        qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level = 3)
        qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
        transpile_list.append(qc_transpiled)
        
        result_list = sampler.run(transpile_list).result()
        
        for j in range (num_qubits1D + 1):
            result = result_list[j].data.my_creg.get_counts()
            if j == num_qubits1D - 1:
                evs += (result.get('0', 0) - result.get('1', 0)) / num_shots
            elif j == num_qubits1D:
                evs += (result.get('0' * num_qubits1D, 0) + result.get('1' * num_qubits1D, 0)) / num_shots
            else:
                evs += (result.get('0' + '1' + '0' * (num_qubits1D - j - 2), 0) - result.get('1' + '1' + '0' * (num_qubits1D - j - 2), 0)) / num_shots
            
        return evs - 2
    
    elif axis == 'y':
        transpile_list = []
        for j in range (num_qubits1D):
            
            c = ClassicalRegister(num_qubits1D - j, 'my_creg')
            
            qc = QuantumCircuit(num_qubits)
            qc.add_register(c)
            qc.append(psi_param_gate, qc.qubits)
            
            for i in range (num_qubits1D-j-1):
                qc.cx(num_qubits1D - i - 2, num_qubits1D - i - 1)
                
            qc.h(j)
            
            qc.measure([(num_qubits1D - 1 - index) for index in range (num_qubits1D - j)], c)
            
            qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level = 3)
            qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
            
            transpile_list.append(qc_transpiled)
            
        
                
        qc = QuantumCircuit(num_qubits)
        qc.append(psi_param_gate, qc.qubits)
        c = ClassicalRegister(num_qubits1D, 'my_creg')
        
        qc.add_register(c)
        qc.measure([i for i in range (num_qubits1D)], c)
        
        qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level = 3)
        qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
        transpile_list.append(qc_transpiled)
        
        result_list = sampler.run(transpile_list).result()
        
        for j in range (num_qubits1D + 1):
            result = result_list[j].data.my_creg.get_counts()
            
            if j == num_qubits1D - 1:
                evs += (result.get('0', 0) - result.get('1', 0)) / num_shots
            elif j == num_qubits1D:
                evs += (result.get('0' * num_qubits1D, 0) + result.get('1' * num_qubits1D, 0)) / num_shots
            else:
                evs += (result.get('0' + '1' + '0' * (num_qubits1D - j - 2), 0) - result.get('1' + '1' + '0' * (num_qubits1D - j - 2), 0)) / num_shots
                
        evs += (result.get('0' * num_qubits1D, 0) + result.get('1' * num_qubits1D, 0)) / num_shots
        
        return evs - 2
        
    else:
        raise ValueError("Invalid axis: must be either 'x' or 'y'.")
    
def laplacian_evs_2D_hardware_neumann(params, parameters, psi_param_gate, num_qubits, hardware_backend, sampler, num_shots, axis):
    evs = 0
    num_qubits1D = np.int64(num_qubits / 2)
    
    if axis == 'y':
        transpile_list = []
        for j in range (num_qubits1D):
            
            c = ClassicalRegister(num_qubits1D - j, 'my_creg')
            
            qc = QuantumCircuit(num_qubits)
            qc.add_register(c)
            qc.append(psi_param_gate, qc.qubits)
            
            for i in range (num_qubits1D-j-1):
                # qc.cx(num_qubits - 1  - (num_qubits - i - 2) + num_qubits1D, num_qubits - 1 - (num_qubits - i - 1) + num_qubits1D)
                qc.cx((i + 1) + num_qubits1D, i + num_qubits1D)
            
            # qc.h(num_qubits - 1 - (j + num_qubits1D) + num_qubits1D)
            qc.h(num_qubits - 1 - j)
            
            qc.measure([(index + num_qubits1D) for index in range (num_qubits1D - j)], c)
            
            qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level = 3)
            qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
            
            transpile_list.append(qc_transpiled)
        qc = QuantumCircuit(num_qubits)
        qc.append(psi_param_gate, qc.qubits)
        c = ClassicalRegister(num_qubits1D, 'my_creg')
        
        qc.add_register(c)
        qc.measure([(i + num_qubits1D) for i in range (num_qubits1D)], c)
        
        qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level = 3)
        qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
        transpile_list.append(qc_transpiled)
        
        job = sampler.run(transpile_list)
        print(f"Job ID: {job.job_id()}")
        
        result_list = job.result()
        for j in range (num_qubits1D + 1):
            result = result_list[j].data.my_creg.get_counts()
            
            if j == num_qubits1D - 1:
                evs += (result.get('0', 0) - result.get('1', 0)) / num_shots
            elif j == num_qubits1D:
                evs += (result.get('0' * num_qubits1D, 0) + result.get('1' * num_qubits1D, 0)) / num_shots
            else:
                evs += (result.get('0' + '1' + '0' * (num_qubits1D - j - 2), 0) - result.get('1' + '1' + '0' * (num_qubits1D - j - 2), 0)) / num_shots

        return evs - 2
    
    elif axis == 'x':
        transpile_list = []
        for j in range (num_qubits1D):
            
            c = ClassicalRegister(num_qubits1D - j, 'my_creg')
            
            qc = QuantumCircuit(num_qubits)
            qc.add_register(c)
            qc.append(psi_param_gate, qc.qubits)
            
            for i in range (num_qubits1D-j-1):
                qc.cx(num_qubits1D - 1 - (num_qubits1D - i - 2), num_qubits1D - 1 - (num_qubits1D - i-1))
                
            qc.h(num_qubits1D - 1 - j)
            
            qc.measure([(num_qubits1D - 1 - (num_qubits1D - 1 - index)) for index in range (num_qubits1D - j)], c)
            
            qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level = 3)
            qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
            
            transpile_list.append(qc_transpiled)
        
        qc = QuantumCircuit(num_qubits)
        qc.append(psi_param_gate, qc.qubits)
        c = ClassicalRegister(num_qubits1D, 'my_creg')
        
        qc.add_register(c)
        qc.measure([i for i in range (num_qubits1D)], c)
        
        qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level = 3)
        qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
        
        transpile_list.append(qc_transpiled)
        
        job = sampler.run(transpile_list)
        print(f"Job ID: {job.job_id()}")
        
        result_list = job.result()
        
        for j in range (num_qubits1D + 1):
            result = result_list[j].data.my_creg.get_counts()
            
            if j == num_qubits1D - 1:
                evs += (result.get('0', 0) - result.get('1', 0)) / num_shots
            elif j == num_qubits1D:
                evs += (result.get('0' * num_qubits1D, 0) + result.get('1' * num_qubits1D, 0)) / num_shots
            else:
                evs += (result.get('0' + '1' + '0' * (num_qubits1D - j - 2), 0) - result.get('1' + '1' + '0' * (num_qubits1D - j - 2), 0)) / num_shots
        return evs - 2
        
    else:
        raise ValueError("Invalid axis: must be either 'x' or 'y'.")
    
def laplacian_evs_2D_hardware_periodic(params, parameters, psi_param_gate, num_qubits, hardware_backend, sampler, num_shots, axis):
    
    num_qubits1D = np.int64(num_qubits / 2)
    
    if axis == 'x':
        qc = QuantumCircuit(num_qubits + 1)
        c = ClassicalRegister(1, 'my_creg')
        qc.add_register(c)
        qc.append(psi_param_gate, qc.qubits[1:])
        
        qc = QFT_LNN(qc, num_qubits1D, 1 + num_qubits1D)
        
        qc.h(0)
        
        for idx in range(num_qubits1D):
            qc.cp((2 * np.pi / (2**num_qubits1D)) * (2**idx), 0, idx + 1 + num_qubits1D)
            
        qc.h(0)
        
        qc.measure([0], c)
        
        qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level=3)
        qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
        
        counts = sampler.run([qc_transpiled]).result()[0].data.my_creg.get_counts()
        
        qevs = 2 * ((counts.get('0', 0) - counts.get('1', 0))) / num_shots
        return qevs - 2
    
    elif axis == 'y':
        qc = QuantumCircuit(num_qubits + 1)
        c = ClassicalRegister(1, 'my_creg')
        qc.add_register(c)
        qc.append(psi_param_gate, qc.qubits[1:])
        
        qc = QFT_LNN(qc, num_qubits1D, 1)
        qc.h(0)
        for idx in range(num_qubits1D):
            qc.cp((2 * np.pi / (2**num_qubits1D)) * (2**idx), 0, idx+1)
            
        qc.h(0)
        
        qc.measure([0], c)
        
        qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level=3)
        qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
        
        counts = sampler.run([qc_transpiled]).result()[0].data.my_creg.get_counts()
        
        qevs = 2 * ((counts.get('0', 0) - counts.get('1', 0))) / num_shots
        return qevs - 2
    else:
        raise ValueError("Invalid axis: must be either 'x' or 'y'.")
        

def laplacian_evs_2D_hardware_periodic(params, parameters, psi_param_gate, num_qubits, hardware_backend, sampler, num_shots, axis):
    
    num_qubits1D = np.int64(num_qubits / 2)
    
    if axis == 'y':
        qc = QuantumCircuit(num_qubits + 1)
        c = ClassicalRegister(1, 'my_creg')
        qc.add_register(c)
        qc.append(psi_param_gate, qc.qubits[1:])
        
        qc = QFT_LNN(qc, num_qubits1D, 1 + num_qubits1D)
        
        qc.h(0)
        
        for idx in range(num_qubits1D):
            qc.cp((2 * np.pi / (2**num_qubits1D)) * (2**idx), 0, num_qubits1D - idx + num_qubits1D)
            
        qc.h(0)
        
        qc.measure([0], c)
        
        qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level=3)
        qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
        
        job = sampler.run([qc_transpiled])
        print(f"Job ID: {job.job_id()}")
        
        counts = job.result()[0].data.my_creg.get_counts()
        qevs = 2 * ((counts.get('0', 0) - counts.get('1', 0))) / num_shots
        return qevs - 2
    
    elif axis == 'x':
        qc = QuantumCircuit(num_qubits + 1)
        c = ClassicalRegister(1, 'my_creg')
        qc.add_register(c)
        qc.append(psi_param_gate, qc.qubits[1:])
        
        qc = QFT_LNN(qc, num_qubits1D, 1)
        qc.h(0)
        for idx in range(num_qubits1D):
            qc.cp((2 * np.pi / (2**num_qubits1D)) * (2**idx), 0, num_qubits1D - idx)
            
        qc.h(0)
        
        qc.measure([0], c)
        
        qc_transpiled = transpile(qc, backend = hardware_backend, optimization_level=3)
        qc_transpiled = qc_transpiled.assign_parameters({parameters[i]: params[i] for i in range (len(parameters))})
        
        job = sampler.run([qc_transpiled])
        print(f"Job ID: {job.job_id()}")
        
        counts = job.result()[0].data.my_creg.get_counts()
        
        qevs = 2 * ((counts.get('0', 0) - counts.get('1', 0))) / num_shots
        return qevs - 2
    else:
        raise ValueError("Invalid axis: must be either 'x' or 'y'.")
    