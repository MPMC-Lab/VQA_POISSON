import numpy as np
import matplotlib.pyplot as plt

num_qubit_list = [7]

shotlist = [2**20]
num_shot_label = [r"$2^{20}$"]
linestylelist = ['--']
markerlist = ['o']

for num_qubits in num_qubit_list:
    FD_trace = []
    FD_l2 = []
    FD_evs = []
    
    SD_trace = []
    SD_l2 = []
    SD_evs = []
    
    classical_trace = []
    classical_l2 = []
    classical_evs = []
    
    with open(f"log/evs_log_FD_LNN_{num_qubits}_1048576.txt", 'r') as file:
        for line in file:
            FD_evs.append(np.float64(line))
    with open(f"log/trace_log_FD_LNN_{num_qubits}_1048576.txt", 'r') as file:
        for line in file:
            FD_trace.append(np.float64(line))
    with open(f"log/l2_log_FD_LNN_{num_qubits}_1048576.txt", 'r') as file:
        for line in file:
            FD_l2.append(np.float64(line))
            
    with open(f"log/evs_log_SD_{num_qubits}_1048576.txt", 'r') as file:
        for line in file:
            SD_evs.append(np.float64(line))
    with open(f"log/trace_log_SD_{num_qubits}_1048576.txt", 'r') as file:
        for line in file:
            SD_trace.append(np.float64(line))
    with open(f"log/l2_log_SD_{num_qubits}_1048576.txt", 'r') as file:
        for line in file:
            SD_l2.append(np.float64(line))
            
    with open(f"classical_optimization/log/cost_log.txt", 'r') as file:
        for line in file:
            classical_evs.append(np.float64(line))
    with open(f"classical_optimization/log/trace_log.txt", 'r') as file:
        for line in file:
            classical_trace.append(np.float64(line))
    with open(f"classical_optimization/log/l2_log.txt", 'r') as file:
        for line in file:
            classical_l2.append(np.float64(line))

    max_iter = np.max([len(FD_trace), len(SD_trace)])

    plt.figure(figsize = (12,9))
    plt.plot(SD_l2, label = "SD", color = 'red', linewidth = 10)
    plt.plot(FD_l2, label = "FD", color = 'green', linewidth = 10)
    plt.plot(classical_l2[:max_iter], label = "Classical", color = 'black', linewidth = 10)
    plt.ylabel(r"$L_2$-norm error $\left(\log_{10}\right)$", fontsize = 40)
    plt.xlabel("Iterations", fontsize = 40)
    plt.xticks(fontsize = 35)
    plt.yticks(fontsize = 35)
    plt.legend(fontsize = 35, loc = 'upper right')
    plt.tight_layout()
    plt.savefig('history/l2_history.png', dpi = 200, transparent = True)
    plt.close()

    plt.figure(figsize = (12,9))
    plt.plot(SD_trace, label = "SD", color = 'red', linewidth = 10)
    plt.plot(FD_trace, label = "FD", color = 'green', linewidth = 10)
    plt.plot(classical_trace[:max_iter], label = "Classical", color = 'black', linewidth = 10)
    plt.ylabel(r"Trace Distance", fontsize = 40)
    plt.xlabel("Iterations", fontsize = 40)
    plt.xticks(fontsize = 35)
    plt.yticks(fontsize = 35)
    plt.legend(fontsize = 35, loc = 'upper right')
    plt.tight_layout()
    plt.savefig('history/trace_history.png', dpi = 200, transparent = True)
    plt.close()

    plt.figure(figsize = (12,9))
    plt.plot(SD_evs, label = "SD", color = 'red', linewidth = 10)
    plt.plot(FD_evs, label = "FD", color = 'green', linewidth = 10)
    plt.plot(classical_evs[:max_iter], label = "Classical", color = 'black', linewidth = 10)
    plt.ylabel(r"Cost Function", fontsize = 40)
    plt.xlabel("Iterations", fontsize = 40)
    plt.xticks(fontsize = 35)
    plt.yticks(fontsize = 35)
    plt.legend(fontsize = 35, loc = 'upper right')
    plt.tight_layout()
    plt.savefig('history/evs_history.png', dpi = 200, transparent = True)
    plt.close()
