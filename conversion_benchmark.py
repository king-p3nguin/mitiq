import time
import warnings
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from cirq import DensityMatrixSimulator, depolarize
from qiskit_aer import AerSimulator, noise
from tqdm import tqdm

from mitiq import benchmarks, pec, zne
from mitiq.interface.conversions import convert_from_mitiq, convert_to_mitiq

warnings.filterwarnings("ignore")


def run_benchmark(n_qubits, num_cliffords, mode, backend):
    if backend == "qiskit":
        circuit = benchmarks.generate_rb_circuits(
            n_qubits=n_qubits,
            num_cliffords=num_cliffords,
            return_type="qiskit",
        )[0]
        circuit.measure_all()

        noise_model = noise.NoiseModel()
        error = noise.depolarizing_error(0.01, 1)
        noise_model.add_all_qubit_quantum_error(
            error, ["x", "y", "z", "sx", "sxdg", "rx", "ry", "rz"]
        )
        shots = 1024

        noisy_simulator = AerSimulator(noise_model=noise_model, method="density_matrix")
        simulator = AerSimulator(method="density_matrix")

        def execute(circuit, noise=True):
            if noise:
                result = noisy_simulator.run(circuit, shots=shots).result()
            else:
                result = simulator.run(circuit, shots=shots).result()
            return result.get_counts(circuit)["0" * circuit.num_qubits] / shots

    elif backend == "cirq":
        circuit = benchmarks.generate_rb_circuits(
            n_qubits=n_qubits,
            num_cliffords=num_cliffords,
            return_type="cirq",
        )[0]

        def execute(circuit, noise=True):
            if noise:
                circuit = circuit.with_noise(depolarize(p=0.01))
            rho = DensityMatrixSimulator().simulate(circuit).final_density_matrix
            return rho[0, 0].real

    # Compute the expectation value of the |0><0| observable.
    # noisy_value = execute_qiskit(circuit)
    # ideal_value = execute_qiskit(circuit, noise=False)
    # print(f"Error without mitigation: {abs(ideal_value - noisy_value) :.5f}")
    convert_to_mitiq.counter = 0
    convert_from_mitiq.counter = 0
    convert_to_mitiq.time = 0.0
    convert_from_mitiq.time = 0.0

    if mode == "ZNE":
        # print("Run ZNE...")
        start_time = time.perf_counter()
        mitigated_result = zne.execute_with_zne(circuit, execute)
        end_time = time.perf_counter()

        # print(f"Error with mitigation (ZNE) : {abs(ideal_value - mitigated_result):.{3}}")

        # print(f"convert_to_mitiq call count: {convert_to_mitiq.counter}")
        # print(f"convert_from_mitiq call count: {convert_from_mitiq.counter}")
        # print(
        #     f"conversion time: {convert_to_mitiq.time + convert_from_mitiq.time} seconds"
        # )
        zne_conversion_time = copy(convert_to_mitiq.time + convert_from_mitiq.time)

        return zne_conversion_time, end_time - start_time

    elif mode == "PEC":
        # print("Run PEC...")
        noise_level = 0.01
        reps = pec.represent_operations_in_circuit_with_local_depolarizing_noise(
            circuit, noise_level
        )
        # circuit.remove_final_measurements()
        start_time = time.perf_counter()
        mitigated_result = pec.execute_with_pec(circuit, execute, representations=reps)
        end_time = time.perf_counter()

        # print(f"Error with mitigation (PEC) : {abs(ideal_value - mitigated_result):.{3}}")

        # print(f"convert_to_mitiq call count: {convert_to_mitiq.counter}")
        # print(f"convert_from_mitiq call count: {convert_from_mitiq.counter}")
        # print(
        #     f"conversion time: {convert_to_mitiq.time + convert_from_mitiq.time} seconds"
        # )
        pec_conversion_time = copy(convert_to_mitiq.time + convert_from_mitiq.time)

        return pec_conversion_time, end_time - start_time


def plot_benchmark(n_qubits, num_cliffords, mitigation_type, backend):
    conv_times = []
    total_times = []
    for i in tqdm(range(len(num_cliffords))):
        ct, tt = run_benchmark(n_qubits, num_cliffords[i], mitigation_type, backend)
        conv_times.append(ct)
        total_times.append(tt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"{mitigation_type} Benchmark for {n_qubits} qubits ({backend})")
    ax1.plot(num_cliffords, conv_times, label=f"{mitigation_type} Conversion Time")
    ax1.plot(num_cliffords, total_times, label="Total Time")
    ax1.set_xlabel("Number of Cliffords")
    ax1.set_ylabel("Time (s)")
    ax1.legend()
    ax2.bar(
        list(range(len(num_cliffords))),
        np.array(conv_times) / np.array(total_times),
        tick_label=num_cliffords,
        align="center",
        label="Conversion Time / Total Simulation Time",
    )
    ax2.set_xlabel("Number of Cliffords")
    ax2.set_ylabel("Conversion Time / Total Simulation Time")
    ax2.legend()
    plt.show()


n_qubits = 2
num_cliffords = list(range(1, 100, 10))
plot_benchmark(n_qubits, num_cliffords, "ZNE", "qiskit")
plot_benchmark(n_qubits, num_cliffords, "ZNE", "cirq")


n_qubits = 1
num_cliffords = list(range(1, 10, 1))
plot_benchmark(n_qubits, num_cliffords, "PEC", "qiskit")
plot_benchmark(n_qubits, num_cliffords, "PEC", "cirq")
