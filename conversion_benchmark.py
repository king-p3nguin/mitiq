import numpy as np
from qiskit_aer import AerSimulator, noise

from mitiq import benchmarks, pec, zne
from mitiq.interface.conversions import convert_to_mitiq

circuit = benchmarks.generate_rb_circuits(
    n_qubits=1,
    num_cliffords=2,
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


def execute_qiskit(circuit, noise=True):
    try:
        execute_qiskit.counter += 1
    except AttributeError:
        execute_qiskit.counter = 1
    if noise:
        result = noisy_simulator.run(circuit, shots=shots).result()
    else:
        result = simulator.run(circuit, shots=shots).result()
    return result.get_counts(circuit)["0" * circuit.num_qubits] / shots


# Compute the expectation value of the |0><0| observable.
noisy_value = execute_qiskit(circuit)
ideal_value = execute_qiskit(circuit, noise=False)
print(f"Error without mitigation: {abs(ideal_value - noisy_value) :.5f}")
convert_to_mitiq.counter = 0

print("Run ZNE...")
mitigated_result = zne.execute_with_zne(circuit, execute_qiskit)

print(f"Error with mitigation (ZNE) : {abs(ideal_value - mitigated_result):.{3}}")

print(f"convert_to_mitiq call count: {convert_to_mitiq.counter}")
convert_to_mitiq.counter = 0
print("----------------------")

print("Run PEC...")
noise_level = 0.01
reps = pec.represent_operations_in_circuit_with_local_depolarizing_noise(
    circuit, noise_level
)
# circuit.remove_final_measurements()
mitigated_result = pec.execute_with_pec(circuit, execute_qiskit, representations=reps)

print(f"Error with mitigation (PEC) : {abs(ideal_value - mitigated_result):.{3}}")

print(f"convert_to_mitiq call count: {convert_to_mitiq.counter}")
