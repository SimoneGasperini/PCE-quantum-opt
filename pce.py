import pennylane as qml
from pennylane import numpy as np
from itertools import combinations, product
from base import Base


class MaxCutPCE(Base):
    # TODO:
    #   -introduce regularization term in the objective function
    #   -generalize the algorithm to MaxCut problems over weighted graphs
    #   -enable Pennylane jit-compilation to speed-up the qnode execution

    def __init__(self, device, optimizer, ansatz, alpha):
        self.device = device
        self.optimizer = optimizer
        self.ansatz = ansatz
        self.alpha = alpha

    @staticmethod
    def get_pauli_ops(n, k, bases):
        pauli_ops = []
        char2op = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
        for qubit_indices in combinations(range(n), k):
            for pauli_string in product(bases, repeat=k):
                pauli_op = [qml.Identity(i) for i in range(n)]
                for i, wire in enumerate(qubit_indices):
                    qml_op = char2op[pauli_string[i]]
                    pauli_op[wire] = qml_op(wire)
                pauli_ops.append(qml.prod(*pauli_op))
        return pauli_ops

    def build_model(self, graph, num_qubits, pauli_ops):
        self.graph = graph
        self.num_qubits = num_qubits

        @qml.qnode(self.device)
        def circuit(params):
            self.ansatz(params)
            return [qml.expval(op) for op in pauli_ops]

        def objective(params):
            expvals = circuit(params)
            return sum(
                np.tanh(self.alpha * expvals[i]) * np.tanh(self.alpha * expvals[j])
                for i, j in self.graph.edges
            )

        self.circuit = circuit
        self.objective = objective

    def run_model(self, params, iters):
        cost_history = []
        obj_history = []
        for _ in range(iters):
            params, cost = self.optimizer.step_and_cost(self.objective, params)
            cost_history.append(cost)
            expvals = self.circuit(params)
            solution = {i: 1 if expvals[i] > 0 else 0 for i in self.graph.nodes}
            obj = self.compute_maxcut(x=solution)
            obj_history.append(obj)
        return solution, cost_history, obj_history


if __name__ == "__main__":

    import networkx as nx
    import pylab as plt

    num_nodes = 6
    graph = nx.star_graph(n=num_nodes - 1)

    num_qubits = 4
    num_layers = 2
    k_body_corr = 2

    def ansatz(params):
        for l in range(num_layers):
            thetas = params[l]
            for w in range(num_qubits):
                qml.RY(phi=thetas[w], wires=w)
            for w in range(num_qubits - 1):
                qml.CNOT(wires=[w, w + 1])

    dev = qml.device("default.qubit")
    opt = qml.optimize.AdamOptimizer(stepsize=0.1)
    alpha = num_qubits ** (k_body_corr / 2)
    pce = MaxCutPCE(device=dev, optimizer=opt, ansatz=ansatz, alpha=alpha)

    pauli_ops = MaxCutPCE.get_pauli_ops(n=num_qubits, k=k_body_corr, bases=["X"])
    pce.build_model(graph=graph, num_qubits=num_qubits, pauli_ops=pauli_ops)

    params = np.random.uniform(size=(num_layers, num_qubits), requires_grad=True)
    solution, cost_history, obj_history = pce.run_model(params=params, iters=100)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    pce.show_result(sol=solution, ax=ax1)
    pce.show_optimization(cost_history=cost_history, obj_history=obj_history, ax=ax2)
    plt.show()
