import pennylane as qml
from pennylane import numpy as np
from base import Base


class MaxCutQAOA(Base):

    def __init__(self, device, optimizer, p):
        self.device = device
        self.optimizer = optimizer
        self.p = p

    def build_model(self, graph):
        self.graph = graph
        self.num_qubits = len(graph.nodes)
        self.cost_ham, self.mixer_ham = qml.qaoa.maxcut(self.graph)

        def circuit(params):
            for i in range(self.num_qubits):
                qml.Hadamard(wires=i)
            for gamma, beta in zip(params[0], params[1]):
                qml.qaoa.cost_layer(gamma, self.cost_ham)
                qml.qaoa.mixer_layer(beta, self.mixer_ham)

        @qml.qnode(self.device)
        def circuit_expval(params):
            circuit(params)
            return qml.expval(self.cost_ham)

        @qml.qnode(self.device)
        def circuit_probs(params):
            circuit(params)
            return qml.probs()

        self.circuit_expval = circuit_expval
        self.circuit_probs = circuit_probs

    def run_model(self, iters):
        params = np.random.uniform(size=(2, self.p), requires_grad=True)
        for _ in range(iters):
            params, _ = self.optimizer.step_and_cost(self.circuit_expval, params)
        probs = self.circuit_probs(params)
        bitstr = format(np.argmax(probs), f"0{self.num_qubits}b")
        solution = {i: int(bitstr[i]) for i in self.graph.nodes}
        objval = self.compute_maxcut(x=solution)
        return solution, objval


if __name__ == "__main__":

    import networkx as nx
    import pylab as plt

    num_nodes = 6
    graph = nx.star_graph(n=num_nodes - 1)

    dev = qml.device("default.qubit")
    opt = qml.optimize.AdamOptimizer(stepsize=0.1)
    p = 2
    qaoa = MaxCutQAOA(device=dev, optimizer=opt, p=p)

    qaoa.build_model(graph=graph)
    solution, objval = qaoa.run_model(iters=100)

    fig, ax = plt.subplots()
    qaoa.show_result(sol=solution, obj=objval, ax=ax)
