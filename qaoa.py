import pennylane as qml
import pennylane.numpy as pnp
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
        params = pnp.random.uniform(size=(2, self.p), requires_grad=True)
        loss_hist = pnp.empty(iters)
        obj_hist = pnp.empty(iters)
        for i in range(iters):
            params, loss = self.optimizer.step_and_cost(self.circuit_expval, params)
            loss_hist[i] = loss
            probs = self.circuit_probs(params)
            bitstr = format(pnp.argmax(probs), f"0{self.num_qubits}b")
            solution = {i: int(bitstr[i]) for i in self.graph.nodes}
            obj_hist[i] = self.compute_maxcut(x=solution)
        return solution, loss_hist, obj_hist


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
    solution, loss_hist, obj_hist = qaoa.run_model(iters=100)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    qaoa.show_result(sol=solution, ax=ax1)
    qaoa.show_optimization(loss_hist=loss_hist, obj_hist=obj_hist, ax=ax2)
    plt.show()
