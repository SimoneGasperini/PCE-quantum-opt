import pennylane as qml
import jax, optax
import numpy as np

# CODE for distribution:

import pennylane.numpy as pnp
import jax.numpy as jnp
from itertools import combinations, product
from base import Base


class MaxCutPCE(Base):

    def __init__(self, device, optimizer, ansatz, alpha, jit=True):
        self.device = device
        self.optimizer = optimizer
        self.ansatz = ansatz
        self.alpha = alpha
        self.jit = jit

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
        
        def circuit(params):
            self.ansatz(params)
            return [qml.expval(op) for op in pauli_ops]

        if self.jit:

            circuit = qml.QNode(circuit, device=self.device, interface="jax")

            @jax.jit
            def loss_func(params):
                expvals = circuit(params)
                print("Number of variables in circuit:", len(expvals))
                return jnp.array(
                    [
                        self.graph[i][j].get("weight", 1.0) * jnp.tanh(self.alpha * expvals[i])
                        * jnp.tanh(self.alpha * expvals[j])
                        for i, j in self.graph.edges
                    ]
                ).sum()

            @jax.jit
            def update_step(i, args):
                params, opt_state, loss_hist, obj_hist = args
                loss, grad = jax.value_and_grad(loss_func)(params)
                loss_hist = loss_hist.at[i].set(loss)
                updates, opt_state = self.optimizer.update(grad, opt_state)
                params = optax.apply_updates(params, updates)
                expvals = circuit(params)
                solution = {
                    i: jnp.where(expvals[i] > 0, 1, 0) for i in self.graph.nodes
                }
                obj = self.compute_maxcut(x=solution)
                obj_hist = obj_hist.at[i].set(obj)
                return params, opt_state, loss_hist, obj_hist

            def optimization(params, iters):
                opt_state = self.optimizer.init(params)
                loss_hist = jnp.empty(iters)
                obj_hist = jnp.empty(iters)
                args = (params, opt_state, loss_hist, obj_hist)
                params, opt_state, loss_hist, obj_hist = jax.lax.fori_loop(
                    0, iters, update_step, args
                )
                expvals = circuit(params)
                solution = {
                    i: jnp.where(expvals[i] > 0, 1, 0) for i in self.graph.nodes
                }
                return solution, loss_hist, obj_hist

            optimization = jax.jit(optimization, static_argnums=(1,))

        else:

            circuit = qml.QNode(circuit, device=self.device)

            def loss_func(params):
                expvals = circuit(params)
                for i, j in self.graph.edges:
                    w = self.graph[i][j].get("weight", 1.0) 
                    vi = expvals[i]
                    vj = expvals[j]
                    tanh_i = pnp.tanh(self.alpha * vi)
                    tanh_j = pnp.tanh(self.alpha * vj)
                    contrib = w * tanh_i * tanh_j
                return pnp.sum(
                    self.graph[i][j].get("weight", 1.0) * pnp.tanh(self.alpha * expvals[i])
                    * pnp.tanh(self.alpha * expvals[j])
                    for i, j in self.graph.edges
                )

            def optimization(params, iters):
                loss_hist = np.empty(iters)
                obj_hist = np.empty(iters)
                for i in range(iters):
                    params, loss = self.optimizer.step_and_cost(loss_func, params)
                    loss_hist[i] = loss
                    expvals = circuit(params)
                    solution = {i: 1 if expvals[i] > 0 else 0 for i in self.graph.nodes}
                    obj_hist[i] = self.compute_maxcut(x=solution)
                return solution, loss_hist, obj_hist

        self.optimization = optimization

    def run_model(self, params, iters):
        params = jnp.array(params) if self.jit else pnp.array(params)
        solution, loss_hist, obj_hist = self.optimization(params=params, iters=iters)
        if self.jit:
            solution = {i: solution[i].item() for i in self.graph.nodes}
        loss_hist = np.array(loss_hist)
        obj_hist = np.array(obj_hist)
        return solution, loss_hist, obj_hist


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

    dev = qml.device("default.qubit", wires=num_qubits)
    # opt = qml.AdamOptimizer(stepsize=0.1)
    opt = optax.adam(learning_rate=0.1)
    alpha = num_qubits ** (k_body_corr / 2)
    # pce = MaxCutPCE(device=dev, optimizer=opt, ansatz=ansatz, alpha=alpha, jit=False)
    pce = MaxCutPCE(device=dev, optimizer=opt, ansatz=ansatz, alpha=alpha, jit=True)

    pauli_ops = MaxCutPCE.get_pauli_ops(n=num_qubits, k=k_body_corr, bases=["X"])
    pce.build_model(graph=graph, num_qubits=num_qubits, pauli_ops=pauli_ops)

    params = np.random.uniform(size=(num_layers, num_qubits))
    solution, loss_hist, obj_hist = pce.run_model(params=params, iters=100)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    pce.show_result(sol=solution, ax=ax1)
    pce.show_optimization(loss_hist=loss_hist, obj_hist=obj_hist, ax=ax2)
    plt.show()
