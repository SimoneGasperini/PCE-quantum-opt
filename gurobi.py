import gurobipy as gp
from base import Base


class MaxCutGurobi(Base):

    def __init__(self):
        pass

    def build_model(self, graph):
        self.graph = graph
        self.model = gp.Model()
        self.vars = {}
        for i in self.graph.nodes:
            self.vars[i] = self.model.addVar(vtype=gp.GRB.BINARY)
        self.obj = sum(
            -self.vars[i] - self.vars[j] + 2 * self.vars[i] * self.vars[j]
            for i, j in graph.edges
        )
        self.model.setObjective(self.obj, sense=gp.GRB.MINIMIZE)

    def run_model(self, verbose):
        if not verbose:
            self.model.setParam("OutputFlag", 0)
        self.model.optimize()
        solution = {i: int(var.x) for i, var in self.vars.items()}
        return solution


if __name__ == "__main__":

    import networkx as nx
    import pylab as plt

    num_nodes = 6
    graph = nx.star_graph(n=num_nodes - 1)

    grb = MaxCutGurobi()
    grb.build_model(graph=graph)
    solution = grb.run_model(verbose=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    grb.show_result(sol=solution, ax=ax)
    plt.show()
