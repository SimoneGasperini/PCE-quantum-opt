import networkx as nx


class Base:

    def compute_maxcut(self, x):
        total = 0
        for i, j in self.graph.edges:
            w = self.graph[i][j].get("weight", 1.0)
            val = w * (x[i] + x[j] - 2 * x[i] * x[j])
            total += val
        return total

    def show_result(self, sol, ax):
        pos = nx.spring_layout(self.graph, seed=42)
        lab2col = {0: "tab:blue", 1: "tab:orange"}
        node_color = [lab2col[sol[i]] for i in self.graph.nodes]
        nx.draw(self.graph, pos=pos, node_color=node_color, with_labels=True, ax=ax)
        obj = self.compute_maxcut(x=sol)
        ax.set_title(f"maxcut = {obj}")

    def show_optimization(self, loss_hist, obj_hist, ax):
        ax.plot(loss_hist, color="tab:red", label="loss")
        ax.plot(obj_hist, color="tab:green", label="maxcut")
        ax.set_xlabel("iteration")
        ax.legend()
