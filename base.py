import networkx as nx


class Base:

    def compute_maxcut(self, x):
        # binary vars in {0,1}
        return sum(
            edge.get("weight", 1.0) * (x[i] + x[j] - 2 * x[i] * x[j])
            for i, j, edge in self.graph.edges(data=True)
        )

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
