import networkx as nx
import pylab as plt


class Base:

    def show_result(self, sol, obj, ax):
        pos = nx.spring_layout(self.graph, seed=42)
        lab2col = {0: "tab:blue", 1: "tab:orange"}
        node_color = [lab2col[sol[i]] for i in self.graph.nodes]
        nx.draw(self.graph, pos=pos, node_color=node_color, with_labels=True, ax=ax)
        ax.set_title(f"objective function = {obj}")
        plt.show()
