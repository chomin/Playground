import sys
from graphviz import Digraph


def plot():
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    steps = 4

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        j = i + 1
        while j < steps:
            u = str(j)
            v = str(i)
            # g.edge(u, v, label=f'{op}\n{attn}', fillcolor="gray")
            g.edge(v, u, fillcolor="gray")
            j += 1

    g.render("graph.pdf", view=True)


if __name__ == '__main__':
    plot()
