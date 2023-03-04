from graphviz import Digraph


def trace(root):
    '''
    Collect all nodes, edges from root node by DFS traversal.
    '''
    nodes, edges = set(), set()

    def buildDFS(node):
        # visit unvisited nodes
        if (node not in nodes):
            nodes.add(node)
            # visit children
            for child in node._prev:
                edges.add((child, node))
                buildDFS(child)
    buildDFS(root)
    return nodes, edges


def getNodeID(node):
    return str(id(node))


def draw(root, format='svg', rankdir='LR'):
    '''
    Draws the computational graph (tree of math expressions + operators)
    '''

    nodes, edges = trace(root)
    # Digraph : directed graph
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    # draw nodes
    for n in nodes:
        # draw data node with 4 decimal places
        dot.node(name=getNodeID(n),
                 label=f"data {n.data:.4f} | grad {n.grad:.4f}", shape='record')
        if n._op:
            # draw operator node + edge
            dot.node(name=getNodeID(n) + n._op, label=n._op)
            # edge outgoing from operator node
            dot.edge(getNodeID(n) + n._op, getNodeID(n))

    # draw edges
    for src, dst in edges:
        # edge incoming to data node
        dot.edge(getNodeID(src), getNodeID(dst) + dst._op)

    return dot
