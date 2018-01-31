import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def __init__:
    graph_list = []
    node_list = []

    g = [
    [0,0,0,0,1,2,3,4,0],
    [0,0,0,0,2,1,2,3,0],
    [0,0,0,0,3,2,1,2,0],
    [0,0,0,0,4,3,2,1,0],
    [1,2,3,4,0,0,0,0,1],
    [2,1,2,3,0,0,0,0,1],
    [3,2,1,2,0,0,0,0,1],
    [4,3,2,1,0,0,0,0,1],
    [0,0,0,0,1,1,1,1,0]
    ]
    node = np.array(['s1', 's2', 's3', 's4', 'p1', "p2","p3","p4","g"])

    graph_list.append(g)
    node_list.append(node)

    g = [
        [0,0,0,1,2,3,0],
        [0,0,0,2,1,2,0],
        [0,0,0,3,2,1,0],
        [1,2,3,0,0,0,1],
        [2,1,2,0,0,0,1],
        [3,2,1,0,0,0,1],
        [0,0,0,1,1,1,0]
    ]
    node = np.array(['s1', 's2', 's3', 'p1', "p2","p3","g"])

    graphs.append(g)
    node_list.append(node)

    g = [
        [0,   0,   0.1, 0.4, 0],
        [0,   0,   0.4, 0.1, 0],
        [0.1, 0.4, 0,   0,   1],
        [0.4, 0.1, 0,   0,   1],
        [0,   0,   1,   1,   0]
    ]
    node = np.array(['s1', 's2', 'p1', "p2", "g"])

    graphs.append(g)
    node_list.append(node)
