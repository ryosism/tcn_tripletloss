import numpy as np
from glob import glob
import os
import sys
import time
import random

import networkx as nx

def make_tensor(all_confidences, order_num, alpha):

    # all_confidences[#手順画像][#frame]

    table = np.zeros([order_num,order_num])
    np.set_printoptions(precision=3, suppress=True) #指数表示の禁止

    for i in range(0,order_num):
        table[i][i] = 0

    for i in range(0,order_num-1):
        table[i][i+1] = 0.02

    for i in range(0,order_num-1):
        table[i+1][i] = 1.1

    for c in range(2, order_num):
        for b in range(0, order_num-c):
            table[b][b+c] = 0.05*(c-1)

    for c in range(1, order_num):
        for b in range(0, order_num-c):
            table[b+c][b] = 1.2*c

    # ここまでで重みテーブルは完成
    print(table)
    table = table * alpha
    print(table)

    # ノード作成、と同時にノードの値(confidence)も設定
    G = nx.DiGraph()
    G.add_node(0, value = 0)

    frames = len(all_confidences[0])
    for i in range(frames * order_num):
        G.add_node(i+1)

        # try:
        G.nodes[i+1]['value'] = all_confidences[int(i % order_num)][int(i/order_num)]
        #     break
        # except ValueError:
        #     print("[ValueError] int(i % order_num) = {}, int(i/order_num) = {}".format(int(i % order_num), int(i/order_num)))
    print("{} nodes created.".format(i))

    G.add_node(frames * order_num+1, value = 0)

    # スタートから最初のキーへのエッジ
    for i in range(1,order_num+2):
        G.add_edge(0, i, weight = 1)

    # 全エッジに対して重み設定
    for t in range(frames-1):
        for i in range(order_num*(t)+1, order_num*(t+1)+1):
            for j in range(order_num*(t+1)+1, order_num*(t+2)+1):
                if i < 100:
                    print("i, j, {}, {}, weight = {}(table[{}][{}])+ {}".format(i, j, table[(i % order_num)-1][(j % order_num)-1], (i % order_num)-1, (j % order_num)-1, G.nodes[j]['value']))
                elif i > 96650:
                    print("i, j, {}, {}, weight = {}(table[{}][{}])+ {}".format(i, j, table[(i % order_num)-1][(j % order_num)-1], (i % order_num)-1, (j % order_num)-1, G.nodes[j]['value']))

                G.add_edge(i, j, weight = (table[i % order_num][j % order_num] + G.nodes[j]['value']))

    # 最後のノードからゴールへのエッジ
    for i in range((frames-1) * order_num, frames * order_num+1):
        G.add_edge(i, frames * order_num+1, weight = 1)

    return G

def print_path(prev, cost):
    for i in range(len(prev)):
        print("%d, prev = %d, cost = %d" %  (i, prev[i], cost[i]))

def get_path(start, goal, prev):
    route = []
    now = goal
    route.append(now)
    while True:
        route.append(prev[now])
        if prev[now] == start: break
        now = prev[now]
    route.reverse()
    return route

def search(glaph, start, goal):
    MAX_VAL = 0x10000000
    g_size = len(glaph)
    visited = [False] * g_size
    cost = [MAX_VAL] * g_size
    prev = [None] * g_size
    cost[start] = 0
    prev[start] = start
    now = start
    while True:
        min = MAX_VAL
        next = -1
        visited[now] = True
        for i in range(g_size):
            if visited[i]: continue
            if glaph[now][i]:
                tmp_cost = glaph[now][i] + cost[now]
                if cost[i] > tmp_cost:
                    cost[i] = tmp_cost
                    prev[i] = now
            if min > cost[i]:
                min = cost[i]
                next = i
        if next == -1: break
        now = next
    return [get_path(start, goal, prev), cost[goal]]
