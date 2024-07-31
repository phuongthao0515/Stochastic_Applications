from typing import List, Tuple
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from networkx.drawing.nx_pydot import graphviz_layout

class Edge:
    def __init__(self, from_node, to, capacity, cost):
        self.from_node = from_node
        self.to = to
        self.capacity = capacity
        self.cost = cost

def read_graph(file):
    with open(file, 'r') as f:
        line = f.readline().strip().split()
        nodes, edges, people = map(int, line)

        start_nodes = []
        end_nodes = []
        capacities = []
        costs = []

        for _ in range(edges):
            line = f.readline().strip().split()
            start, end, capacity, cost = map(int, line)
            start_nodes.append(start)
            end_nodes.append(end)
            capacities.append(capacity)
            costs.append(cost)
    #tự return thêm people sau nha
    return start_nodes, end_nodes, capacities, costs, nodes

def create_graph_with_vector(start_nodes, end_nodes, capacities, costs,nodes, vector):
    G = nx.Graph()
    for start, end, capacity, cost in zip(start_nodes, end_nodes, capacities, costs):
        G.add_edge(start, end, capacity=capacity, cost=cost)

    plt.figure(figsize=(8,8))
    pos = nx.kamada_kawai_layout(G, scale = 0.5)
    # Flip the graph so that source node is on the left
    pos = {node: (-pos[node][0], pos[node][1]) for node in G.nodes}
    # Specify color mapping for nodes
    node_colors = ['red' if node == 1 else 'yellow' if node == nodes else 'skyblue' for node in G.nodes]
    # Change vector format:
    vector_new = []
    for j in range(len(vector)):
        for i in range(len(vector[j])):
            if i == len(vector[j]) - 1:
                break
            vector_new.append((vector[j][i], vector[j][i+1]))
    # Define graph format
    options = {
    'node_color': node_colors,
    'node_size': 700,
    'width': 2,
    'arrowstyle': '-|>',
    'arrowsize': 12,
    'edge_color': ['yellow' if edge in vector_new else 'gray' for edge in G.edges()],
    }
    nx.draw_networkx(G, pos, arrows=True, **options)
    plt.show()

def shortest_paths(n, v0, adj, capacity, cost):
    INF = float('inf')
    d = [INF] * n
    d[v0] = 0
    inq = [False] * n
    q = deque([v0])
    p = [-1] * n

    while q:
        u = q.popleft()
        inq[u] = False
        for v in adj[u]:
            if capacity[u][v] > 0 and d[v] > d[u] + cost[u][v]:
                d[v] = d[u] + cost[u][v]
                p[v] = u
                if not inq[v]:
                    inq[v] = True
                    q.append(v)

    return d, p

def min_cost_flow(N,adj, cost, capacity, edges, K, s, t):
    for e in edges:
      cost[e.to][e.from_node] = -e.cost
    flow = 0
    total_cost = 0
    d, p = [], []
    path = []

    while flow < K:
        d, p = shortest_paths(N, s, adj, capacity, cost)
        if d[t] == float('inf'):
            break
        vector_temp = []
        # find max flow on that path
        f = K - flow
        cur = t
        while cur != s:
            f = min(f, capacity[p[cur]][cur])
            cur = p[cur]

        # apply flow
        flow += f
        total_cost += f * d[t]
        cur = t
        vector_temp.append(cur+1)
        while cur != s:
            vector_temp.append(p[cur] + 1)
            capacity[p[cur]][cur] -= f
            capacity[cur][p[cur]] += f
            cur = p[cur]
        path.append(vector_temp)
    return total_cost, path, flow

#SIMULATE DATA
def random_capacity_and_time_for_4_scenerio( capacity , cost):
    #capacity_sce_1 = [ np.random.randint(0, capacity[]) * n for _ in range(n)]
    # scenerio 1
    cost_sce_1 = [[0] * n for _ in range(n)]
    capacity_sce_1 = [[0] * n for _ in range(n)]
    # scenerio 2
    cost_sce_2 = [[0] * n for _ in range(n)]
    capacity_sce_2 = [[0] * n for _ in range(n)]
    # scenerio 3
    cost_sce_3 = [[0] * n for _ in range(n)]
    capacity_sce_3 = [[0] * n for _ in range(n)]
    # scenerio 4
    cost_sce_4 = [[0] * n for _ in range(n)]
    capacity_sce_4 = [[0] * n for _ in range(n)]
    # random scenrio 1
    for i in range (0,n):
      for j in range (0,n):
        if (capacity[i][j] > 0):
          capacity_sce_1[i][j] = np.random.randint(0, capacity[i][j])
        if (cost[i][j] > 0):
          cost_sce_1[i][j] = np.random.randint(cost[i][j], 50)
    # random scenrio 2
    for i in range (0,n):
      for j in range (0,n):
        if (capacity[i][j] > 0):
          capacity_sce_2[i][j] = np.random.randint(0, capacity[i][j])
        if (cost[i][j] > 0):
          cost_sce_2[i][j] = np.random.randint(cost[i][j], 50)
    # random scenrio 3
    for i in range (0,n):
      for j in range (0,n):
        if (capacity[i][j] > 0):
          capacity_sce_3[i][j] = np.random.randint(0, capacity[i][j])
        if (cost[i][j] > 0):
          cost_sce_3[i][j] = np.random.randint(cost[i][j], 50)
    # random scenrio 4
    for i in range (0,n):
      for j in range (0,n):
        if (capacity[i][j] > 0):
          capacity_sce_4[i][j] = np.random.randint(0, capacity[i][j])
        if (cost[i][j] > 0):
          cost_sce_4[i][j] = np.random.randint(cost[i][j], 50)
    return capacity_sce_1, cost_sce_1, capacity_sce_2, cost_sce_2, capacity_sce_3, cost_sce_3, capacity_sce_4, cost_sce_4

def adaptive_plan( path, cost, capacity, k):
  middle_node = [1] * len(path)
  middle_flow = [500] * len(path)
  for i in range(0, len(path)):
     temp_time  = 0
     for j in range(len(path[i])-1, 0, -1):
        #print( path[i][j] )
        if (temp_time + cost[ path[i][j] - 1 ][ path[i][j-1] - 1 ] < 60):
           temp_time = temp_time + cost[ path[i][j] - 1 ][ path[i][j-1] - 1 ]
           #print("Temp time: ", temp_time)
           middle_node[i] = path[i][j - 1]
           #print("Middle node: ", middle_node[i])
           if ( capacity[ path[i][j] - 1 ][ path[i][j-1] - 1 ] > 0 and capacity[ path[i][j] - 1 ][ path[i][j-1] - 1 ] < middle_flow[i]):
              middle_flow[i] = capacity[ path[i][j] - 1 ][ path[i][j-1] - 1 ]
              #print("Middle flow: ", middle_flow[i])
        elif ( temp_time + cost[ path[i][j] - 1 ][ path[i][j-1] - 1 ] == 60):
           middle_node[i] = path[i][j]
           if ( capacity[ path[i][j] - 1 ][ path[i][j-1] - 1 ] > 0 and capacity[ path[i][j] - 1 ][ path[i][j-1] - 1 ] < middle_flow[i]):
              middle_flow[i] = capacity[ path[i][j] - 1 ][ path[i][j-1] - 1 ]
           break
        elif (  temp_time + cost[ path[i][j] - 1 ][ path[i][j-1] - 1 ] > 60):
           break

  # Check middle flow
  sum = 0
  for i in range (0, len(middle_flow) - 1):
       sum = sum + middle_flow[i]
  if ( k - sum < middle_flow[ len(middle_flow) - 1 ] ):
     middle_flow[ len(middle_flow) - 1 ] = k - sum
  sum = sum + middle_flow[ len(middle_flow) - 1 ]
  return middle_node, middle_flow, sum

def create_super_source( adj, middle_node):
   adj[0] = []
   for i in middle_node:
        adj[0].append(i-1)
        adj[i-1].append(0)
   return adj

def change_capacity_cost (n, middle_flow, middle_node, capacity, cost ):
    for i in range(0, n):
      capacity[0][i] = 0

    for i in range (0, len(middle_node)):
       capacity[0][ middle_node[i] - 1] = middle_flow[i]
       cost[0][ middle_node[i] - 1] = 0
    return capacity, cost



if __name__ == "__main__":
    file_path = "D:\\pt.txt"  # Replace with the path to your file
    with open(file_path, 'r') as file:
        n, m, k = map(int, file.readline().split())
        edges = []
# Using iteration over lines
        for line in file:
            u, v, c, w = map(int, line.split())
            edges.append(Edge(u - 1, v - 1, c, w))
# Put value into initial matrix
    adj = [[] for _ in range(n)]
    cost = [[0] * n for _ in range(n)]
    capacity = [[0] * n for _ in range(n)]
    for e in edges:
        adj[e.from_node].append(e.to)
        adj[e.to].append(e.from_node)
        cost[e.from_node][e.to] = e.cost
        capacity[e.from_node][e.to] = e.capacity
# Draw the initial graph:
    start_nodes, end_nodes, a, b, number_of_nodes = read_graph('D:\\pt.txt')
    vector = []
    print("\n The initial graph for the evacuation plan: ")
    create_graph_with_vector(start_nodes, end_nodes, a, b, number_of_nodes, vector)
# Generate random value for 4 scenerio
    random_values = np.random.rand(4)
    probabilities = random_values / np.sum(random_values)
    # scenerio 1
    cost_sce_1 = [[0] * n for _ in range(n)]
    capacity_sce_1 = [[0] * n for _ in range(n)]
    # scenerio 2
    cost_sce_2 = [[0] * n for _ in range(n)]
    capacity_sce_2 = [[0] * n for _ in range(n)]
    # scenerio 3
    cost_sce_3 = [[0] * n for _ in range(n)]
    capacity_sce_3 = [[0] * n for _ in range(n)]
    # scenerio 4
    cost_sce_4 = [[0] * n for _ in range(n)]
    capacity_sce_4 = [[0] * n for _ in range(n)]
    # random value for scenerio
    capacity_sce_1, cost_sce_1, capacity_sce_2, cost_sce_2, capacity_sce_3, cost_sce_3, capacity_sce_4, cost_sce_4 = random_capacity_and_time_for_4_scenerio(capacity, cost)
    # Print
    print ("\nIntial capacity:")
    for row in capacity:
        print(row)
    print ("\nInitial cost:")
    for row in cost:
        print(row)
    print ("\nNew capacity for scenerio 1: ")
    for row in capacity_sce_1:
        print(row)
    print ("\nNew cost for scenerio 1: ")
    for row in cost_sce_1:
        print(row)
    print ("\nNew capacity for scenerio 2: ")
    for row in capacity_sce_2:
        print(row)
    print ("\nNew cost for scenerio 2: ")
    for row in cost_sce_2:
        print(row)
    print ("\n New capacity for scenerio 3: ")
    for row in capacity_sce_3:
        print(row)
    print ("\nNew cost for scenerio 3: ")
    for row in cost_sce_3:
        print(row)
    print ("\nNew capacity for scenerio 4: ")
    for row in capacity_sce_4:
        print(row)
    print ("\nNew cost for scenerio 4: ")
    for row in cost_sce_4:
        print(row)

# Calculate average value base on prob, n is number of node
    cost_ave = [[0] * n for _ in range(n)]
    capacity_ave = [[0] * n for _ in range(n)]

    for i in range (0,n):
      for j in range (0, n):
        cost_ave[i][j] = probabilities[0]*cost_sce_1[i][j] + probabilities[1]*cost_sce_2[i][j] + probabilities[2]*cost_sce_3[i][j] + probabilities[3]*cost_sce_4[i][j]
        capacity_ave[i][j] = probabilities[0]*capacity_sce_1[i][j] + probabilities[1]*capacity_sce_2[i][j] + probabilities[2]*capacity_sce_3[i][j] + probabilities[3]*capacity_sce_4[i][j]
        cost_ave[i][j] = round(cost_ave[i][j])
        capacity_ave[i][j] = round(capacity_ave[i][j])

        capacity_ave_initial = capacity_ave
    print("\nThe average capacity matrix: ")
    for row in capacity_ave:
        print(row)
    print( "\nThe average cost matrix: ")
    for row in cost_ave:
        print(row)
    path = []
    result, path, flow = min_cost_flow(n, adj, cost_ave, capacity_ave, edges, k, 0, n - 1)

    # Print the initial evacuation plan used for all scenerio
    print("\n Initial evacuation plan for all scenerio: ")
    for row in path:
        print("  +", row)
    print("\nThe maximum number of people can be evacuted: ", flow, "\n")

    # Draw the graph for initial evacution plan
    reverse_path = [ row[::-1] for row in path]
    print( "\nThe initial evacuation plan is marked as the yellow line on the graph: ")
    create_graph_with_vector(start_nodes, end_nodes, a, b, number_of_nodes, reverse_path)

    middle_node, middle_flow, sum = adaptive_plan( path, cost, capacity, k)
    # create super source
    adj_2 = create_super_source( adj, middle_node)

    capacity_sce_1, cost_sce_1 = change_capacity_cost( n, middle_flow, middle_node, capacity_sce_1, cost_sce_1)
    capacity_sce_2, cost_sce_2 = change_capacity_cost( n, middle_flow, middle_node, capacity_sce_2, cost_sce_2)
    capacity_sce_3, cost_sce_3 = change_capacity_cost( n, middle_flow, middle_node, capacity_sce_3, cost_sce_3)
    capacity_sce_4, cost_sce_4 = change_capacity_cost( n, middle_flow, middle_node, capacity_sce_4, cost_sce_4)


    result_sce_1, path_sce_1, flow_sce_1 = min_cost_flow(n, adj_2, cost_sce_1, capacity_sce_1, edges, sum, 0, n - 1)
    result_sce_2, path_sce_2, flow_sce_2 = min_cost_flow(n, adj_2, cost_sce_2, capacity_sce_2, edges, sum, 0, n - 1)
    result_sce_3, path_sce_3, flow_sce_3 = min_cost_flow(n, adj_2, cost_sce_3, capacity_sce_3, edges, sum, 0, n - 1)
    result_sce_4, path_sce_4, flow_sce_4 = min_cost_flow(n, adj_2, cost_sce_4, capacity_sce_4, edges, sum, 0, n - 1)

    print("\nNew evacuation plan for secnerio 1: ", path_sce_1)
    print("The maximum number of people can be evacuated in scenerio 1: ", flow_sce_1)
    print("\nNew evacuation plan for secnerio 2: ", path_sce_2)
    print("The maximum number of people can be evacuated in scenerio 2: ", flow_sce_2)
    print("\nNew evacuation plan for secnerio 3: ", path_sce_3)
    print("The maximum number of people can be evacuated in scenerio 3: ", flow_sce_3)
    print("\nNew evacuation plan for secnerio 4: ", path_sce_4)
    print("The maximum number of people can be evacuated in scenerio 4: ", flow_sce_4)
