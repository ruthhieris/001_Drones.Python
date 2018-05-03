
# coding: utf-8

# # Probabilistic Roadmap
# 
# 
# In this notebook you'll expand on previous random sampling exercises by creating a graph from the points and running A*.
# 
# 1. Load the obstacle map data
# 2. Sample nodes (use KDTrees here)
# 3. Connect nodes (use KDTrees here)
# 4. Visualize graph
# 5. Define heuristic
# 6. Define search method
# 7. Execute and visualize
# 
# We'll load the data for you and provide a template for visualization.

# In[1]:


# Again, ugly but we need the latest version of networkx!
# This sometimes fails for unknown reasons, please just 
# "reset and clear output" from the "Kernel" menu above 
# and try again!
import sys
get_ipython().system('{sys.executable} -m pip install -I networkx==2.1')
import pkg_resources
pkg_resources.require("networkx==2.1")
import networkx as nx


# In[2]:


nx.__version__ # should be 2.1


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from queue import PriorityQueue

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


plt.rcParams['figure.figsize'] = 14, 14


# ## Step 1 - Load Data

# In[5]:


# This is the same obstacle data from the previous lesson.
filename = 'colliders.csv'
data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
print(data)


# ## Step 2 - Sample Points
# 
# 
# You may want to limit the z-axis values.

# In[70]:


from sampling import Sampler

# TODO: sample points randomly
# then use KDTree to find nearest neighbor polygon
# and test for collision
num_samp = 500
sampler = Sampler(data)
polygons = sampler._polygons
nodes = sampler.sample(num_samp)
print(len(nodes))


# ## Step 3 - Connect Nodes
# 
# Now we have to connect the nodes. There are many ways they might be done, it's completely up to you. The only restriction being no edge connecting two nodes may pass through an obstacle.
# 
# NOTE: You can use `LineString()` from the `shapely` library to create a line. Additionally, `shapely` geometry objects have a method `.crosses` which return `True` if the geometries cross paths, for instance your `LineString()` with an obstacle `Polygon()`!

# In[71]:


# TODO: connect nodes
# Suggested method
    # 1) cast nodes into a graph called "g" using networkx

    # 2) write a method "can_connect()" that:
        # casts two points as a shapely LineString() object
        # tests for collision with a shapely Polygon() object
        # returns True if connection is possible, False otherwise
def can_connect(p1,p2):
    line = LineString([tuple(p1),tuple(p2)])
    for p in polygons:
        if p.crosses(line) and p.height >= min(p1[2], p2[2]):
            return False
    return True

    # 3) write a method "create_graph()" that:
        # defines a networkx graph as g = Graph()
        # defines a tree = KDTree(nodes)
        # test for connectivity between each node and 
            # k of it's nearest neighbors
        # if nodes are connectable, add an edge to graph
def create_graph(nodes,k):
    g = nx.Graph()
    tree = neighbors.KDTree(nodes)
    for n1 in nodes:
        ind = tree.query([n1], k, return_distance = False)[0]
        #print(ind)
        for j in ind:
            n2 = nodes[j]
            #print(n2)
            #print(j)
            if n2 == n1:
                continue
                
            if can_connect(n1, n2):
                dist = np.linalg.norm(np.array(n1) - np.array(n2))
                g.add_edge(n1, n2, weight = dist)
    return g
    # Iterate through all candidate nodes!

import time
t0 = time.time()
g = create_graph(nodes, 10)
print('graph took {0} seconds to build'.format(time.time()-t0))
print("Number of edges", len(g.edges))


# ## Step 4 - Visualize Graph

# In[72]:


# Create a grid map of the world
from grid import create_grid
# This will create a grid map at 1 m above ground level
grid = create_grid(data, sampler._zmax, 1)

fig = plt.figure()

plt.imshow(grid, cmap='Greys', origin='lower')

nmin = np.min(data[:, 0] - data[:, 3])
emin = np.min(data[:, 1] - data[:, 4])

# If you have a graph called "g" these plots should work
# Draw edges
for (n1, n2) in g.edges:
    plt.plot([n1[1] - emin, n2[1] - emin], [n1[0] - nmin, n2[0] - nmin], 'black' , alpha=0.5)

# Draw all nodes connected or not in blue
for n1 in nodes:
    plt.scatter(n1[1] - emin, n1[0] - nmin, c='blue')
    
# Draw connected nodes in red
for n1 in g.nodes:
    plt.scatter(n1[1] - emin, n1[0] - nmin, c='red')
    


plt.xlabel('NORTH')
plt.ylabel('EAST')

plt.show()


# ## Step 5 - Define Heuristic

# In[73]:


def heuristic(n1, n2):
    # TODO: complete
    return np.linalg.norm(np.array(n1) - np.array(n2))


# ## Step 6 - Complete A*

# In[74]:


def a_star_graph(graph, h, start, goal):
    """
    Modified A* to work with NetworkX graphs.
    path, cost = a_star_graph(networkx.Graph(), heuristic_func,
                tuple(skel_start), tuple(skel_goal))
    INPUT: start, goal = tuple(x,y)
    """
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)
    #print(len(visited), start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        current_cost = item[0]
        
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                #print(next_node)
                cost = graph.edges[current_node, next_node]['weight']
                branch_cost = current_cost + cost
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node)
                    queue.put((queue_cost, next_node))
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost

   
start = list(g.nodes)[0]
k = np.random.randint(len(g.nodes))
print(k, len(g.nodes))
goal = list(g.nodes)[k]

path, cost = a_star_graph(g, heuristic, start, goal)
print(len(path), path, cost)
print()

path_pairs = zip(path[:-1], path[1:])
#for (n1, n2) in path_pairs:
    #print(n1, n2)


# ## Step 7 - Visualize Path

# In[75]:


fig = plt.figure()

plt.imshow(grid, cmap='Greys', origin='lower')

nmin = np.min(data[:, 0] - data[:, 3])
emin = np.min(data[:, 1] - data[:, 4])

# draw nodes
for n1 in g.nodes:
    plt.scatter(n1[1] - emin, n1[0] - nmin, c='red')
    
# draw edges
for (n1, n2) in g.edges:
    plt.plot([n1[1] - emin, n2[1] - emin], [n1[0] - nmin, n2[0] - nmin], 'grey')
    
# TODO: add code to visualize the path
path_pairs = zip(path[:-1], path[1:])
for (n1, n2) in path_pairs:
    plt.plot([n1[1] - emin, n2[1] - emin], [n1[0] - nmin, n2[0] - nmin], 'green')
plt.plot(start[1]-emin, start[0]-nmin, 'bv')
plt.plot(goal[1]-emin, goal[0]-nmin, 'bx')

plt.xlabel('NORTH')
plt.ylabel('EAST')

plt.show()


# [solution](/notebooks/Probabilistic-Roadmap-Solution.ipynb)

# In[13]:



