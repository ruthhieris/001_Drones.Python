
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


#from sampling import Sampler

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

nmin = np.min(data[:, 0])
emin = np.min(data[:, 1])

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

nmin = np.min(data[:, 0])
emin = np.min(data[:, 1])

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


import numpy as np
from shapely.geometry import Polygon, Point
from sklearn import neighbors

class Sampler():
    
    def __init__(self, data, zlim = 10):
        self._zmax = zlim
        self._polygons = self.extract_polygons(data)
        self.__d = data
        
    @property
    def _zmax(self):
        return self.__zmax
    
    @_zmax.setter
    def _zmax(self,zlim):
        if zlim < 0:
            self.__zmax = 0
        else:
            self.__zmax = zlim
        
    def sample(self,num_samp):
        """
        sampling points and removing
        ones conflicting with obstacles.
        """
        nodes = self.random_sample(self.__d, self._zmax, num_samp, True)
        tree = self.KDTree_from_poly(self._polygons)
        to_keep_tree = []
        for point in nodes:
            if not self.collides_tree(tree, self._polygons, point):
                to_keep_tree.append(point)
                
        return to_keep_tree

    # In[31]:
    
    @staticmethod
    def collides_tree(tree, polygons, point):  
        """
        Determine whether the point collides with any obstacles
        Input: KDTree, polygons, random_node
        Output: True or False
        """
        dist,ind = tree.query(np.asarray(point[:2]).reshape(1,2),k=3)
        collision = False
        for j in range(ind.shape[1]):
            pnum = ind[0][j]
            (p,height) = polygons[pnum].poly
            if p.contains(Point(point)) and height >= point[2]:
                collision = True
                break
        return collision

    # In[32]:
    
    @staticmethod
    def KDTree_from_poly(polygons, debug = False):
        center = [np.asarray(pol.p.centroid) for pol in polygons]
        if debug:
            print(center[:10], np.asarray(center).shape)
        tree = neighbors.KDTree(center,leaf_size = 40)
        return tree

            
    # In[33]:
    
    @staticmethod
    def random_sample(data, z_lim, num_samples = 200, explicit = True):
    # # Sampling 3D Points
    # 
    # Now that we have the extracted the polygons, we need to sample random 3D points. 
    #Currently we don't know suitable ranges for x, y, and z. 
    #Let's figure out the max and min values for each dimension.
    
        xmin = np.min(data[:, 0] - data[:, 3])
        xmax = np.max(data[:, 0] + data[:, 3])
        
        ymin = np.min(data[:, 1] - data[:, 4])
        ymax = np.max(data[:, 1] + data[:, 4])
        
        zmin = 0
        # Limit the z axis for the visualization
        zmax = z_lim #np.max(data[:,2] + data[:,5] + 10) #10
        
        if explicit:
            print("X")
            print("min = {0}, max = {1}\n".format(xmin, xmax))
            
            print("Y")
            print("min = {0}, max = {1}\n".format(ymin, ymax))
            
            print("Z")
            print("min = {0}, max = {1}".format(zmin, zmax))
            
        # Next, it's time to sample points. All that's left is picking the 
        #distribution and number of samples. The uniform distribution makes 
        #sense in this situation since we we'd like to encourage searching the whole space.
            
        #np.random.seed(0)
        xvals = np.random.uniform(xmin, xmax, num_samples)
        yvals = np.random.uniform(ymin, ymax, num_samples)
        zvals = np.random.uniform(zmin, zmax, num_samples)
        
        return list(zip(xvals, yvals, zvals))


    @staticmethod
    def extract_polygons(data):
        polygons = []
        for i in range(data.shape[0]):
            north, east, alt, d_north, d_east, d_alt = data[i, :]
            
            # TODO: Extract the 4 corners of the obstacle
            # 
            # NOTE: The order of the points matters since
            # `shapely` draws the sequentially from point to point.
            #
            # If the area of the polygon is 0 you've likely got a weird
            # order.
            p1 = (north + d_north, east - d_east)
            p2 = (north + d_north, east + d_east)
            p3 = (north - d_north, east + d_east)
            p4 = (north - d_north, east - d_east)
            corners = [p1, p2, p3, p4]
            
            # TODO: Compute the height of the polygon
            height = alt + d_alt
    
            # TODO: Once you've defined corners, define polygons
            polygons.append(Poly(corners, height))
    
        return polygons


# In[32]:
# ## Create Polygons
class Poly():
    
    def __init__(self, corners, height):
        self.p = Polygon(corners)
        self.height = height
        
        self.poly = (self.p, self.height)
        
    def __str__(self):
        return '(' + str(self.p) + ',' + str(self.height) + ')'
    
    def crosses(self, line):
        """shapely geometry objects have a method .crosses which return 
        True if the geometries cross paths.
        Use LineString to create a line. """
        return self.p.crosses(line)

