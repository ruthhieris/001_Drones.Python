
# coding: utf-8

# # Receding Horizon
# 
# This notebook is your playground to pull together techniques from the previous lessons! A solution here can be built from previous solutions (more or less) so we will offer no solution notebook this time.
# 
# Here's a suggested approach:
# 
# 1. Load the colliders data
# 2. Discretize your search space into a grid or graph
# 3. Define a start and goal location
# 4. Find a coarse 2D plan from start to goal
# 5. Choose a location along that plan and discretize
#    a local volume around that location (for example, you
#    might try a 40x40 m area that is 10 m high discretized
#    into 1m^3 voxels)
# 6. Define your goal in the local volume to a a node or voxel
#    at the edge of the volume in the direction of the next
#    waypoint in your coarse global plan.
# 7. Plan a path through your 3D grid or graph to that node
#    or voxel at the edge of the local volume.  
# 
# We'll import some of the routines from previous exercises that you might find useful here.  

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# Grid creation routine
from grid import create_grid
# Voxel map creation routine
from voxmap import create_voxmap
# 2D A* planning routine (can you convert to 3D??)
from planning import a_star
# Random sampling routine
#from sampling import Sampler
from shapely.geometry import Polygon, Point, LineString
from queue import PriorityQueue
from scipy.spatial import Voronoi, voronoi_plot_2d
import math

get_ipython().run_line_magic('matplotlib', 'inline')

import sys
get_ipython().system('{sys.executable} -m pip install -I networkx==2.1')
import pkg_resources
pkg_resources.require("networkx==2.1")
import networkx as nx
nx.__version__ # should be 2.1


# In[2]:


plt.rcParams['figure.figsize'] = 14, 14


# ## Load Data

# In[3]:


# This is the same obstacle data from the previous lesson.
filename = 'colliders.csv'
data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
print(data)


# In[4]:


def heuristic(n1, n2):
    """
    Euclidian heuristic
    """
    return np.linalg.norm(np.array(n1) - np.array(n2))


# In[87]:


import numpy as np
from shapely.geometry import Polygon, Point, LineString
from sklearn import neighbors
from mpl_toolkits.mplot3d import Axes3D

class LocVolume():
    """
    discretize a local volume around the location (for example, you might try 
    a 40x40 m area that is 10 m high discretized into 1m^3 voxels)
    Input: data, tuple_center(North,East,Z) d_n, d_e, d_z, voxel_size(default = 1)
    Output = Prism(Polygon + height)
    """
    
    def __init__(self, data, center, d_n, d_e, d_z, voxel_size = 1):
        self._vol = voxel_size
        self._c = center #(tuple(N,E,Z))
        self._height = self.prismheight(center, d_z/2) #tuple(z_bottom, z_top)
        self._corners = self.basecorners(data, center, d_n/2, d_e/2) #([p1,p2,p3,p4])
        self._locPrism = Prism(self._corners, self._height)
        self._nmin = self._corners[0][0]
        self._nmax = self._corners[2][0]
        self._emin = self._corners[0][1]
        self._emax = self._corners[2][1]
        self.vox_sides()
    
    def basecorners(self, data, center, dn, de):
        nmin, nmax, emin, emax = Sampler.datalimits(data)
        print('datalimits',nmin, nmax, emin, emax)
        p1 = tuple(np.clip((center[0] - dn, center[1] - de, center[2]),(0,0,0), (nmax-nmin, emax-emin, self._height[1])))
        p2 = tuple(np.clip((center[0] + dn, center[1] - de, center[2]),(0,0,0), (nmax-nmin, emax-emin, self._height[1])))
        p3 = tuple(np.clip((center[0] + dn, center[1] + de, center[2]),(0,0,0), (nmax-nmin, emax-emin, self._height[1])))
        p4 = tuple(np.clip((center[0] - dn, center[1] + de, center[2]),(0,0,0), (nmax-nmin, emax-emin, self._height[1])))
        #print('cen',center,'p1',p1,'p2',p2,'p3',p3,'p4',p4)
        return [p1, p2, p3, p4]
    
    def prismheight(self, center, dz, zmin = 0):
        """
        Output: tuple (z_bottom, z_top)
        """
        zmax = center[2] + 2*dz
        z = tuple(np.clip((center[2] - dz, center[2] + dz), zmin, zmax))
        return z
    
    def localgoal(self, path, ind):
        p1 = self._c
        loc_g = []
        #print(len(path))
        while not loc_g and ind+1 < len(path):
            #print(path[ind+1],p1)
            if len(path[ind+1]) < 3:
                p2 = (path[ind+1][0],path[ind+1][1],p1[2])
            else:
                p2 = path[ind+1]
            #print('p1,p2',p1,p2)
            #print(ind, p1, p2)
            if self._locPrism.intersects(LineString([p1,p2])):
                print('yes')
                loc_g = self._locPrism.intersection(LineString([p1,p2]))
                break
            else:
                p1 = p2
                ind += 1
    
        if loc_g ==[]:
            print("Failed to find a new local goal!")
        else:
            loc_g = np.asarray(loc_g)[1] - [self._nmin, self._emin, 0]
            #print('Voxel',self._locPrism)
            #print('Line',LineString([p1,p2]))
            print('Local Goal in voxel', loc_g, 'ind',ind)
            
        goal_voxmap = self.vox_plot(loc_g)
        return loc_g, ind, goal_voxmap

    def vox_plot(self, loc_g, convert = False):
        """
        Input: Point in Voxel frame
        'convert = True' => convert coordinates from global frame
        """
        if convert:
            loc_g = np.asarray(loc_g) - [self._nmin, self._emin, 0]
        goal_voxmap = self.voxmap_blank()

        goal_vox = [
            int(np.clip(loc_g[0]//self._vol, 0, self._north_size-1)),
            int(np.clip(loc_g[1]//self._vol, 0, self._east_size-1)),
            int(np.clip(loc_g[2]//self._vol, self._loc_zmin, self._loc_zmax-1))
        ]
        goal_voxmap[goal_vox[0],goal_vox[1],goal_vox[2]] = True
        
        return goal_voxmap
        
    def vox_sides(self):
        self._north_size = int(np.ceil((self._nmax - self._nmin))) // self._vol
        self._east_size = int(np.ceil((self._emax - self._emin))) // self._vol
        self._loc_zmax = int(self._height[1]) // self._vol 
        self._loc_zmin = int(self._height[0]) // self._vol 
        
    def voxmap_blank(self, z_above = 10):
        """blank vor voxel 3d mpl"""
        alt_size = int(self._height[1] + z_above) // self._vol    
        map_blank = np.zeros((self._north_size, self._east_size, alt_size), dtype=np.bool)
        
        return map_blank
    
    @staticmethod
    def create_voxmap(data, voxel_size=5):
        """
        Returns a grid representation of a 3D configuration space
        based on given obstacle data.
        
        The `voxel_size` argument sets the resolution of the voxel map. 
        """
    
        # minimum and maximum north coordinates
        north_min = np.floor(np.amin(data[:, 0] - data[:, 3]))
        north_max = np.ceil(np.amax(data[:, 0] + data[:, 3]))
    
        # minimum and maximum east coordinates
        east_min = np.floor(np.amin(data[:, 1] - data[:, 4]))
        east_max = np.ceil(np.amax(data[:, 1] + data[:, 4]))
    
        alt_max = np.ceil(np.amax(data[:, 2] + data[:, 5]))
        alt_min = 0
        
        print("N")
        print("min = {0}, max = {1}\n".format(north_min, north_max))
        
        print("E")
        print("min = {0}, max = {1}\n".format(east_min, east_max))
        
        print("Z")
        print("min = {0}, max = {1}".format(alt_min, alt_max))
        print()
        
        # given the minimum and maximum coordinates we can
        # calculate the size of the grid.
        north_size = int(np.ceil((north_max - north_min))) // voxel_size
        east_size = int(np.ceil((east_max - east_min))) // voxel_size
        alt_size = int(alt_max) // voxel_size
    
        voxmap = np.zeros((north_size, east_size, alt_size), dtype=np.bool)
        
        # Given an interval, values outside the interval are clipped to the interval 
        # edges. For example, if an interval of [0, 1] is specified, values smaller 
        # than 0 become 0, and values larger than 1 become 1
        for i in range(data.shape[0]):
            north, east, alt, d_north, d_east, d_alt = data[i,:]
            obstacle = [
                    int(np.clip((north - d_north - north_min)// voxel_size, 0, north_size-1)),
                    int(np.clip((north + d_north - north_min)// voxel_size, 0, north_size-1)),
                    int(np.clip((east - d_east - east_min)// voxel_size, 0, east_size-1)),
                    int(np.clip((east + d_east - east_min)// voxel_size, 0, east_size-1)),
                    int(alt + d_alt)//voxel_size
                ]
            voxmap[obstacle[0]:obstacle[1], obstacle[2]:obstacle[3],
                    0:obstacle[4]] = True
    
            # TODO: fill in the voxels that are part of an obstacle with `True`
            #
            # i.e. grid[0:5, 20:26, 2:7] = True
    
        return voxmap
    
    @staticmethod
    def databoundary(data, explicit = False):
        # minimum and maximum north coordinates
        north_min = np.floor(np.amin(data[:, 0] - data[:, 3]))
        north_max = np.ceil(np.amax(data[:, 0] + data[:, 3]))
    
        # minimum and maximum east coordinates
        east_min = np.floor(np.amin(data[:, 1] - data[:, 4]))
        east_max = np.ceil(np.amax(data[:, 1] + data[:, 4]))
    
        alt_max = np.ceil(np.amax(data[:, 2] + data[:, 5]))
        alt_min = np.floor(np.amin(data[:, 2] - data[:, 5]))
        
        if explicit:
            print("Data boundaries")
            print("N")
            print("min = {0}, max = {1}\n".format(north_min, north_max))

            print("E")
            print("min = {0}, max = {1}\n".format(east_min, east_max))

            print("Z")
            print("min = {0}, max = {1}\n".format(alt_min, alt_max))
            
        return north_min, north_max, east_min, east_max, alt_min, alt_max
        
    def create_localvoxmap(self, data, explicit = True):
        """
        Returns a grid representation of a 3D configuration space inside 
        local Voxel based on given obstacle data.
        The `voxel_size` argument sets the resolution of the voxel map. 
        """
        #Voxel data boundary in data-[nmin,emin] format
        nmin, nmax, emin, emax, alt_min, alt_max = self.databoundary(data)
        zmin, zmax = self._height
        # given the minimum and maximum coordinates we can
        # calculate the size of the grid.
    
        loc_voxmap = self.voxmap_blank()

        if explicit:
            print('#localmapposition')
            print("N")
            print("min = {0}, max = {1}\n".format(self._nmin, self._nmax))

            print("E")
            print("min = {0}, max = {1}\n".format(self._emin, self._emax))

            print("Z")
            print("min = {0}, max = {1}\n".format(self._height[0], self._height[1]))
        
        for i in range(data.shape[0]):
            north, east, alt, d_north, d_east, d_alt = data[i,:]
            obstacle = [
                    int(np.clip((north - d_north - nmin - self._nmin)// self._vol, 0, self._north_size)),
                    int(np.clip((north + d_north - nmin - self._nmin)// self._vol, 0, self._north_size)),
                    int(np.clip((east - d_east - emin - self._emin)// self._vol, 0, self._east_size)),
                    int(np.clip((east + d_east - emin - self._emin)// self._vol, 0, self._east_size)),
                    int(np.clip((alt + d_alt)//self._vol, self._loc_zmin, self._loc_zmax))
                    ]
            if (obstacle[1]-obstacle[0]) and (obstacle[3] - obstacle[2]):
                #print('obst',obstacle)
                loc_voxmap[obstacle[0]:obstacle[1], obstacle[2]:obstacle[3],
                        0:obstacle[4]] = True
    
        return loc_voxmap
                

class Sampler():
    
    def __init__(self, data, zlim = 10, safety_distance = 0):
        self._zmax = zlim
        self._polygons = self.extract_polygons(data, safety_distance)
        self.__d = data
        self.NElimits(data)
        
    @property
    def _zmax(self):
        return self.__zmax
    
    @_zmax.setter
    def _zmax(self,zlim):
        if zlim < 0:
            self.__zmax = 0
        else:
            self.__zmax = zlim
            
    def NElimits(self,data):
        self._nmin = self.datalimits(data)[0]
        self._nmax = self.datalimits(data)[1]
        
        self._emin = self.datalimits(data)[2]
        self._emax = self.datalimits(data)[3]
        self._zmin = 0
        
    @staticmethod
    def datalimits(data):
        """
        set data borders
        Input: data
        Output: (nmin, nmax, emin, emax)
        """
        nmin = np.min(data[:, 0] - data[:, 3])
        nmax = np.max(data[:, 0] + data[:, 3])
        emin = np.min(data[:, 1] - data[:, 4])
        emax = np.max(data[:, 1] + data[:, 4])
        return nmin, nmax, emin, emax
        
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
    
        nmin = np.min(data[:, 0] - data[:, 3])
        nmax = np.max(data[:, 0] + data[:, 3])
        
        emin = np.min(data[:, 1] - data[:, 4])
        emax = np.max(data[:, 1] + data[:, 4])
        
        zmin = 0
        # Limit the z axis for the visualization
        zmax = z_lim #np.max(data[:,2] + data[:,5] + 10) #10
        
        if explicit:
            print("N")
            print("min = {0}, max = {1}\n".format(nmin, nmax))
            
            print("E")
            print("min = {0}, max = {1}\n".format(emin, emax))
            
            print("Z")
            print("min = {0}, max = {1}".format(zmin, zmax))
            print()
            
        # Next, it's time to sample points. All that's left is picking the 
        #distribution and number of samples. The uniform distribution makes 
        #sense in this situation since we we'd like to encourage searching the whole space.
            
        #np.random.seed(0)
        nvals = np.random.uniform(nmin, nmax, num_samples)
        evals = np.random.uniform(emin, emax, num_samples)
        zvals = np.random.uniform(zmin, zmax, num_samples)
        
        return list(zip(nvals, evals, zvals))


    @staticmethod
    def extract_polygons(data, sdist = 0):
        """Polygons with or without safety_distance"""
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
            p1 = (north + d_north + sdist, east - d_east - sdist)
            p2 = (north + d_north + sdist, east + d_east + sdist)
            p3 = (north - d_north - sdist, east + d_east + sdist)
            p4 = (north - d_north - sdist, east - d_east - sdist)
            corners = [p1, p2, p3, p4]
            
            # TODO: Compute the height of the polygon
            height = alt + d_alt + sdist
    
            # TODO: Once you've defined corners, define polygons
            polygons.append(Prism(corners, height))
    
        return polygons


# In[32]:
# ## Create Polygons
class Prism():
    """
    3D structure Prism = 2D Polygon + height
    """
    
    def __init__(self, corners, height):
        self.p = Polygon(corners)
        self.height = height
        
        self.poly = (self.p, self.height)
        
    def __str__(self):
        return '(' + str(self.p) + ',' + str(self.height) + ')'
    
    def crosses(self, line):
        """
        shapely geometry objects have a method .crosses which return 
        True if the geometries cross paths.
        Input: line (from shapely.geometry import LineString)
                or list(tuple1, tuple2, tuple3...)
                or (tuple1, tuple2,...)
        if points = [tuple1, tuple]
        """
        #print('crosses', line, type(line))
        if not type(line) == LineString:
            #print(line, type(line))
            line = LineString(list(line))
        #coords = list(zip(*line)) #[(x1,x2),(y1,y2),(z1,z2)]
        return self.p.crosses(line)
    
    def intersects(self,line):
        #print('crosses', line, type(line))
        if not type(line) == LineString:
            #print(line, type(line))
            line = LineString(list(line))
        #coords = list(zip(*line)) #[(x1,x2),(y1,y2),(z1,z2)]
        return self.p.intersects(line)
        
    def touches(self,line):
        #print('crosses', line, type(line))
        if not type(line) == LineString:
            #print(line, type(line))
            line = LineString(list(line))
        #coords = list(zip(*line)) #[(x1,x2),(y1,y2),(z1,z2)]
        return self.p.touches(line)
    
    def bounds(self):
        """
        Returns a (minx, miny, maxx, maxy) tuple (float values) that bounds the object
        """
        return self.p.bounds
    
    def intersection(self, line):
        """
        Returns a representation of the intersection of this object with the other geometric object.
        """
        if not type(line) == LineString:
            line = LineString(list(line))
        return self.p.intersection(line)


# In[ ]:


flight_altitude = 7
safety_distance = 3
#Define a start and goal location
start_ne = Sampler.random_sample(data, flight_altitude, 1)
goal_ne = Sampler.random_sample(data, flight_altitude, 1,False)
print("RandomStart", start_ne, "RandomGoal", goal_ne)


# In[7]:


num_samp = 500
sampler = Sampler(data, flight_altitude, safety_distance)
polygons = sampler._polygons
nodes = sampler.sample(num_samp)
print("Number of nodes", len(nodes))


# In[8]:


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
        #print(type(line), 'canconnect', line)
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
            #print('n1, n2',n1,n2)    
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


# In[9]:


start = list(g.nodes)[0]
k = np.random.randint(len(g.nodes))
print(k, len(g.nodes))
goal = list(g.nodes)[k]


grid = create_grid(data, flight_altitude, safety_distance)


# In[10]:


fig = plt.figure()

plt.imshow(grid, cmap='copper', origin='lower', alpha = 0.7)

nmin = np.min(data[:, 0] - data[:, 3])
emin = np.min(data[:, 1] - data[:, 4])

for (n1, n2) in g.edges:
    plt.plot([n1[1] - emin, n2[1] - emin], [n1[0] - nmin, n2[0] - nmin], 'orange' , alpha=0.5)

# Draw all nodes connected or not in blue
for n1 in nodes:
    plt.scatter(n1[1] - emin, n1[0] - nmin, c='blue')
    
# Draw connected nodes in red
for n1 in g.nodes:
    plt.scatter(n1[1] - emin, n1[0] - nmin, c='red')
    
plt.plot(start[1]-emin, start[0]-nmin, 'gv')
plt.plot(goal[1]-emin, goal[0]-nmin, 'gx')
    

plt.xlabel('EAST')
plt.ylabel('NORTH')

plt.show()


# In[7]:


#Voronoi graph
plt.rcParams["figure.figsize"] = [14, 14]


# In[8]:


# If you want to use the prebuilt bresenham method
# Import the Bresenham package
#from bresenham import bresenham
def bres(p1, p2, conservative = True): 
    """
    Extended method for any p1 and p2
    """
    x1, y1 = p1
    x2, y2 = p2
    # First, set dx = x2 - x1 and dy = y2 - y1
    dx, dy = x2 - x1, y2 - y1
    #print('dx',dx)
    try:
        x_st = dx/abs(dx)
    except ZeroDivisionError:
        x_move = 0
        y_move = 1
        x_st = 1
    #Creepy Jupyter gets nan when dividing 0.0/0.0 in try block
    if math.isnan(x_st):
        x_move = 0
        y_move = 1
        x_st = 1
    try:
        y_st = dy/abs(dy)
    except ZeroDivisionError:
        y_move = 0
        x_move = 1
        y_st = 1
    if math.isnan(y_st):
        y_move = 0
        x_move = 1
        y_st = 1
        
    cells = []
    # TODO: Determine valid grid cells

    try:
        m = (y2 - y1)/(x2 - x1) #slope
        b = y2 - m * x2
        s = dx/abs(dx) #sign to multipy without replacing < with >
    except ZeroDivisionError:
        b = 0
        s = 1
    if math.isnan(s):
        b = 0
        s = 1
    # The condition we care about is whether 
    # (x + x_step) * m  + b < y + y_step
    # (x + x_step) dy / dx < y + y_step - b 
    # which implies (dx < 0 case included): 
    # s *(x dy - y dx) < s *(y_st*dx - x_st*dy -b*dx)
    # Then define a new quantity: d = x dy - y dx
    # new condition: s*d < s*(y_st*dx - x_st*dy - b*dx)
    # and set d = 0 initially    
    d = x1 * dy - y1 * dx
    # Initialize i, j indices
    i = x1
    j = y1    
    while abs(i-x1) <= abs(dx) and abs(j-y1) <= abs(dy):  
#        print('x,y',(i,j), abs(i-x1), abs(dx),abs(j-y1), abs(dy))
        cells.append([i,j])
#        print('cells',cells)
        if dx == 0 or dy == 0:
            cells.append([i - x_st*y_move, j - y_st*x_move])
        elif s*d < s*(y_st * dx - x_st * dy - b * dx):
            #(x+1)m<y+1      (x+1)m=y+1   (x+1)m>y+1, m > 0
            #|----------|    |-----|      |--|     
            #|          |dy  |     |dy    |  |dy
            #|----------|    |-----|      |--|
            #  dx              dx          dx 
            
            #(x+1)m+b < y+1 => __/ x += 1, dy>0
            #OR 
            #(x-1)m+b < y-1 => y -= 1, dy<0
#            print('<')
            x_move = (abs(dy) + dy)//(2 * abs(dy)) #1 in case dy>0
            y_move = (abs(dy) - dy)//(2 * abs(dy)) #1 in case dy<0
        elif s*d > s*(y_st * dx - x_st * dy - b * dx):
            #(x+1)m+b > y+1 => __/ y += 1, dy>0
            #OR 
            #(x-1)m+b > y-1 => x -= 1, dy<0
#            print('>')
            x_move = (abs(dy) - dy)//(2 * abs(dy))
            y_move = (abs(dy) + dy)//(2 * abs(dy))
            #print('ij',i,j,'xmv,ymv',x_move,y_move,'ix1,jy1',i-x1,j-y1) 
        elif s*d == s*(y_st * dx - x_st * dy - b * dx): 
            # uncomment these two lines for conservative approach
            if conservative:
                cells.append([i + x_st, j])
                cells.append([i, j + y_st])
#            print('=',s*d,s*(- b))
            x_move = 1
            y_move = 1
        else:
            x_move = 0
            y_move = 0
        i += x_st * x_move 
        j += y_st * y_move 
        d += x_st*x_move*dy - y_st*y_move*dx
    return np.array(cells)

def line_in_cells_plt(p1, p2, cells):
    x1, y1 = p1
    x2, y2 = p2
    # First, set dx = x2 - x1 and dy = y2 - y1
    dx, dy = x2 - x1, y2 - y1
    try:
        x_st = dx//abs(dx)
    except ZeroDivisionError:
        x_st = 1
    try:
        y_st = dy//abs(dy)
    except ZeroDivisionError:
        y_st = 1
        
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]])

    for q in cells:
#        print(q)
        plt.plot([q[0], q[0]+x_st], [q[1], q[1]], 'k')
        plt.plot([q[0], q[0]+x_st], [q[1]+y_st, q[1]+y_st], 'k')
        plt.plot([q[0], q[0]], [q[1],q[1]+y_st], 'k')
        plt.plot([q[0]+x_st, q[0]+x_st], [q[1], q[1]+y_st], 'k')

    plt.grid()
    plt.axis('equal')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Integer based Bresenham algorithm")
    plt.show()
    
# In this new function you'll record obstacle centres and
# create a Voronoi graph around those points
def create_grid_and_edges(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    along with Voronoi graph edges given obstacle data and the
    drone's altitude.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil((north_max - north_min)))
    east_size = int(np.ceil((east_max - east_min)))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))
    # Center offset for grid
    north_min_center = np.min(data[:, 0])
    east_min_center = np.min(data[:, 1])
    
    # Define a list to hold Voronoi points
    points = []
    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]

        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(north - d_north - safety_distance - north_min_center),
                int(north + d_north + safety_distance - north_min_center),
                int(east - d_east - safety_distance - east_min_center),
                int(east + d_east + safety_distance - east_min_center),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1
            
            # add center of obstacles to points list
            points.append([north - north_min, east - east_min])

    # TODO: create a voronoi graph based on
    # location of obstacle centres
    graph = Voronoi(points)
    voronoi_plot_2d(graph)
    plt.show()
    #print(points)
    # TODO: check each edge from graph.ridge_vertices for collision
    edges = []
    for v in graph.ridge_vertices:
        p1 = graph.vertices[v[0]]
        p2 = graph.vertices[v[1]]
        p1_gr = [int(round(x)) for x in p1]
        p2_gr = [int(round(x)) for x in p2]
        p = [p1_gr,p2_gr]
        #print(p1, p1_grid, p2, p2_grid)
    
        in_collision = True
        if np.amin(p) > 0 and np.amax(p[:][0]) < grid.shape[0] and np.amax(p[:][1]) < grid.shape[1]:
            track = bres(p1_gr,p2_gr)
            for q in track:
                #print(q)
                q = [int(x) for x in q]
                if grid[q[0],q[1]] == 1:
                    in_collision = True
                    break
                else:
                    in_collision = False
        if not in_collision:
            edges.append((p1,p2))

    return grid, edges


# In[9]:


# Define a flying altitude (feel free to change this)
#flight_altitude = 3
#safety_distance = 3
import time
t0 = time.time()

grid, edges = create_grid_and_edges(data, flight_altitude, safety_distance)
print('Found %5d edges' % len(edges))
print(flight_altitude, safety_distance)
print('graph took {0} seconds to build'.format(time.time()-t0))


# In[10]:


# TODO: create the graph with the weight of the edges
# set to the Euclidean distance between the points
G = nx.Graph()
for e in edges:
    p1 = tuple(e[0])
    p2 = tuple(e[1])
    dist = np.linalg.norm(np.array(p2) - np.array(p1))
    G.add_edge(p1, p2, weight=dist)
    


# In[11]:


def start_goal_graph(G, start, goal):
    """INPUT: (N,E)
    OUTPUT: (N,E)"""
    # TODO: find start and goal on Graph
    # Some useful functions might be:
        # np.nonzero()
        # np.transpose()
        # np.linalg.norm()
        # np.argmin()
    gr_start = point_on_graph(G, start)
    gr_goal = point_on_graph(G, goal)
    return gr_start, gr_goal

def point_on_graph(G, point):
    """Project point onto Graph
    INPUT: G = networkx.Graph()
            point = (x,y)
    OUTPUT: gr_point = (x,y)"""
    if G.has_node(point):
        gr_point = point
    else:
        graph_points = np.array(G.nodes)
        point_addr = np.linalg.norm(np.array(point) - graph_points,
                               axis = 1).argmin()
        gr_point = tuple(graph_points[point_addr])
    
    return gr_point


# In[56]:


#place random start and goal points on graph
north_min = Sampler.datalimits(data)[0]
east_min = Sampler.datalimits(data)[2]
start_v = (start_ne[0][0] - north_min, start_ne[0][1] - east_min)
goal_v = (goal_ne[0][0] - north_min, goal_ne[0][1] - east_min)

gr_start, gr_goal = start_goal_graph(G, start_v, goal_v)
print('Start and Goal on the graph', gr_start, gr_goal)


# In[13]:


def a_star_graph(graph, h, start, goal):
    """path, cost = a_star_graph(networkx.Graph(), heuristic_func,
                    tuple(skel_start), tuple(skel_goal))
        INPUT: start, goal = tuple(x,y)
        """
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for next_node in graph[current_node]:
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


# In[58]:


path, cost = a_star_graph(G, heuristic, gr_start, gr_goal)
print('Number of edges',len(path), 'Cost', cost)
print()


# In[59]:


# equivalent to
# plt.imshow(np.flip(grid, 0))
# Plot it up!
plt.imshow(grid, origin='lower', cmap='copper', alpha = 0.7) 

# Stepping through each edge
for e in edges:
    p1 = e[0]
    p2 = e[1]
    plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'orange', alpha = 0.5)
    
plt.plot(start_v[1], start_v[0], 'rp')
plt.plot(goal_v[1], goal_v[0], 'rx')

plt.plot(gr_start[1], gr_start[0], 'gp')
plt.plot(gr_goal[1], gr_goal[0], 'gx')

pp = np.array(path)
plt.plot(pp[:,1], pp[:,0], 'r-')


plt.xlabel('EAST')
plt.ylabel('NORTH')
plt.show()


# In[88]:


#Choose a location along that plan and discretize a local volume 
#around that location (for example, you might try a 40x40 m area 
#that is 10 m high discretized into 1m^3 voxels)
d_n = 40
d_e = 40
d_z = 10
voxel_vol = 1
ind = 65
if len(path[ind]) < 3:
    l_start = (path[ind][0],path[ind][1],flight_altitude)
else:
    l_start = path[ind]
locVol = LocVolume(data, l_start, d_n, d_e, d_z, voxel_vol)
#print(locVol._nmin)
#print(locVol._nmax)
#print(locVol._corners,'corners')
l_goal, ind, goal_vox = locVol.localgoal(path, ind)
l_voxmap = locVol.create_localvoxmap(data, voxel_vol)
start_vox = locVol.vox_plot(locVol._c, convert = True)
print(l_voxmap.shape)
fig = plt.figure()

voxels = l_voxmap | goal_vox | start_vox

# set the colors of each object
colors = np.empty(voxels.shape, dtype=object)
colors[l_voxmap] = 'red'
colors[goal_vox] = 'blue'
colors[start_vox] = 'green'

ax = fig.gca(projection='3d')
ax.voxels(voxels, facecolors=colors, edgecolor='k', alpha = 0.7)
ax.set_xlim(l_voxmap.shape[0],0)
ax.set_ylim(0, l_voxmap.shape[1])
ax.set_zlim(0, l_voxmap.shape[2])
plt.xlabel('North', fontsize = 20)
plt.ylabel('East', fontsize = 20)
plt.show()


"""p = 0
while ind < len(path):
    l_start = path[ind]
    locVol = LocalVolume(data, l_start, d_n, d_e, d_z, voxel_vol)
    l_goal, ind = locVol.localgoal(path, ind)
    l_obst = locVol.create_localvoxmap(data, voxel_vol)
"""
    
    


# In[47]:


# Define your goal in the local volume to a a node or voxel at the edge 
# of the volume in the direction of the next waypoint in your 
# coarse global plan.


# In[ ]:


# Plan a path through your 3D grid or graph to that node or voxel 
# at the edge of the local volume.

