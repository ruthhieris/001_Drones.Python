
# coding: utf-8

# # Random Sampling
# 
# In this notebook you'll work with the obstacle's polygon representation itself.
# 
# Your tasks will be:
# 
# 1. Create polygons.
# 2. Sample random 3D points.
# 3. Remove points contained by an obstacle polygon.
# 
# Recall, a point $(x, y, z)$ collides with a polygon if the $(x, y)$ coordinates are contained by the polygon and the $z$ coordinate (height) is less than the height of the polygon.

# In[1]:


import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Polygon, Point
get_ipython().run_line_magic('matplotlib', 'inline')



# In[2]:


plt.rcParams['figure.figsize'] = 12, 12


# In[3]:


# This is the same obstacle data from the previous lesson.
filename = 'colliders.csv'
data = np.loadtxt(filename, delimiter=',', dtype='Float64', skiprows=2)
print(data)


# ## Create Polygons

# In[32]:


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
        p = Polygon(corners)
        polygons.append((p, height))

    return polygons


# In[33]:


polygons = extract_polygons(data)
#p_arr = np.asarray(polygons)
#center = [np.append(p.centroid, height*0.5) for (p,height) in polygons]
center = [np.asarray(p.centroid) for (p,height) in polygons]
print(center[:10], np.asarray(center).shape)
from sklearn import neighbors
tree = neighbors.KDTree(center,leaf_size = 40)


# # Sampling 3D Points
# 
# Now that we have the extracted the polygons, we need to sample random 3D points. Currently we don't know suitable ranges for x, y, and z. Let's figure out the max and min values for each dimension.

# In[46]:


xmin = np.min(data[:, 0] - data[:, 3])
xmax = np.max(data[:, 0] + data[:, 3])

ymin = np.min(data[:, 1] - data[:, 4])
ymax = np.max(data[:, 1] + data[:, 4])

zmin = 0
# Limit the z axis for the visualization
zmax = 20 #np.max(data[:,2] + data[:,5] + 10) #10

print("X")
print("min = {0}, max = {1}\n".format(xmin, xmax))

print("Y")
print("min = {0}, max = {1}\n".format(ymin, ymax))

print("Z")
print("min = {0}, max = {1}".format(zmin, zmax))


# Next, it's time to sample points. All that's left is picking the distribution and number of samples. The uniform distribution makes sense in this situation since we we'd like to encourage searching the whole space.

# In[56]:


num_samples = 200
#np.random.seed(0)
xvals = np.random.uniform(xmin, xmax, num_samples)
yvals = np.random.uniform(ymin, ymax, num_samples)
zvals = np.random.uniform(zmin, zmax, num_samples)

samples = list(zip(xvals, yvals, zvals))


# In[57]:


samples[:10]


# ## Removing Points Colliding With Obstacles
# 
# Prior to remove a point we must determine whether it collides with any obstacle. Complete the `collides` function below. It should return `True` if the point collides with *any* obstacle and `False` if no collision is detected.

# In[58]:


def collides_tree(tree, polygons, point):  
    dist,ind = tree.query(np.asarray(point[:2]).reshape(1,2),k=3)
    collision = False
    for j in range(ind.shape[1]):
        pnum = ind[0][j]
        (p,height) = polygons[pnum]
        if p.contains(Point(point)) and height >= point[2]:
     #       print(polygons[j])
            collision = True
            break
    # TODO: Determine whether the point collides
    # with any obstacles. 
    #print(point, collision)
    return collision

def collides(polygons, point):   
    # TODO: Determine whether the point collides
    # with any obstacles.    
    for (p, height) in polygons:
        if p.contains(Point(point)) and height >= point[2]:
            return True
    return False


# Use `collides` for all points in the sample.

# In[59]:


t0 = time.time()
to_keep = []
for point in samples:
    if not collides(polygons, point):
        to_keep.append(point)
time_taken = time.time() - t0
print("Time taken for brute force {0} seconds ...".format(time_taken))

t0 = time.time()
to_keep_tree = []
for point in samples:
    if not collides_tree(tree, polygons, point):
        to_keep_tree.append(point)
time_taken = time.time() - t0
print("Time taken for KDTree {0} seconds ...".format(time_taken))


# In[60]:


print('Length Tree {0} and naive {1}'.format(len(to_keep_tree), len(to_keep)))


# ## Points Visualization

# In[61]:


from grid import create_grid
def create_grid2(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.amin(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.amax(data[:, 0] + data[:, 3]))
    print(north_min, north_max)
    #print(north_max)
    # minimum and maximum east coordinates
    east_min = np.floor(np.amin(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.amax(data[:, 1] + data[:, 4]))
    print(east_min,east_max)
    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))
    print(north_size,east_size)
    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))
    # Center offset for grid
#    north_min_center = np.min(data[:, 0])
#    east_min_center = np.min(data[:, 1])
#    print(north_min_center,east_min_center)

    ###########Like this one more##########3
        # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        #print(alt+d_alt+safety_distance, drone_altitude)
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1
    return grid
grid = create_grid(data, zmax, 1)


# In[62]:


fig = plt.figure()

plt.imshow(grid, cmap='Greys', origin='lower')

nmin = np.min(data[:, 0])
emin = np.min(data[:, 1])

# draw points
tree_pts = np.array(to_keep_tree)
north_tr = tree_pts[:,0]
east_tr = tree_pts[:,1]
plt.scatter(east_tr - emin, north_tr - nmin, c='green')

all_pts = np.array(to_keep)
north_vals = all_pts[:,0]
east_vals = all_pts[:,1]
plt.scatter(east_vals - emin, north_vals - nmin, c='red')

plt.ylabel('NORTH',fontsize = 20)
plt.xlabel('EAST', fontsize = 20)

plt.show()


# [Solution](/notebooks/Random-Sampling-Solution.ipynb)

# ## Epilogue
# 
# You may have noticed removing points can be quite lengthy. In the implementation provided here we're naively checking to see if the point collides with each polygon when in reality it can only collide with one, the one that's closest to the point. The question then becomes 
# 
# "How do we efficiently find the closest polygon to the point?"
# 
# One such approach is to use a *[k-d tree](https://en.wikipedia.org/wiki/K-d_tree)*, a space-partitioning data structure which allows search queries in $O(log(n))$. The *k-d tree* achieves this by cutting the search space in half on each step of a query.
# 
# This would bring the total algorithm time down to $O(m * log(n))$ from $O(m*n)$.
# 
# The scikit-learn library has an efficient implementation [readily available](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree).
