#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 06:44:53 2018

@author: yudzhi
"""

# coding: utf-8
# Receding Horizon
# Choose a location along that plan and discretize a local volume around that location 
# (for example, you might try a 40x40 m area that is 10 m high discretized into 1m^3 voxels)
# Define your goal in the local volume to a a node or voxel at the edge of the volume 
# in the direction of the next waypoint in your coarse global plan.


# # Random Sampling and Local Volume discretization
# TODO: sample points randomly
# then use KDTree to find nearest neighbor polygon
# and test for collision
# 
# from sampling import Sampler
# sampler = Sampler(data)
# polygons = sampler._polygons
# nodes = sampler.sample(300)
# grid = create_grid(data,sampler._zmax, 1)
# for p in polygons:
#   if p.crosses(l) and p.height >= min(n1[2], n2[2])

# In[1]:

import numpy as np
from shapely.geometry import Polygon, Point, LineString
from sklearn import neighbors

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
        self._corners = self.basecorners(data, center, d_n/2, d_e/2) #([p1,p2,p3,p4])
        self._height = self.prismheight(data, center, d_z/2) #tuple(z_bottom, z_top)
        self._locPrism = Sampler.Prism(self._corners, self._height)
        self._nmin = self._corners[0][0]
        self._nmax = self._corners[1][0]
        self._emin = self._corners[0][1]
        self._emax = self._corners[1][1]
    
    def basecorners(self, data, center, dn, de):
        nmin, nmax, emin, emax = Sampler.datalimits(data)
        
        p1 = tuple(np.clip((center[0] - dn, center[1] - de),(nmin, emin), (nmax, emax)))
        p2 = tuple(np.clip((center[0] + dn, center[1] - de),(nmin, emin), (nmax, emax)))
        p3 = tuple(np.clip((center[0] + dn, center[1] + de),(nmin, emin), (nmax, emax)))
        p4 = tuple(np.clip((center[0] - dn, center[1] + de),(nmin, emin), (nmax, emax)))
        return [p1, p2, p3, p4]
    
    def prismheight(self, center, dz, drone_height, zmin = 0):
        """
        Output: tuple (z_bottom, z_top)
        """
        zmax = center[2] + 2*dz
        z = tuple(np.clip((center[2] - dz, center[2] + dz), zmin, zmax))
        return z
    
    def localgoal(self, path, ind):
        p1 = self._c
        loc_g = []
        while not loc_g and ind < len(path):
            p2 = path[ind+1]
            if self._locPrism.crosses(p1,p2):
                loc_g = self._locPrism.intersection(p1,p2)
                break
            else:
                p1 = p2
                ind += 1
        if loc_g ==[]:
            print("Failed to find a new local goal!")
        return loc_g, ind
        
    
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
    
    def create_localvoxmap(self, data, voxel_size):
        """
        Returns a grid representation of a 3D configuration space inside 
        local Voxel based on given obstacle data.
        The `voxel_size` argument sets the resolution of the voxel map. 
        """
        
        # minimum and maximum north coordinates
        north_min = self._nmin
        north_max = self._nmax
        # minimum and maximum east coordinates
        east_min = self._emin
        east_max = self._emax
    
        alt_min, alt_max = self._height
        
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
    
        loc_voxmap = np.zeros((north_size, east_size, alt_size), dtype=np.bool)
        for i in range(data.shape[0]):
            north, east, alt, d_north, d_east, d_alt = data[i,:]
            obstacle = [
                    int(np.clip((north - d_north - north_min)// voxel_size, north_min, north_max)),
                    int(np.clip((north + d_north - north_min)// voxel_size, north_min, north_max)),
                    int(np.clip((east - d_east - east_min)// voxel_size, east_min, east_max)),
                    int(np.clip((east + d_east - east_min)// voxel_size, east_min, east_max)),
                    int(np.clip((alt + d_alt)//voxel_size, alt_min, alt_max))
                    ]
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
    
    def crosses(self, *line):
        """shapely geometry objects have a method .crosses which return 
        True if the geometries cross paths.
        Input: line (from shapely.geometry import LineString)
                or list(tuple1, tuple2, tuple3...)
                or (tuple1, tuple2,...)
        if points = [tuple1, tuple]
        """
        if not type(line) == LineString:
            line = LineString(list(line))
        #coords = list(zip(*line)) #[(x1,x2),(y1,y2),(z1,z2)]
        return self.p.crosses(line)
    
    def bounds(self):
        """
        Returns a (minx, miny, maxx, maxy) tuple (float values) that bounds the object
        """
        return self.p.bounds
    
    def intersection(self, *line):
        """
        Returns a representation of the intersection of this object with the other geometric object.
        """
        if not type(line) == LineString:
            line = LineString(list(line))
        return self.p.intersection(line)

    def localgoal(self, path, ind):
        loc_goal = []
        while not loc_goal and ind < len(path):
            if self.crosses(path[ind], path[ind+1]):
                loc_goal = self.p.intersection(path[ind],path[ind + 1])
                break
            else:
                ind += 1
            if loc_goal ==[]:
                print("Failed to find a new local goal!")
        return loc_goal, ind
# In[62]:

if __name__ == "__main__":
    pass    