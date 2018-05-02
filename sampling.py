#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 06:44:53 2018

@author: yudzhi
"""

# coding: utf-8

# # Random Sampling
# TODO: sample points randomly
# then use KDTree to find nearest neighbor polygon
# and test for collision
# 
# from sampling import Sampler
# sampler = Sampler(data)
# polygons = sampoer._polygons
# nodes = sampler.sample(300)
# grid = create_grid(data,sampler._zmax, 1)
# for p in polygons:
#   if p.crosses(l) and p.height >= min(n1[2], n2[2])

# In[1]:

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



# In[62]:

if __name__ == "__main__":
    pass    