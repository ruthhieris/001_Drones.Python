#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 23:45:00 2018

@author: yudzhi
"""
import numpy as np
np.set_printoptions(precision = 3, suppress = True)

class qt(object):
    """
    Quaternions
    
    INPUT: qt(a, b, c, d) - float
                OR
           qt(angle, axis): u0 - float, u - vector
    
    Attributes:
        _a: q0,   0th element of quaternion
        _b: q1*i, 1st element of quaternion
        _c: q2*j, 2nd element of quaternion
        _d: q3*k, 3rd element of quaternion
        _axis:  axis of rotation [qi/sqrt(1-q0^2)]
        _angle: angle of rotation 2*acos(q0)
        _array: array representation 
                |a+id   -b-ic| _ |alpha  -beta |
                |b-ic    a-id| - |beta*  alpha*|
    """
    
    def __init__(self, *args):
        if len(args) == 4:
            self.init_abc(args[0], args[1], args[2], args[3])
        if len(args) == 2:
            self.init_AngAx(args[0], args[1])
        if len(args) == 1:
            self.init_Array(args[0])
            
    def __str__(self):
#        return('<' + str(self._a) + " + " + str(self._b) + "i + " +
#                str(self._c) + "j + " + str(self._d) + "k >")
        return("(" + str(self.q0) + ", " + str(self.q1) + "," +
                 str(self.q2) + "," + str(self.q3) + ")")
        
    def init_abc(self, a,b,c,d):
        self._a = a
        self._b = b
        self._c = c
        self._d = d

    def init_AngAx(self, ang, ax):
        """angle in rad and normalized vector"""
        self._a = np.cos(ang/2)
        self._b = np.sin(ang/2) * ax[0]
        self._c = np.sin(ang/2) * ax[1]
        self._d = np.sin(ang/2) * ax[2]  
        
    def init_Array(self, quatArray):
        self._a = quatArray[0]
        self._b = quatArray[1]
        self._c = quatArray[2]
        self._d = quatArray[3]
                        
    def qdot(qt2, qt1):
        """Quaternion multiplication"""
        qt3_array = np.matmul(qt2.qarray, qt1.qarray)
        #print(qt3_array)
        qt3 = qt.qt_from_array(qt3_array)
        return qt3
    
    def qt_from_array(qarray):
        """find a,b,c,d from qt array"""
        a3 = qarray.real[0][0]
        b3 = qarray.real[1][0]
        c3 = - qarray.imag[1][0] 
        d3 = qarray.imag[0][0]
        
        qt3 = qt(a3,b3,c3,d3)
        return qt3
    
    def qt_from_ScVec(u0,vec):
        """Quaternion from scalar and vector components tuple (u0,vec)"""
        a = u0
        b = vec[0]
        c = vec[1]
        d = vec[2]
        q = qt(a,b,c,d)
        return q
    
    def qInverse(self):
        qt_inv = qt.qt_from_ScVec(self.q0, - self.vec)
        return qt_inv
    
    def qMod(self):
        qt_module = qt.qdot(self, self.qInverse())
        return qt_module.q0
    
    @property
    def qarray(self):
        """array quanternion representation"""
        a = self.q0
        b = self.q1
        c = self.q2
        d = self.q3
        self._array = np.array([[complex(a,d), -complex(b,c)],
                               [complex(b,-c),  complex(a,-d)]])
        return self._array
                
    @property
    def q0(self):
        """float: 0th element of quaternion"""
        return self._a
    
    @property
    def q1(self):
        """float: 1st element of quaternion"""
        return self._b
    
    @property
    def q2(self):
        """float: 2nd element of quaternion"""
        return self._c
    
    @property
    def q3(self):
        """float: 3rd element of quaternion"""
        return self._d
    
    @property
    def vec(self):
        """Vector Component"""
        self._vec = np.array([self.q1, self.q2, self.q3])
        return self._vec

def euler_to_quaternion(angles):
    roll = angles[0]
    pitch = angles[1]
    yaw = angles[2]
    
    q_roll = qt(roll,np.array([1,0,0]))
    #print('roll',q_roll.qarray)
    q_pitch = qt(pitch, np.array([0,1,0]))
    #3print('pitch',pitch, q_pitch.qarray)
    q_yaw = qt(yaw, np.array([0,0,1]))
    #print('yaw',q_yaw.qarray)
                    
    q_yx = qt.qdot(q_pitch, q_roll)
    qt_rot = qt.qdot(q_yaw, q_yx)
    q = np.array([qt_rot.q0,qt_rot.q1,qt_rot.q2,qt_rot.q3])
    return q
    
    # TODO: complete the conversion
    # and return a numpy array of
    # 4 elements representing a quaternion [a, b, c, d]

def quaternion_to_euler(quaternion):
    q = qt(quaternion)
    a = q.q0
    b = q.q1
    c = q.q2
    d = q.q3
    #q._b = -q._b
    roll = np.arctan2(2*(a * b + c * d),1.0 - 2.0*(b**2 + c**2))  
    pitch = np.arcsin(2*(a * c - b * d))
    yaw =  np.arctan2(2*(a * d + b * c),1.0 - 2.0*(c**2 + d**2))  
    Rot = np.array([roll, pitch, yaw])
    return Rot
    # TODO: complete the conversion
    # and return a numpy array of
    # 3 element representing the euler angles [roll, pitch, yaw]
    """
    quaternion multiplication
    INPUT: qt1 = [a1,b1,c1,d1]
           qt2 = [a2,b2,c2,d2]
           qt = a + bi + cj + dk
    OUTPUT: qt = qt1 * qt2
        Re(q) = a = a2*a1 - b2*b1 - c2c1 - d2d1
        Im(q) = a2*vec(q1) + a1*vec(q2) + [vec(qt1),vec(qt2)]
        
    euler = np.array([np.deg2rad(90), np.deg2rad(30), np.deg2rad(10)])
    
    %%%%%%%%%%%%%%%%%%%%%%
    q = euler_to_quaternion(euler) # should be [ 0.683  0.683  0.183 -0.183]
    print(q)
    
    # should be [ 1.570  0.523  0.]
    e = quaternion_to_euler(q)
    print([np.deg2rad(90), np.deg2rad(30), np.deg2rad(0)])
    print(e)
    
    assert np.allclose(euler, quaternion_to_euler(q))
    """
    