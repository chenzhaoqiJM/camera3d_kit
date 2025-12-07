#!/usr/bin/env python
# -*- coding: utf-8 -*-


from math import acos, pi, sin, cos
import math
# from tf.transformations import euler_from_quaternion, quaternion_from_euler

import numpy 
import numpy as np



class TF():
    def __init__(self):
        self._NEXT_AXIS = 0
        self._AXES2TUPLE = 0
        self._TUPLE2AXES = 0
        self._EPS = 0
        self.init_const()
        

    def vector_to_angle(self,vcl):
        
        length = math.sqrt(np.dot(vcl, vcl))
        x = vcl[0]; y = vcl[1]; z = vcl[2]
        crx = np.dot(vcl, np.array([0,0,1]))/ length; ax = math.acos(crx)
        if(y >0):
            ax = math.pi*2 - ax
        
        vcl_xy = [vcl[0], vcl[1], 0]; vcl_xy = numpy.array(vcl_xy)
        # print(vcl_xy)
        crz = np.dot(vcl_xy, np.array([0,1,0]))/ (math.sqrt(np.dot(vcl_xy, vcl_xy)))
        az=math.acos(crz)
        if(x >0 and y >0):
            az = math.pi*2 - az
        if(x<=0 and y > 0):
            az = az
        if(x > 0 and y < 0):
            az = math.pi - az
        if(x <0 and y < 0):
            az = math.pi + az
        cry = 1.0; ay=math.acos(cry)
        return [ax, ay, az]

    def vector_to_angle_sphere(self,vcl):
        
        vcl = numpy.array(vcl)
        vcl = vcl / math.sqrt(numpy.dot(vcl, vcl))
        a = vcl[0]; b = vcl[1]; c = vcl[2]
        # print("^%$**$#%##^&",vcl, a, b, c)
        if c >1.0:
            c=1.0
        if c<-1.0:
            c = -1.0
        ax = math.acos(c)

        ay = 0

        k = math.sin(ax)
        if abs(k)<0.000001:
            k = 0.000001
        # print("!$$##@$%^^$##@#", a, k)
        t = a / k
        if t >1.0:
            t = 1.0
        if t < -1.0:
            t = -1.0
        az = math.asin( t )
        

        return [ax, ay, az]  
    
    def init_const(self):
                # axis sequences for Euler angles
        self._NEXT_AXIS = [1, 2, 0, 1]   
        self._EPS = numpy.finfo(float).eps * 4.0

        # map axes strings to/from tuples of inner axis, parity, repetition, frame
        self._AXES2TUPLE = {
            'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
            'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
            'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
            'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
            'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
            'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
            'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
            'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

        self._TUPLE2AXES = dict((v, k) for k, v in self._AXES2TUPLE.items())

    def quaternion_from_euler(self ,ai, aj, ak, axes='sxyz'):
        """Return quaternion from Euler angles and axis sequence.

        ai, aj, ak : Euler's roll, pitch and yaw angles
        axes : One of 24 axis sequences as string or encoded tuple

        >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
        >>> numpy.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
        True

        """
        _AXES2TUPLE = self._AXES2TUPLE; _TUPLE2AXES = self._TUPLE2AXES; _NEXT_AXIS = self._NEXT_AXIS
        try:
            firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
        except (AttributeError, KeyError):
            _ = _TUPLE2AXES[axes]
            firstaxis, parity, repetition, frame = axes

        i = firstaxis
        j = _NEXT_AXIS[i+parity]
        k = _NEXT_AXIS[i-parity+1]

        if frame:
            ai, ak = ak, ai
        if parity:
            aj = -aj

        ai /= 2.0
        aj /= 2.0
        ak /= 2.0
        ci = math.cos(ai)
        si = math.sin(ai)
        cj = math.cos(aj)
        sj = math.sin(aj)
        ck = math.cos(ak)
        sk = math.sin(ak)
        cc = ci*ck
        cs = ci*sk
        sc = si*ck
        ss = si*sk

        quaternion = np.empty((4, ), dtype=np.float64)
        if repetition:
            quaternion[i] = cj*(cs + sc)
            quaternion[j] = sj*(cc + ss)
            quaternion[k] = sj*(cs - sc)
            quaternion[3] = cj*(cc - ss)
        else:
            quaternion[i] = cj*sc - sj*cs
            quaternion[j] = cj*ss + sj*cc
            quaternion[k] = cj*cs - sj*sc
            quaternion[3] = cj*cc + sj*ss
        if parity:
            quaternion[j] *= -1

        return quaternion
        
    def eulerAnglesToRotationMatrix(self, theta) :
        import numpy as np
        dvx = -1.0
        dvy = -1.0
        dvz = -1.0
        R_x = np.array([[1,         0,                  0                   ],
                        [0,         math.cos(theta[0]), dvx* -math.sin(theta[0]) ],
                        [0,         dvx* math.sin(theta[0]), math.cos(theta[0])  ]
                        ])
            
            
                        
        R_y = np.array([[math.cos(theta[1]),    0,      dvy* math.sin(theta[1])  ],
                        [0,                     1,      0                   ],
                        [dvy* -math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                        ])
                    
        R_z = np.array([[math.cos(theta[2]),    dvz* -math.sin(theta[2]),    0],
                        [dvz* math.sin(theta[2]),    math.cos(theta[2]),     0],
                        [0,                     0,                      1]
                        ])
                        
                        
        R = np.dot(np.dot( R_x, R_y ), R_z)
    
        return R

    def euler_from_quaternion(self, quaternion, axes='sxyz'):
        """Return Euler angles from quaternion for specified axis sequence.

        >>> angles = euler_from_quaternion([0.06146124, 0, 0, 0.99810947])
        >>> numpy.allclose(angles, [0.123, 0, 0])
        True

        """
        return self.euler_from_matrix(self.quaternion_matrix(quaternion), axes)

    def euler_from_matrix(self, matrix, axes='sxyz'):
        """Return Euler angles from rotation matrix for specified axis sequence.

        axes : One of 24 axis sequences as string or encoded tuple

        Note that many Euler angle triplets can describe one matrix.

        >>> R0 = euler_matrix(1, 2, 3, 'syxz')
        >>> al, be, ga = euler_from_matrix(R0, 'syxz')
        >>> R1 = euler_matrix(al, be, ga, 'syxz')
        >>> numpy.allclose(R0, R1)
        True
        >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
        >>> for axes in _AXES2TUPLE.keys():
        ...    R0 = euler_matrix(axes=axes, *angles)
        ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
        ...    if not numpy.allclose(R0, R1): print axes, "failed"

        """
        _AXES2TUPLE = self._AXES2TUPLE
        _TUPLE2AXES = self._TUPLE2AXES; _NEXT_AXIS = self._NEXT_AXIS
        _EPS = self._EPS
        try:
            firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
        except (AttributeError, KeyError):
            _ = _TUPLE2AXES[axes]
            firstaxis, parity, repetition, frame = axes

        i = firstaxis
        j = _NEXT_AXIS[i+parity]
        k = _NEXT_AXIS[i-parity+1]

        M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:3, :3]
        if repetition:
            sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
            if sy > _EPS:
                ax = math.atan2( M[i, j],  M[i, k])
                ay = math.atan2( sy,       M[i, i])
                az = math.atan2( M[j, i], -M[k, i])
            else:
                ax = math.atan2(-M[j, k],  M[j, j])
                ay = math.atan2( sy,       M[i, i])
                az = 0.0
        else:
            cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
            if cy > _EPS:
                ax = math.atan2( M[k, j],  M[k, k])
                ay = math.atan2(-M[k, i],  cy)
                az = math.atan2( M[j, i],  M[i, i])
            else:
                ax = math.atan2(-M[j, k],  M[j, j])
                ay = math.atan2(-M[k, i],  cy)
                az = 0.0

        if parity:
            ax, ay, az = -ax, -ay, -az
        if frame:
            ax, az = az, ax
        return ax, ay, az

    def quaternion_matrix(self, quaternion):
        """Return homogeneous rotation matrix from quaternion.

        >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
        >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
        True

        """
        _EPS = self._EPS
        q = numpy.array(quaternion[:4], dtype=numpy.float64, copy=True)
        nq = numpy.dot(q, q)
        if nq < _EPS:
            return numpy.identity(4)
        q *= math.sqrt(2.0 / nq)
        q = numpy.outer(q, q)
        return numpy.array((
            (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
            (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
            (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
            (                0.0,                 0.0,                 0.0, 1.0)
            ), dtype=numpy.float64)

    def quaternion_multiply(self, quaternion1, quaternion0):
        """Return multiplication of two quaternions.

        >>> q = quaternion_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
        >>> numpy.allclose(q, [-44, -14, 48, 28])
        True

        """
        x0, y0, z0, w0 = quaternion0
        x1, y1, z1, w1 = quaternion1
        return numpy.array((
            x1*w0 + y1*z0 - z1*y0 + w1*x0,
            -x1*z0 + y1*w0 + z1*x0 + w1*y0,
            x1*y0 - y1*x0 + z1*w0 + w1*z0,
            -x1*x0 - y1*y0 - z1*z0 + w1*w0), dtype=numpy.float64)
        
    def quaternion_from_matrix(self, matrix):
        """Return quaternion from rotation matrix.

        >>> R = rotation_matrix(0.123, (1, 2, 3))
        >>> q = quaternion_from_matrix(R)
        >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
        True

        """
        q = numpy.empty((4, ), dtype=numpy.float64)
        M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
        t = numpy.trace(M)
        if t > M[3, 3]:
            q[3] = t
            q[2] = M[1, 0] - M[0, 1]
            q[1] = M[0, 2] - M[2, 0]
            q[0] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
        return q
    
    def euler_from_matrix(self, matrix, axes='sxyz'):
        """Return Euler angles from rotation matrix for specified axis sequence.

        axes : One of 24 axis sequences as string or encoded tuple

        Note that many Euler angle triplets can describe one matrix.

        >>> R0 = euler_matrix(1, 2, 3, 'syxz')
        >>> al, be, ga = euler_from_matrix(R0, 'syxz')
        >>> R1 = euler_matrix(al, be, ga, 'syxz')
        >>> numpy.allclose(R0, R1)
        True
        >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
        >>> for axes in _AXES2TUPLE.keys():
        ...    R0 = euler_matrix(axes=axes, *angles)
        ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
        ...    if not numpy.allclose(R0, R1): print axes, "failed"

        """
        _AXES2TUPLE = self._AXES2TUPLE; _TUPLE2AXES = self._TUPLE2AXES
        _NEXT_AXIS = self._NEXT_AXIS; _EPS = self._EPS
        try:
            firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
        except (AttributeError, KeyError):
            _ = _TUPLE2AXES[axes]
            firstaxis, parity, repetition, frame = axes

        i = firstaxis
        j = _NEXT_AXIS[i+parity]
        k = _NEXT_AXIS[i-parity+1]

        M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:3, :3]
        if repetition:
            sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
            if sy > _EPS:
                ax = math.atan2( M[i, j],  M[i, k])
                ay = math.atan2( sy,       M[i, i])
                az = math.atan2( M[j, i], -M[k, i])
            else:
                ax = math.atan2(-M[j, k],  M[j, j])
                ay = math.atan2( sy,       M[i, i])
                az = 0.0
        else:
            cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
            if cy > _EPS:
                ax = math.atan2( M[k, j],  M[k, k])
                ay = math.atan2(-M[k, i],  cy)
                az = math.atan2( M[j, i],  M[i, i])
            else:
                ax = math.atan2(-M[j, k],  M[j, j])
                ay = math.atan2(-M[k, i],  cy)
                az = 0.0

        if parity:
            ax, ay, az = -ax, -ay, -az
        if frame:
            ax, az = az, ax
        return ax, ay, az
    
    def rotation_to_homogeneous(self, r):
        r = numpy.array(r)
        eye_one = numpy.eye(4)
        #行向量
        eye_one[0, 0:3] = r[0]
        eye_one[1, 0:3] = r[1]
        eye_one[2, 0:3] = r[2]
        return eye_one

    def matrix_orthogonalization(self, creep_m, row_vector_mode = True):
        #行向量
        creep_m = numpy.array(creep_m)
        if row_vector_mode == False:
            creep_m = creep_m.T
        temp_vx = creep_m[0, 0:3]; temp_vy = creep_m[1, 0:3]; temp_vz = creep_m[2, 0:3]
        real_vx = temp_vx
        real_vy = temp_vy - numpy.dot(temp_vy, real_vx)/numpy.dot(real_vx, real_vx) * real_vx
        real_vz = temp_vz - numpy.dot(temp_vz, real_vx)/numpy.dot(real_vx, real_vx) * real_vx \
                    - numpy.dot(temp_vz, real_vy)/numpy.dot(real_vy, real_vy) * real_vy
       
        real_vx = real_vx / math.sqrt(numpy.dot(real_vx, real_vx))
        real_vy = real_vy / math.sqrt(numpy.dot(real_vy, real_vy))
        real_vz = real_vz / math.sqrt(numpy.dot(real_vz, real_vz))

        creep_m[0, 0:3] = real_vx
        creep_m[1, 0:3] = real_vy
        creep_m[2, 0:3] = real_vz
        return creep_m
        

        