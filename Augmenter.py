#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:37:27 2020

@author: reisertm
"""

import numpy as np
#from PIL import Image
#import cv2

import tensorflow as tf


from random import sample 
from improc_utils import *

def Augmenter( morph_width = 150,
                morph_strength=0.25,
                rotation_dphi=0.1,
                flip = None ,
                scaling = None,
                normal_noise=0,
                repetitions=1,
                include_original=True):
                

    def augment(data,labels):
        
            sz = data.shape
            if len(sz) == 4:
                nD = 2
            if len(sz) == 5:
                nD = 3
    
            if not include_original:
                data_res = []
                if labels is not None:
                    labels_res = []
            else:
                data_res = [data]
                if labels is not None:
                    labels_res = [labels]
    
            for k in range(repetitions):            
                if nD == 2:
                    X,Y = sampleDefField_2D(sz)                    
                    data_ = interp2lin(data,Y,X)        
                    if labels is not None:
                        labels_ = interp2lin(labels,Y,X)            
                if nD == 3:
                    X,Y,Z = sampleDefField_3D(sz)
                    data_ = interp3lin(data,X,Y,Z)        
                    if labels is not None:                
                        labels_ = interp3lin(labels,X,Y,Z)            
                
                if normal_noise > 0:
                    data_ = data_ + tf.random.normal(data_.shape, mean=0,stddev=normal_noise)
                
                if flip is not None:
                    for j in range(nD):
                        if flip[j]:
                            if np.random.uniform() > 0.5:
                                data_ = np.flip(data_,j+1)
                                if labels is not None:
                                    labels_ = np.flip(labels_,j+1)
                                
                
                
                data_res.append(data_)
                if labels is not None:            
                    labels_res.append(labels_)
            data_res = tf.concat(data_res,0)
            
            if labels is not None:            
                labels_res = tf.concat(labels_res,0)
                return data_res,labels_res
            else:            
                return data_res,None
            
    


    def sampleDefField_2D(sz):
        
        X,Y = np.meshgrid(np.arange(0,sz[2]),np.arange(0,sz[1]))
        
        phi = np.random.uniform(low=-rotation_dphi,high=rotation_dphi)
        
        wid = morph_width/4
        s = wid*wid*morph_strength
        dx = conv_gauss2D_fft(np.expand_dims(np.expand_dims(np.random.normal(0,1,X.shape),0),3),wid)
        dx = np.squeeze(dx)
        dy = conv_gauss2D_fft(np.expand_dims(np.expand_dims(np.random.normal(0,1,X.shape),0),3),wid)
        dy = np.squeeze(dy)
        
        scfacs = [1,1]
        if scaling is not None:
            if isinstance(scaling,list):
                scfacs = [1+np.random.uniform(-1,1)*scaling[0],1+np.random.uniform(-1,1)*scaling[1]]
            else:
                sciso = scaling*np.random.uniform(-1,1)
                scfacs = [1+sciso,1+sciso]
        
        
        cx = 0.5*sz[2]
        cy = 0.5*sz[1]
        nX = scfacs[0]*tf.math.cos(phi)*(X-cx) - scfacs[0]*tf.math.sin(phi)*(Y-cy) + cx + s*dx
        nY = scfacs[1]*tf.math.sin(phi)*(X-cx) + scfacs[1]*tf.math.cos(phi)*(Y-cy) + cy + s*dy
        #dY = np.random.normal(0,s,X.shape)
        
        return nX,nY        
    

    def sampleDefField_3D(sz):
        
        X,Y,Z = np.meshgrid(np.arange(0,sz[1]),np.arange(0,sz[2]),np.arange(0,sz[3]),indexing='ij')
        
        
        wid = morph_width/4
        s = wid*wid*morph_strength
        dx = conv_gauss3D_fft(np.expand_dims(np.expand_dims(np.random.normal(0,1,X.shape),0),4),wid)
        dx = np.squeeze(dx)
        dy = conv_gauss3D_fft(np.expand_dims(np.expand_dims(np.random.normal(0,1,X.shape),0),4),wid)
        dy = np.squeeze(dy)
        dz = conv_gauss3D_fft(np.expand_dims(np.expand_dims(np.random.normal(0,1,X.shape),0),4),wid)
        dz = np.squeeze(dz)
        
        
        scfacs = [1,1,1]
        if scaling is not None:
            if isinstance(scaling,list):
                scfacs = [1+np.random.uniform(-1,1)*scaling[0],1+np.random.uniform(-1,1)*scaling[1],1+np.random.uniform(-1,1)*scaling[2]]
            else:
                sciso = scaling*np.random.uniform(-1,1)
                scfacs = [1+sciso,1+sciso,1+sciso]
        
        
        cx = 0.5*sz[1]
        cy = 0.5*sz[2]
        cz = 0.5*sz[3]
        
        u, _, vh = np.linalg.svd(np.eye(3) + rotation_dphi*np.random.normal(0,1,[3,3]), full_matrices=True)
        R = np.dot(u[:, :6] , vh)
        
        dx = s*dx
        dy = s*dy
        dz = s*dz
        
        nX = scfacs[0]*R[0,0]*(X-cx) + scfacs[0]*R[0,1]*(Y-cy) + scfacs[0]*R[0,2]*(Z-cz) + cx + dx
        nY = scfacs[1]*R[1,0]*(X-cx) + scfacs[1]*R[1,1]*(Y-cy) + scfacs[1]*R[1,2]*(Z-cz) + cy + dy
        nZ = scfacs[2]*R[2,0]*(X-cx) + scfacs[2]*R[2,1]*(Y-cy) + scfacs[2]*R[2,2]*(Z-cz) + cz + dz
        
        return nX,nY,nZ
    

    return augment


