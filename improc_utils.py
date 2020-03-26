#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:29:09 2020

@author: reisertm
"""

import tensorflow as tf

import numpy as np

########## Cropmodel ##########################################

def gaussian2D(std):  
  size = tf.cast(tf.math.floor(2.5*std),tf.int32).numpy()
  X,Y = tf.meshgrid(list(range(-size,size)),list(range(-size,size)))
  X = tf.cast(X,dtype=tf.float32)
  Y = tf.cast(Y,dtype=tf.float32)
  gauss_kernel = tf.exp(-(X*X + Y*Y)/(2*std*std))
  return gauss_kernel / tf.reduce_sum(gauss_kernel)

def conv_gauss2D(img,std):
  g = gaussian2D(std)
  g = tf.expand_dims(g,2)
  g = tf.expand_dims(g,3)
  r = []
  for k in range(img.shape[3]):
    r.append(tf.nn.conv2d(img[:,:,:,k:k+1],g,1,'SAME'))
  r = tf.concat(r,3)
  return r

def gaussian3D(std):  
  size = tf.cast(tf.math.floor(2.5*std),tf.int32).numpy()
  X,Y,Z = tf.meshgrid(list(range(-size,size)),list(range(-size,size)),list(range(-size,size)))
  X = tf.cast(X,dtype=tf.float32)
  Y = tf.cast(Y,dtype=tf.float32)
  Z = tf.cast(Y,dtype=tf.float32)
  gauss_kernel = tf.exp(-(X*X + Y*Y + Z*Z)/(2*std*std))
  return gauss_kernel / tf.reduce_sum(gauss_kernel)

def conv_gauss3D(img,std):
  g = gaussian3D(std)
  g = tf.expand_dims(g,3)
  g = tf.expand_dims(g,4)
  r = []
  for k in range(img.shape[4]):
    r.append(tf.nn.conv3d(img[:,:,:,:,k:k+1],g,[1,1,1,1,1],'SAME'))
  r = tf.concat(r,4)
  return r


  # to get a nice shape which dividable by divisor 
def getClosestDivisable(x,divisor):
    for i in range(x,2*x):
      if i % divisor == 0:
        break
    return i


# a simple resize of the image by nearest neighbor interp.
# 2D image array of size [w,h,f] (if batch_dim=False) or [b,w,h,f] (if batch_dim=True)
# 3D image array of size [w,h,d,f] (if batch_dim=False) or [b,w,h,d,f] (if batch_dim=True)
# dest_shape a list of new size (len(dest_shape) = 2 or 3)
def resizeND(image,dest_shape,batch_dim=False,nD=2):
    if not batch_dim:
      image = tf.expand_dims(image,0)
    sz = image.shape
    rans = [None] * nD
    for k in range(nD):
      scfac = sz[k+1] / dest_shape[k]
      rans[k] = tf.cast(tf.range(dest_shape[k]),dtype=tf.float32)*scfac
    qwq = rep_rans(rans,dest_shape,nD)

    index = tf.dtypes.cast(qwq,dtype=tf.int32)
    res = []
    for i in range(sz[0]):
      res.append(tf.expand_dims(tf.gather_nd(tf.squeeze(image[i,...]),index),0))
    res = tf.concat(res,0)
    if len(res.shape) == 3:
      res = tf.expand_dims(res,3)
    if not batch_dim:
      res = tf.squeeze(res[0,...])

    return res


# a simple resize of the image by linear interpolation
# 2D image array of size [w,h,f] (if batch_dim=False) or [b,w,h,f] (if batch_dim=True)
# 3D image array of size [w,h,d,f] (if batch_dim=False) or [b,w,h,d,f] (if batch_dim=True)
# dest_shape a list of new size (len(dest_shape) = 2 or 3)
def resizeNDlinear(image,dest_shape,batch_dim=False,nD=2,edge_center=False):
    if not batch_dim:
      image = tf.expand_dims(image,0)
    sz = image.shape
    rans0 = [None] * nD
    rans1 = [None] * nD
    fracs = [None] * nD


    for k in range(nD):
      scfac = sz[k+1] / dest_shape[k]
      tmp = np.arange(dest_shape[k])*scfac
      if edge_center:
          tmp = tmp + 0.5
      ints = np.floor(tmp)
      rans0[k] = tf.convert_to_tensor(ints,dtype=tf.int32)
      rans1[k] = ints+1
      rans1[k][rans1[k]>=sz[k+1]] = sz[k+1] - 1    
      rans1[k] = tf.convert_to_tensor(rans1[k],dtype=tf.int32)
      fracs[k] = tf.convert_to_tensor(tmp-ints,dtype=tf.float32)
    
    
    weights = rep_rans(fracs,dest_shape,nD)
        
    res = [0]*sz[0]
    
    if nD == 2:

        index =   rep_rans([rans0[0],rans0[1]],dest_shape,nD)
        w = tf.expand_dims((1-weights[...,0])*(1-weights[...,1]),nD)
        for i in range(sz[0]): 
            res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(tf.squeeze(image[i,...]),index),0)

        index =   rep_rans([rans1[0],rans0[1]],dest_shape,nD)
        w = tf.expand_dims((weights[...,0])*(1-weights[...,1]),nD)
        for i in range(sz[0]): 
            res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(tf.squeeze(image[i,...]),index),0)

        index =   rep_rans([rans0[0],rans1[1]],dest_shape,nD)
        w = tf.expand_dims((1-weights[...,0])*(weights[...,1]),nD)
        for i in range(sz[0]): 
            res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(tf.squeeze(image[i,...]),index),0)

        index =   rep_rans([rans1[0],rans1[1]],dest_shape,nD)
        w = tf.expand_dims((weights[...,0])*(weights[...,1]),nD)
        for i in range(sz[0]): 
            res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(tf.squeeze(image[i,...]),index),0)
    
    elif nD == 3:
        if image.shape[4] == 1:
            im =  lambda i: tf.expand_dims(tf.squeeze(image[i,...]),3)
        else:
            im =  lambda i: tf.squeeze(image[i,...])

        index =   rep_rans([rans0[0],rans0[1],rans0[2]],dest_shape,nD)
        w = tf.expand_dims((1-weights[...,0])*(1-weights[...,1])*(1-weights[...,2]),nD)
        for i in range(sz[0]): 
            res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

        index =   rep_rans([rans1[0],rans0[1],rans0[2]],dest_shape,nD)
        w = tf.expand_dims((weights[...,0])*(1-weights[...,1])*(1-weights[...,2]),nD)
        for i in range(sz[0]): 
            res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

        index =   rep_rans([rans0[0],rans1[1],rans0[2]],dest_shape,nD)
        w = tf.expand_dims((1-weights[...,0])*(weights[...,1])*(1-weights[...,2]),nD)
        for i in range(sz[0]): 
            res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

        index =   rep_rans([rans1[0],rans1[1],rans0[2]],dest_shape,nD)
        w = tf.expand_dims((weights[...,0])*(weights[...,1])*(1-weights[...,2]),nD)
        for i in range(sz[0]): 
            res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

        index =   rep_rans([rans0[0],rans0[1],rans1[2]],dest_shape,nD)
        w = tf.expand_dims((1-weights[...,0])*(1-weights[...,1])*(weights[...,2]),nD)
        for i in range(sz[0]): 
            res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

        index =   rep_rans([rans1[0],rans0[1],rans1[2]],dest_shape,nD)
        w = tf.expand_dims((weights[...,0])*(1-weights[...,1])*(weights[...,2]),nD)
        for i in range(sz[0]): 
            res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

        index =   rep_rans([rans0[0],rans1[1],rans1[2]],dest_shape,nD)
        w = tf.expand_dims((1-weights[...,0])*(weights[...,1])*(weights[...,2]),nD)
        for i in range(sz[0]): 
            res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

        index =   rep_rans([rans1[0],rans1[1],rans1[2]],dest_shape,nD)
        w = tf.expand_dims((weights[...,0])*(weights[...,1])*(weights[...,2]),nD)
        for i in range(sz[0]): 
            res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

    res = tf.concat(res,0)
    if len(res.shape) == 3:
      res = tf.expand_dims(res,3)
    if not batch_dim:
      res = tf.squeeze(res[0,...])

    return res



# computes a meshgrid like thing, but  suitable for gather_nd
# output shape 2D : w h 2  and 3D: w h d 3
def rep_rans(rans,sizes,nD):
    if nD == 2:
      qwq = tf.concat( [       
                        tf.expand_dims(tf.tile(tf.expand_dims(rans[0],1),[1, sizes[1]]),2) ,
                        tf.expand_dims(tf.tile(tf.expand_dims(rans[1],0),[sizes[0], 1]),2) 
                        ] , 2) 
    elif nD == 3:        
      qwq = tf.concat( [       
                        tf.expand_dims(tf.tile(tf.expand_dims(tf.expand_dims(rans[0],1),2),[1, sizes[1], sizes[2]]),3) ,
                        tf.expand_dims(tf.tile(tf.expand_dims(tf.expand_dims(rans[1],0),2),[sizes[0], 1, sizes[2]]),3) ,
                        tf.expand_dims(tf.tile(tf.expand_dims(tf.expand_dims(rans[2],0),1),[sizes[0], sizes[1], 1]),3) 
                      ] , 3) 
    else: 
      assert "other than 2D/3D not implemented"
    return qwq


