#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:29:09 2020

@author: reisertm
"""

import tensorflow as tf
import nibabel as nib
import numpy as np
from nibabel.processing import (resample_from_to, resample_to_output, smooth_image)
import json
import os

########## Cropmodel ##########################################

def gaussian2D(std):  
  size = tf.cast(tf.math.floor(2.5*std),tf.int32).numpy()+1
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

def conv_boxcar2D(img,std):
  box = tf.ones(std,dtype=img.dtype) / tf.cast(np.prod(std),dtype=img.dtype)
  box = tf.expand_dims(box,2)
  box = tf.expand_dims(box,3)
  if len(img.shape) < 4:
      img = tf.expand_dims(img,3)  
  r = []
  for k in range(img.shape[3]):
    r.append(tf.nn.conv2d(img[:,:,:,k:k+1],box,1,'SAME'))
  r = tf.concat(r,3)
  return r

def conv_gauss2D_fft(img,std):
    
  if isinstance(std,float) or isinstance(std,int):
      std = [std]*2
    
  sz = img.shape
  pad = np.floor(std*2)
  s0 = np.int32(sz[1]+pad[0])
  s1 = np.int32(sz[2]+pad[1])
  X,Y = tf.meshgrid(list(range(0,s0)),list(range(0,s1)),indexing='ij')
  X = tf.cast(X,dtype=tf.float32)
  Y = tf.cast(Y,dtype=tf.float32)
  X = X - 0.5*(sz[1]+pad[0])
  Y = Y - 0.5*(sz[2]+pad[1])
  gauss_kernel = tf.exp(-(X*X/(2*std[0]*std[0]) + Y*Y/(2*std[1]*std[1])))
  gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
  gauss_kernel = tf.cast(gauss_kernel,dtype=tf.complex64)
  gauss_kernel = tf.signal.fftshift(gauss_kernel,[0,1])
  gfft = tf.signal.fft2d(gauss_kernel)  
  gfft = tf.expand_dims(gfft,0)
  pads = tf.cast([[0,0],[0,pad[0]],[0,pad[1]]],tf.int32)


  if len(img.shape) < 4:
      img = tf.expand_dims(img,3)

  r = []
  for k in range(img.shape[3]):
    t = tf.squeeze(img[:,:,:,k])
    if len(t.shape) == 2:
        t = tf.expand_dims(t,0)
    t = tf.cast(t,dtype=tf.complex64)
    t = tf.pad(t,pads,mode='REFLECT')
    t = tf.signal.fft2d(t)
    t = t * gfft
    t = tf.signal.ifft2d(t )
    t = t[:,0:sz[1],0:sz[2]]
    t = tf.math.real(t)
    t = tf.expand_dims(t,3)                   
    r.append(t)
  r = tf.concat(r,3)
  return r

def conv_gauss3D_fft(img,std):
    
  if isinstance(std,float) or isinstance(std,int):
      std = [std]*3
    
    
  sz = img.shape
  pad = np.floor(std*2)
  s0 = np.int32(sz[1]+pad[0])
  s1 = np.int32(sz[2]+pad[1])
  s2 = np.int32(sz[3]+pad[2])
  X,Y,Z = tf.meshgrid(list(range(0,s0)),list(range(0,s1)),list(range(0,s2)),indexing='ij')
  X = tf.cast(X,dtype=tf.float32)
  Y = tf.cast(Y,dtype=tf.float32)
  Z = tf.cast(Z,dtype=tf.float32)
  X = X - 0.5*(sz[1]+pad[0])
  Y = Y - 0.5*(sz[2]+pad[1])
  Z = Z - 0.5*(sz[3]+pad[2])
  gauss_kernel = tf.exp(-(X*X/(2*std[0]*std[0]) + Y*Y/(2*std[1]*std[1]) + Z*Z/(2*std[2]*std[2])))
  gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
  gauss_kernel = tf.cast(gauss_kernel,dtype=tf.complex64)
  gauss_kernel = tf.signal.fftshift(gauss_kernel,[0,1,2])
  gfft = tf.signal.fft3d(gauss_kernel)  
  gfft = tf.expand_dims(gfft,0)
  pads = tf.cast([[0,0],[0,pad[0]],[0,pad[1]],[0,pad[2]]],tf.int32)

  if len(img.shape) < 5:
      img = tf.expand_dims(img,4)

  r = []
  for k in range(img.shape[4]):
    t = tf.squeeze(img[:,:,:,:,k])
    if len(t.shape) == 3:
        t = tf.expand_dims(t,0)
    t = tf.cast(t,dtype=tf.complex64)
    t = tf.pad(t,pads,mode='REFLECT')
    t = tf.signal.fft3d(t)
    t = t * gfft
    t = tf.signal.ifft3d(t )
    t = t[:,0:sz[1],0:sz[2],0:sz[3]]
    t = tf.math.real(t)
    t = tf.expand_dims(t,4)                   
    r.append(t)
  r = tf.concat(r,4)
  return r


def globalmax(x,dummy):
    s = x.shape
    axis = range(1,len(s)-1)
    return x*tf.cast(0.0,dtype=x.dtype) + tf.reduce_max(x,axis=axis,keepdims=True)


def gaussian3D(std):  
  size = tf.cast(tf.math.floor(2.5*std),tf.int32).numpy()+1
  X,Y,Z = tf.meshgrid(list(range(-size,size)),list(range(-size,size)),list(range(-size,size)))
  X = tf.cast(X,dtype=tf.float32)
  Y = tf.cast(Y,dtype=tf.float32)
  Z = tf.cast(Y,dtype=tf.float32)
  gauss_kernel = tf.exp(-(X*X + Y*Y + Z*Z)/(2*std*std))
  return gauss_kernel / tf.reduce_sum(gauss_kernel)


def conv_boxcar3D(img,std):
  box = tf.ones(std,dtype=img.dtype) / tf.cast(np.prod(std),dtype=img.dtype)
  box = tf.expand_dims(box,3)
  box = tf.expand_dims(box,4)
  r = []
  if len(img.shape) < 5:
      img = tf.expand_dims(img,4)
  for k in range(img.shape[4]):
    r.append(tf.nn.conv3d(img[:,:,:,:,k:k+1],box,(1,1,1,1,1),'SAME'))
  r = tf.concat(r,4)
  return r

def poolmax_boxcar3D(img,std):
  r = []
  if len(img.shape) < 5:
      img = tf.expand_dims(img,4)
  r = tf.cast(tf.nn.max_pool3d(tf.cast(img,tf.float32),ksize=std,strides=[1,1,1],padding='SAME'),dtype=img.dtype)
  return r

def mixture_boxcar3D(img,std):
  r = tf.concat([conv_boxcar3D(img,std),
                 poolmax_boxcar3D(img,std),
                 -poolmax_boxcar3D(-img,std)],4)
  return r

def poolmax_boxcar2D(img,std):
  r = []
  if len(img.shape) < 4:
      img = tf.expand_dims(img,3)
  r = tf.nn.max_pool2d(img,ksize=std,strides=[1,1],padding='SAME')
  return r

  # to get a nice shape which dividable by divisor 
def getClosestDivisable(x,divisor):
    for i in range(x,2*x):
      if i % divisor == 0:
        break
    return i



def sparse_scatter(pbox_index,qq,sha):
    
    idx = tf.math.argmax(qq,axis=-1)
    val = tf.math.reduce_max(qq,axis=-1)
    nD = len(sha)-1
    absidx = tf.cast(pbox_index[...,0],dtype=tf.int64)
    w = sha[0]
    for k in range(1,nD):
        absidx = absidx + pbox_index[...,k]*w
        w = w*sha[k]
    IDX = tf.concat([tf.expand_dims(absidx,3),tf.expand_dims(idx,3)],3)
    IDX = tf.reshape(IDX,[np.prod(idx.shape),2])
    val = tf.reshape(val,[np.prod(idx.shape)])
    T = tf.sparse.SparseTensor(IDX,val,[w,sha[-1]])
    return T

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
def resizeNDlinear_old(image,dest_shape,batch_dim=False,nD=2,edge_center=False):
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
        if image.shape[3] == 1:
            im =  lambda i: tf.expand_dims(tf.squeeze(image[i,...]),2)
        else:
            im =  lambda i: tf.squeeze(image[i,...])

        index =   rep_rans([rans0[0],rans0[1]],dest_shape,nD)
        w = tf.expand_dims((1-weights[...,0])*(1-weights[...,1]),nD)
        for i in range(sz[0]): 
            res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

        index =   rep_rans([rans1[0],rans0[1]],dest_shape,nD)
        w = tf.expand_dims((weights[...,0])*(1-weights[...,1]),nD)
        for i in range(sz[0]): 
            res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

        index =   rep_rans([rans0[0],rans1[1]],dest_shape,nD)
        w = tf.expand_dims((1-weights[...,0])*(weights[...,1]),nD)
        for i in range(sz[0]): 
            res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

        index =   rep_rans([rans1[0],rans1[1]],dest_shape,nD)
        w = tf.expand_dims((weights[...,0])*(weights[...,1]),nD)
        for i in range(sz[0]): 
            res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)
    
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
                        tf.expand_dims(tf.tile(tf.expand_dims(rans[0],1),tf.cast([1, sizes[1]],dtype=tf.int32)),2) ,
                        tf.expand_dims(tf.tile(tf.expand_dims(rans[1],0),tf.cast([sizes[0], 1],dtype=tf.int32)),2) 
                        ] , 2) 
    elif nD == 3:        
      qwq = tf.concat( [       
                        tf.expand_dims(tf.tile(tf.expand_dims(tf.expand_dims(rans[0],1),2),tf.cast([1, sizes[1], sizes[2]],dtype=tf.int32)),3) ,
                        tf.expand_dims(tf.tile(tf.expand_dims(tf.expand_dims(rans[1],0),2),tf.cast([sizes[0], 1, sizes[2]],dtype=tf.int32)),3) ,
                        tf.expand_dims(tf.tile(tf.expand_dims(tf.expand_dims(rans[2],0),1),tf.cast([sizes[0], sizes[1], 1],dtype=tf.int32)),3) 
                      ] , 3) 
    else: 
      assert "other than 2D/3D not implemented"
    return qwq



def resizeNDlinear(image,dest_shape,batch_dim=True,nD=3,edge_center=False):

    if not batch_dim:
        image = tf.expand_dims(image,0)

    off = -0.5
    
    if nD==2:
        sz = image.shape
        X,Y = np.meshgrid(np.arange(0,dest_shape[0]),np.arange(0,dest_shape[1]),indexing='ij')
        if edge_center:
            X = X + off
            Y = Y + off
        X = tf.cast(X,dtype=tf.float32)/(dest_shape[0]-1)*(sz[1]-1)
        Y = tf.cast(Y,dtype=tf.float32)/(dest_shape[1]-1)*(sz[2]-1)
        res = interp2lin(image,X,Y)
    if nD==3:
        sz = image.shape
        X,Y,Z = np.meshgrid(np.arange(0,dest_shape[0]),np.arange(0,dest_shape[1]),np.arange(0,dest_shape[2]),indexing='ij')
        if edge_center:
            X = X + off
            Y = Y + off
            Z = Z + off
        X = tf.cast(X,dtype=tf.float32)/(dest_shape[0]-1)*(sz[1]-1)
        Y = tf.cast(Y,dtype=tf.float32)/(dest_shape[1]-1)*(sz[2]-1)
        Z = tf.cast(Z,dtype=tf.float32)/(dest_shape[2]-1)*(sz[3]-1)
        res = interp3lin(image,X,Y,Z)
        
    if not batch_dim:
        return res[0,...]
    else:
        return res
    



def interp2lin(image,X,Y):

    nD = 2    
    Xint = np.floor(X)
    Yint = np.floor(Y)
    Xfrc = X-Xint
    Yfrc = Y-Yint
    sz = image.shape
    
    if image.shape[3] == 1:
        im =  lambda i: np.expand_dims(np.squeeze(image[i,...]),nD)
    else:
        im =  lambda i: np.squeeze(image[i,...])
    
    def getIndex(X,Y,x,y):
        X = X+x
        X[X<0] = 0
        X[X>=sz[1]] = sz[1]-1
        Y = Y+y
        Y[Y<0] = 0
        Y[Y>=sz[2]] = sz[2]-1        
        index=np.concatenate([np.expand_dims(X,nD),np.expand_dims(Y,nD)],nD)
        index = tf.convert_to_tensor(index,dtype=tf.int32)
        return index
   
    
    ftype = image.dtype
   
    res = [0]*sz[0]


    
    index = getIndex(Xint,Yint,0,0)
    w = tf.expand_dims((1-Xfrc)*(1-Yfrc),nD)
    w = tf.cast(w,ftype)
    for i in range(sz[0]): 
       res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

    index = getIndex(Xint,Yint,0,1)
    w = tf.expand_dims((1-Xfrc)*(Yfrc),nD)
    w = tf.cast(w,ftype)
    for i in range(sz[0]): 
        res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)
    
    index = getIndex(Xint,Yint,1,0)
    w = tf.expand_dims((Xfrc)*(1-Yfrc),nD)
    w = tf.cast(w,ftype)
    for i in range(sz[0]): 
        res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)
    
    index = getIndex(Xint,Yint,1,1)
    w = tf.expand_dims((Xfrc)*(Yfrc),nD)
    w = tf.cast(w,ftype)
    for i in range(sz[0]): 
        res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

    res = tf.concat(res,0)
    

    return res 









def interp3lin(image,X,Y,Z):

    nD = 3    
    Xint = np.floor(X)
    Yint = np.floor(Y)
    Zint = np.floor(Z)
    Xfrc = X-Xint
    Yfrc = Y-Yint
    Zfrc = Z-Zint
    sz = image.shape
    
    if image.shape[4] == 1:
        im =  lambda i: np.expand_dims(np.squeeze(image[i,...]),nD)
    else:
        im =  lambda i: np.squeeze(image[i,...])
    
    def getIndex(X,Y,Z,x,y,z):
        X = X+x
        X[X<0.0] = 0.0
        X[X>=sz[1]] = sz[1]-1
        Y = Y+y
        Y[Y<0] = 0
        Y[Y>=sz[2]] = sz[2]-1        
        Z = Z+z
        Z[Z<0] = 0
        Z[Z>=sz[3]] = sz[3]-1        
        index=np.concatenate([np.expand_dims(X,nD),np.expand_dims(Y,nD),np.expand_dims(Z,nD)],nD)
        index = tf.convert_to_tensor(index,dtype=tf.int32)
        return index
    
    res = [0]*sz[0]
    
    ftype = image.dtype
    
    index = getIndex(Xint,Yint,Zint,0,0,0)
    w = tf.expand_dims((1-Xfrc)*(1-Yfrc)*(1-Zfrc),nD)
    w = tf.cast(w,ftype)
    for i in range(sz[0]): 
       res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

    index = getIndex(Xint,Yint,Zint,1,0,0)
    w = tf.expand_dims((Xfrc)*(1-Yfrc)*(1-Zfrc),nD)
    w = tf.cast(w,ftype)
    for i in range(sz[0]): 
       res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

    index = getIndex(Xint,Yint,Zint,0,1,0)
    w = tf.expand_dims((1-Xfrc)*(Yfrc)*(1-Zfrc),nD)
    w = tf.cast(w,ftype)
    for i in range(sz[0]): 
       res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

    index = getIndex(Xint,Yint,Zint,1,1,0)
    w = tf.expand_dims((Xfrc)*(Yfrc)*(1-Zfrc),nD)
    w = tf.cast(w,ftype)
    for i in range(sz[0]): 
       res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

    index = getIndex(Xint,Yint,Zint,0,0,1)
    w = tf.expand_dims((1-Xfrc)*(1-Yfrc)*(Zfrc),nD)
    w = tf.cast(w,ftype)
    for i in range(sz[0]): 
       res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

    index = getIndex(Xint,Yint,Zint,1,0,1)
    w = tf.expand_dims((Xfrc)*(1-Yfrc)*(Zfrc),nD)
    w = tf.cast(w,ftype)
    for i in range(sz[0]): 
       res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

    index = getIndex(Xint,Yint,Zint,0,1,1)
    w = tf.expand_dims((1-Xfrc)*(Yfrc)*(Zfrc),nD)
    w = tf.cast(w,ftype)
    for i in range(sz[0]): 
       res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

    index = getIndex(Xint,Yint,Zint,1,1,1)
    w = tf.expand_dims((Xfrc)*(Yfrc)*(Zfrc),nD)
    w = tf.cast(w,ftype)
    for i in range(sz[0]): 
       res[i] = res[i] + tf.expand_dims(w*tf.gather_nd(im(i),index),0)

    res = tf.concat(res,0)

    return res 


# loads nifti data into tf.tensors
#
# contrasts = [ { 'subj1' :  '/path/to/subj1/T1.nii' ,
#                 'subj2' :  '/path/to/subj2/T1.nii' },
#               { 'subj1' :  '/path/to/subj1/T2.nii' ,
#                 'subj2' :  '/path/to/subj2/T2.nii' } ]
#              
# labels =    [ { 'subj1' :  '/path/to/subj1/hippo.nii' ,
#                 'subj2' :  '/path/to/subj2/thal.nii' },
#               { 'subj1' :  '/path/to/subj1/hippo.nii' ,
#                 'subj2' :  '/path/to/subj2/thal.nii' } ]
#
# class  =    { 'subj1' :  [0,1,0],
#               'subj2' :  [0,0,1] }
#
# class  =    { 'subj1' :  1,
#               'subj2' :  0 }
#


def bbox2(img):
    if len(img.shape) == 3:
        img = np.max(img,axis=-1)
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    min0, max0 = np.where(rows)[0][[0, -1]]
    min1, max1 = np.where(cols)[0][[0, -1]]    
    return [range(min0,max0+1),range(min1,max1+1)]

def bbox3(img):
    if len(img.shape) == 4:
        img = np.max(img,axis=-1)
    rows = np.any(np.any(img, axis=1),axis=1)
    cols = np.any(np.any(img, axis=2),axis=0)
    deps = np.any(np.any(img, axis=0),axis=0)
    min0, max0 = np.where(rows)[0][[0, -1]]
    min1, max1 = np.where(cols)[0][[0, -1]]    
    min2, max2 = np.where(deps)[0][[0, -1]]
    return [range(min0,max0+1),range(min1,max1+1),range(min2,max2+1)]

def load_data_structured(  contrasts, labels=None, classes=None, subjects=None,
                           annotations_selector=None, 
                           class_selector=None, 
                           exclude_incomplete_labels=True,
                           ignore_incomplete_input=False,
                           use_unlabeled_data=False,
                           add_inverted_label=False,one_hot_index_list=None,max_num_data=None,
                           align_physical=False,
                           crop_fdim=None,
                           crop_fdim_labels=None,
                           crop_sdim=None,
                           crop_only_nonzero=False,
                           unravel_lastdim=False,
                           reslice_labels=True,
                           label_transform=None,
                           integer_labels=False,
                           verbose=False,
                           threshold=None,
                           label_cval=np.nan,
                           nD=3,ftype=tf.float32):

    
    def crop_spatial(img,c):
        if crop_sdim is not None:
            if c == None:
                c = crop_sdim
                if crop_sdim == 'minbox':
                    if nD == 2:
                        c = bbox2(img);
                    if nD == 3:
                        c = bbox3(img);
            if nD == 2:
                img = img[c[0],...]
                img = img[:,c[1],...]
            if nD == 3:
                if c[0] is not None:
                    img = img[c[0],...]
                if c[1] is not None:
                    img = img[:,c[1],...]
                if c[2] is not None:
                    img = img[:,:,c[2],...]
            return img,c
        else:
            return img,None
    
    def load_nifti(fname):
        img = nib.load(fname)        
        if align_physical:            
            img = align_to_physical_coords(img)
        return img
    
    if one_hot_index_list is not None:
        threshold = None
    
    
    if subjects is None:
        subjs = contrasts.keys()
    else:
        subjs = subjects
    
    subjs.sort()
    
    
    goingtoload = str(len(subjs))
    if max_num_data is not None:        
        if max_num_data < len(subjs):
            goingtoload = str(max_num_data)
            p = np.random.uniform(size=(len(subjs)))
            idx = np.argsort(p)
            subjs = [subjs[i] for i in idx] 
                
    
    trainset = []
    labelset = []    
    classset = []
    resolutions = []
    subjects_names = []
    if use_unlabeled_data:
        classes = {}
    
    
    print("going to load " + goingtoload + " items")
    for k in subjs:
        if verbose:
            print("loading: " + k)
        else:
            print("X",end="")

        crop_idx = []

        
        if (exclude_incomplete_labels == 1) and not use_unlabeled_data:
            incomplete = False
            if labels is not None:
                for j in range(len(labels)):
                    if not k in labels[j]:
                       # if verbose:
                        print("  missing label " + str(j) + " for subject " + k + ", skipping")
                       # else:
                       #     print("O",end="")                        
                        incomplete = True
                        break
            if incomplete:
                continue

        
       
        scrop = None
        imgs = []
        template_nii = None
        for j in range(len(contrasts)):
            if ignore_incomplete_input:
                if k not in contrasts[j]:
                    continue                                
            item = contrasts[j][k]
            if isinstance(item,dict):  # this is for DPX_selectFiles compat.
                item = item[next(iter(item))]
                fname = item['FilePath']
            else:
                fname = item
            img = load_nifti(fname)      
            if verbose:            
                    print("   loading file:" + fname)           
                
            if img.shape[2] == 1:
                resolution = np.sqrt(np.sum(img.affine[:,0:2]**2,axis=0))
            else:    
                resolution = {"voxsize": img.header['pixdim'][1:4], "input_edges":img.affine}
            
            if len(img.header.extensions) > 0:
                try:
                    eh = img.header.extensions[0].get_content()
                    eh = eh.decode('utf-8')              
                    eh = json.loads(eh)
                    toarr = lambda a: list(map(float,filter(lambda x: x!="",a.split(" "))))
                    bval = toarr(eh['bval'])
                    bvecs = eh['bvec'].split("\n")
                    bvecs = [toarr(bvecs[0]),toarr(bvecs[1]),toarr(bvecs[2])]       
                    resolution['bval'] = bval;
                    resolution['bvec'] = bvecs;
                    print('successfully read diff. info from exthdr')                    
                except Exception as e:
                    pass
                
            header = img.header
            if template_nii is None:
                template_nii = img
                template_shape = img.shape[0:3]
                template_affine = img.affine
                
                
            else:
                if nD == 3:
                    sz1 = img.header.get_data_shape()
                    sz2 = template_nii.header.get_data_shape()
                    if np.abs(sz1[0]-sz2[0]) > 0 or np.abs(sz1[1]-sz2[1]) > 0 or np.abs(sz1[2]-sz2[2]) > 0 or np.sum(np.abs(template_nii.affine-img.affine)) > 0.01:                           
                        img= resample_from_to(img, template_nii,order=3)
                
                
            img = np.squeeze(img.get_fdata())
            if crop_fdim is not None:
                if len(img.shape) > nD:
                    if crop_fdim == 'mean':
                        img = np.mean(img,axis=-1)
                    else:
                        img = img[...,crop_fdim]
            img,scrop = crop_spatial(img,scrop)
            
            img = np.expand_dims(np.squeeze(img),0)
            if len(img.shape) < nD+2:
                img = np.expand_dims(img,nD+1)                
            img = tf.convert_to_tensor(img,dtype=ftype)
            img_inputcontrast = img;
            imgs.append(img)
            
            
        if labels is not None:
            
            labs = []
            for j in range(len(labels)): # over all label files
                if k in labels[j]:
                    item = labels[j][k]
                    if isinstance(item,dict):
                        item = item[next(iter(item))]
                        fname = item['FilePath']
                        ext = os.path.splitext(fname)[1]
                        if 'json' in item:
                            item = item['json']
                            ext = '.json'
                            fname = 'READING'
                        elif fname == '':
                            ext = '.json'
                            fname = 'PINFO'
                    else:
                        fname = item
                        ext = os.path.splitext(fname)[1]
                    notfound = False
                    
                    if ext == '.json' or ext == '.fcsv':
                        
                        if fname != 'READING' and fname != 'PINFO':
                            if ext == '.json':
                                with open(fname) as json_file:
                                    jsobj = json.load(json_file)
                            else:
                                jsobj = loadAnnotation_fcsv(fname) 
                        else:
                            jsobj = item
    
                        if  annotations_selector is not None :       
                            notfound = False
                            if 'keyseq' in annotations_selector:
                                for n in annotations_selector['keyseq']:
                                    if n in jsobj:                                        
                                        jsobj = jsobj[n]
                                    else:
                                        notfound = True
                                        break
                                        
                                if not isinstance(jsobj,list):
                                    jsobj = [jsobj]
                            else:
                                jsobj = jsobj['annotations']
                            
                            if notfound:
                                sel = ['empty']
                            else:
                                annos = loadAnnotation(jsobj,asdict=True)
                                if 'labels' not in annotations_selector:
                                    sel = [annos.keys().sort()]
                                else:
                                    sel = annotations_selector['labels']   
                                
                            
                            delim='.'
                            sizefac=1
                            normalize=False
                            categorial=False
                            if 'delim' in annotations_selector:
                                delim = annotations_selector['delim']
                            if 'sizefac' in annotations_selector:
                                sizefac = annotations_selector['sizefac']
                            if 'normalize' in annotations_selector:
                                normalize = annotations_selector['normalize']
                            if 'categorial' in annotations_selector:
                                categorial = annotations_selector['categorial']
                            label_num = 1
                            render = renderpoints(header,ftype,nD,normalize,img_inputcontrast)
                            
                            for spec in sel:
                                points = []
                                notfound = False                                
                                if spec != 'empty':
                                    for i in spec:
                                        key = i.split(delim)
                                        if key[0] not in annos and not key[0]=='*':
                                            notfound = True
                                            break
                                        if key[0]=='*':
                                            key[0] = next(iter(annos))
                                        if len(key) == 1:
                                            for z in annos[key[0]]:
                                                a = annos[key[0]][z]
                                                p = a['coords'][0:nD]                                                
                                                p.append(a['size']*sizefac)
                                                if 'thresh' in a:
                                                    p.append(a['thresh'])
                                                    
                                                points.append(p)                                        
                                        else:
                                            if key[1] in annos[key[0]]:
                                                a = annos[key[0]][key[1]]
                                                p = a['coords'][0:nD]
                                                if 'size' in a:
                                                    p.append(a['size']*sizefac)
                                                else:
                                                    p.append(sizefac)
                                                if 'thresh' in a:
                                                    p.append(a['thresh'])
                                                points.append(p)
                                            else:
                                                print('key ' + key[1] + ' not present for ' + fname)
                                                
                                    if notfound:
                                        break
                                    if len(points) == 0:
                                        print('key(s) not present for ' + fname)
                                        notfound=True
                                        if exclude_incomplete_labels:
                                            break
                                print('.',end="")
                                img = render(points)
                                img,_ = crop_spatial(img,scrop)                            
                                img = np.expand_dims(img,0)
                                img = np.expand_dims(img,nD+1)
                                if categorial:
                                    if label_num == 1:
                                        labs.append(img)
                                    else:
                                        #labs[0] = labs[0] + label_num*img
                                        selector = tf.math.logical_or(tf.math.logical_and(img>0,labs[0]==0),
                                                                      tf.math.logical_and(img>0,tf.random.uniform(img.shape)>0.5) )
                                        labs[0] = tf.where(selector,label_num,labs[0])
                                    label_num = label_num + 1                                        
                                else:
                                    if notfound:
                                        img = img*0 + label_cval                                                                    
                                    labs.append(img)
                            print(" labels rendered " + str(label_num))       
                        elif class_selector is not None:
                            delim='.'
                            if 'delim' in class_selector:
                                delim = class_selector['delim']
                            if 'formcontent' in jsobj:
                                form = jsobj['formcontent']
                            elif 'content' in jsobj:
                                form = jsobj['content']
                            else:
                                form = jsobj
                          
                            sel = class_selector['classes']  
                            if classes is None:
                                classes = {}
                            
                            thisclass = []
                            for spec in sel:
                                whichtag = None
                                if len(spec.split(":")) > 1:
                                    whichtag = spec.split(":")[1]
                                    spec = spec.split(":")[0]
                                key = spec.split(delim)
                                obj = form
                                for s in range(len(key)):
                                    if key[s] not in obj :
                                        notfound = True
                                        break
                                    else:
                                        obj = obj[key[s]]
                                if whichtag is not None:
                                    obj = float(obj.find("/"+whichtag+"/")> -1)
                                if not notfound:                                
                                    thisclass.append(float(obj))
                                if notfound:
                                    break
                            if not notfound:       
                                if one_hot_index_list is not None:
                                    thisclass_index_rep = []
                                    for j in one_hot_index_list:
                                        if thisclass[0] == j:
                                            thisclass_index_rep.append(1)
                                        else:
                                            thisclass_index_rep.append(0)                                            
                                    classes[k] = tf.convert_to_tensor(thisclass_index_rep,dtype=ftype)

                                else:
                                    classes[k] = thisclass
                            
    
    
                    else:
                        img = load_nifti(fname)   
                        if verbose: 
                            print("   loading file:" + fname)
                            
                            
                        #if nD == 3:
                        sz1 = img.header.get_data_shape()
                        sz2 = template_nii.header.get_data_shape()
                        if reslice_labels:
                            if np.abs(sz1[0]-sz2[0]) > 0 or np.abs(sz1[1]-sz2[1]) > 0 or np.abs(sz1[2]-sz2[2]) > 0 or np.sum(np.abs(template_nii.affine-img.affine)) > 0.01:                           
                                if integer_labels:
                                    order = 0
                                else:
                                    order = 1
                                print("reslicing label, interp. order " + str(order))
                                if len(img.shape) == 3:
                                    img= resample_from_to(img, (template_shape ,template_affine),order=order,cval=label_cval)                                    
                                else:
                                    img= resample_from_to(img, (template_shape + (img.shape[-1],),template_affine),order=order,cval=label_cval)
                        else:
                            resolution['output_edges'] = img.affine


                            
                        if len(img.header.extensions) > 0:
                            exthdr = str(img.header.extensions[0].get_content())
                            exthdr = exthdr.replace("\\n","").replace("\'","")
                            exthdr = exthdr[1:]
                            resolution['exthdr'] = {'xml':exthdr}

                            
                        img = np.squeeze(img.get_fdata());
                        img,_ = crop_spatial(img,scrop)                        
                        if crop_fdim_labels is not None:
                            if len(img.shape) > nD:
                                img = img[...,crop_fdim_labels]


                        if threshold is not None and one_hot_index_list is not None:
                             assert 0,"not possible to use threshold and one_hot_index_list"
                            
                            
                        if label_transform is not None:
                          if label_transform == 'vector_to_tensor':
                              img = tf.where(tf.math.is_nan(img), 0, img)
                              img = img[...,0:3]
                              img = tf.concat([ img[...,0:1]**2,
                                                img[...,1:2]**2,
                                                img[...,2:3]**2,
                                                img[...,0:1]*img[...,1:2],
                                                img[...,0:1]*img[...,2:3],
                                                img[...,1:2]*img[...,2:3] ],3);
                              imgnorm = tf.reduce_sum(img,axis=-1,keepdims=True)
                              img = img / (10+imgnorm)
                              
                          else:
                              assert(False,"given label transform not implemented")
                            
                          
                        elif threshold is not None:
                            img = (img>threshold)*1

    
                        if crop_only_nonzero:
                            crop_idx.append(tf.expand_dims(tf.reduce_max(img,axis=list(range(0,nD))),1))
                            
    
                        img = np.expand_dims(np.squeeze(img),0)
                        if len(img.shape) == nD+1:
                            img = np.expand_dims(img,nD+1)
    
                            
                        if one_hot_index_list is not None:
                            r = []
                            for j in one_hot_index_list:
                                if isinstance(j,list):
                                   x = 0
                                   for ji in j:
                                       x = x + tf.cast(img==ji,dtype=ftype)
                                   r.append( x )
                                else:
                                   r.append( tf.cast(img==j,dtype=ftype) )
                            img = tf.concat(r,nD+1)
                                                    
                            
                        img = tf.convert_to_tensor(img,dtype=ftype)
                        labs.append(img)
                        
                    if use_unlabeled_data:
                        classes[k] = 1
                        
                else:
                    notfound = False

                    if exclude_incomplete_labels == -2:
                        sz = imgs[0].shape
                        print("  missing label " + str(j) + " for subject " + k + ", using position label")                        
                        if nD==2:
                            X,Y = np.meshgrid(np.arange(0,sz[1]),np.arange(0,sz[2]),indexing='ij')
                            X = tf.expand_dims(tf.expand_dims(tf.cast(X,dtype=ftype),0),-1)
                            Y = tf.expand_dims(tf.expand_dims(tf.cast(Y,dtype=ftype),0),-1)
                            X = X/(sz[1]-1)*2*np.pi
                            Y = Y/(sz[2]-1)*2*np.pi
                          #  img = img / (0.0001+tf.reduce_mean(img,axis=[1,2],keepdims=True))                            
                            img = tf.concat([img,tf.math.cos(X),tf.math.sin(X),tf.math.cos(Y),tf.math.sin(Y)],nD+1)
                        if nD==3:
                            X,Y,Z = np.meshgrid(np.arange(0,sz[1]),np.arange(0,sz[2]),np.arange(0,sz[3]),indexing='ij')
                            X = tf.expand_dims(tf.expand_dims(tf.cast(X,dtype=ftype),0),-1)
                            Y = tf.expand_dims(tf.expand_dims(tf.cast(Y,dtype=ftype),0),-1)
                            Z = tf.expand_dims(tf.expand_dims(tf.cast(Z,dtype=ftype),0),-1)
                            X = X/(sz[1]-1)*2*np.pi
                            Y = Y/(sz[2]-1)*2*np.pi
                            Z = Z/(sz[3]-1)*2*np.pi
                            img = img * 0.0005
                          #  img = img / (0.0001+tf.reduce_mean(img,axis=[1,2,3],keepdims=True))
                            img = tf.concat([img,tf.math.cos(X),tf.math.sin(X),tf.math.cos(Y),tf.math.sin(Y),tf.math.cos(Z),tf.math.sin(Z)],nD+1)                            
                    elif exclude_incomplete_labels == -1:
                        print("  missing label " + str(j) + " for subject " + k + ", extending with nans (dont care label)")
                        img = tf.zeros(imgs[0].shape,dtype=ftype)+np.nan
                    else:
                        print("  missing label " + str(j) + " for subject " + k + ", extending with zeros")
                        img = tf.zeros(imgs[0].shape,dtype=ftype)
                    labs.append(img)
                    if use_unlabeled_data:
                        classes[k] = 0
                    
            
            if notfound:
                continue
                    
            if annotations_selector is not None:
                if 'numberScheme' in annotations_selector:
                    nums = 0
                    indic = 0
                    for j in range(len(labs)):
                        nums = nums + labs[j]*(j+1)
                        #indic = indic + labs[j]
                    #indic = tf.cast(indic>0,dtype=ftype)
                    labs = [nums]
                                    
            if add_inverted_label:
                union = 0
                for j in range(len(labs)):
                    union = union + labs[j]
                union = tf.cast(union>0,dtype=ftype)
                inv = 1-union
                labs.append(inv)
                
                    
        if crop_only_nonzero:
            crop_idx = tf.squeeze(tf.where(tf.reduce_min(tf.concat(crop_idx,1),1)))
            if len(crop_idx.shape) == 0:
                crop_idx = tf.reshape(crop_idx,[1])
                
                
            imgs_ = []
            labs_ = []
            if nD==2:
                perm = [3,1,2,0]
            else:
                perm = [4,1,2,3,0]
            for j in imgs:
                imgs_.append(tf.transpose(tf.gather(j,crop_idx,axis=nD+1),perm))
            for j in labs:
                labs_.append(tf.transpose(tf.gather(j,crop_idx,axis=nD+1),perm))
            print('cropped ' + str(crop_idx.shape[0]) + ' nonzero labels')
            imgs = imgs_
            labs = labs_
            
            
        imgs = tf.concat(imgs,nD+1)        
        if unravel_lastdim:
            n_lastdim = imgs.shape[-1]
            print('*' + str(n_lastdim))
            for j in range(n_lastdim):
                trainset.append(imgs[...,j:j+1])
        else:            
            trainset.append(imgs)
        
        if classes is not None:
            tmp = tf.convert_to_tensor(classes[k],dtype=ftype)
            tmp = tf.expand_dims(tmp,0)
            classset.append(tmp)
        if labels is not None and len(labs) > 0:                
            try:
                labs = tf.concat(labs,nD+1)
            except Exception as e:
                    print('label matrix inconsistent: ' + fname)

            if integer_labels:
                labs = tf.cast(labs,dtype=tf.int16)
                
            if unravel_lastdim:
                if n_lastdim != labs.shape[-1]:
                    if unravel_lastdim == 'onlyinput':
                        if labs.shape[-1] != 1:
                            print("input fdim:"+str(n_lastdim))
                            print("shapelabels: "+str(labs.shape))
                            print('from ' + fname)
                            assert 0,'label matrix has a lastdim which is not 1'                        
                        for j in range(n_lastdim):
                            labelset.append(labs[...,0:1])
                    else:
                        print("input fdim:"+str(n_lastdim))
                        print("shapelabels: "+str(labs.shape))
                        print('from ' + fname)
                        assert 0,'label matrix and image do have the same lastdim'
                else:
                    for j in range(n_lastdim):
                        labelset.append(labs[...,j:j+1])
            else:           
                labelset.append(labs)

        

        if unravel_lastdim:
            for j in range(n_lastdim):            
                resolutions.append(resolution)
                subjects_names.append(k + "_" + str(j))
        else:
            resolutions.append(resolution)
            subjects_names.append(k)
        
        if max_num_data is not None and len(trainset) >= max_num_data:
            break
        
    print("")
        
    if classes is not None and labels is not None and len(labelset) > 0:
        return trainset,labelset,classset,resolutions,subjects_names;
        
    if classes is not None:
        return trainset,classset,resolutions,subjects_names;
    if labels is not None:
        return trainset,labelset,resolutions,subjects_names;

    return trainset,resolutions,subjects_names;


def renderpoints(header,ftype,nD,normalize=False,img_inputcontrast=None):

    sz =header['dim'][1:nD+1]
    A = header.get_best_affine()
    
    if nD==2:
        X,Y = np.meshgrid(np.arange(0,sz[0]),np.arange(0,sz[1]),indexing='ij')
        X = tf.cast(X,dtype=ftype)
        Y = tf.cast(Y,dtype=ftype)
        X_ = A[0][0]*X + A[0][1]*Y +  A[0][3]
        Y_ = A[1][0]*X + A[1][1]*Y +  A[1][3]
        def render(points):
            if not normalize:
                img = tf.zeros(sz,dtype=tf.bool)
            else:
                img = tf.zeros(sz,dtype=ftype)
            for p in points:
                R2 = (X_-p[0])*(X_-p[0]) + (Y_-p[1])*(Y_-p[1]) 
                tmp = R2 < p[2]*p[2]
                if not normalize:
                    img = tf.math.logical_or(img,tmp)
                else:
                    tmp = tf.cast(tmp,dtype=ftype)
                    img = img + tmp /tf.reduce_sum(tmp)
            return tf.cast(img,dtype=ftype)
        return render
    if nD==3:
        X,Y,Z = np.meshgrid(np.arange(0,sz[0]),np.arange(0,sz[1]),np.arange(0,sz[2]),indexing='ij')
        X = tf.cast(X,dtype=ftype)
        Y = tf.cast(Y,dtype=ftype)
        Z = tf.cast(Z,dtype=ftype)
        X_ = A[0][0]*X + A[0][1]*Y + A[0][2]*Z + A[0][3]
        Y_ = A[1][0]*X + A[1][1]*Y + A[1][2]*Z + A[1][3]
        Z_ = A[2][0]*X + A[2][1]*Y + A[2][2]*Z + A[2][3]
        def render(points):
            if not normalize:
                img = tf.zeros(sz,dtype=tf.bool)
            else:
                img = tf.zeros(sz,dtype=ftype)
            for p in points:
                R2 = (X_-p[0])*(X_-p[0]) + (Y_-p[1])*(Y_-p[1])  + (Z_-p[2])*(Z_-p[2]) 
                tmp = R2 < p[3]*p[3]
                if len(p) >= 5:
                    tmp = tf.math.logical_and(tmp,img_inputcontrast[0,:,:,:,0]>p[4])
                if not normalize:
                    img = tf.math.logical_or(img,tmp)
                else:
                    tmp = tf.cast(tmp,dtype=ftype)
                    img = img + tmp /tf.reduce_sum(tmp)
            return tf.cast(img,dtype=ftype)
        return render



def loadAnnotation_fcsv(file):
    import csv
    
    with open(file) as csvfile:
         lines = csvfile.readlines()
         content = []
         hdr = None
         for l in lines:
             l = l.rstrip()
             if l.find("columns")>-1:
                 hdr = l[l.find("=")+1:].split(",")
                 continue
             if len(l) == 0 or l[0] == '#':
                 continue
             cont = l.split(",")
             obj = {}
             for k in range(len(hdr)):
                 obj[hdr[k]] = cont[k]
             content.append(obj)
             
         pset = []
         for k in content:
             pset.append({'name':k['label'],
                          'coords':[float(k['x']),
                                    float(k['y']),
                                    float(k['z']),1] })

         return {"annotations": [{ "name":'*', "points":pset , "type":"pointset" } ] }
             
       
    return

def loadAnnotation(annos,asdict=True):
        if not asdict:
            return annos
        else:
            adict = {}
            for aa in annos:
                x = {}
                for ss in aa['points']:
                    key = ss['name']
                    key = key.replace('<br>','')
                    key = key.strip()
                    if key in x:
                        cnt = 1
                        while True:
                            newkey = key + str(cnt)
                            if newkey not in x:
                                x[newkey] = ss
                                break
                            else:
                                cnt = cnt + 1                        
                    else:
                        x[key] = ss
                name = aa['name']
                if name in adict:
                    cnt = 1
                    while True:
                        newname = name + str(cnt)
                        if newname not in adict:
                            adict[newname] = x
                            break
                        else:
                            cnt = cnt + 1
                else:
                    adict[name] = x
            return adict
                    
                
            
        
def getLocalMaximas(res,affine,threshold,idxMode=False,namemap=None,colormap=None,typ='localmax',maxpoints=50,nD=3,size=2):

    x = tf.expand_dims(res,0)
    if len(x.shape) < 5:
        x = tf.expand_dims(x,4)
    
        

        
    def getLM(x,labelnum=0,labelidx=None):
        points = []
        points_raw = []
        
        if typ == 'localmax':
            x = tf.cast(x,dtype=tf.float32)
            if nD == 2:
                x = tf.squeeze(x,-1)
                if labelidx is not None:
                    labelidx = tf.squeeze(labelidx,-1)
                a = x==tf.nn.max_pool2d(x,3,1,'SAME')
            else:
                a = x==tf.nn.max_pool3d(x,3,1,'SAME')
            a = tf.math.logical_and(a,x>threshold)
            idx = tf.where(a)
            maxis = tf.gather_nd(x,idx)
            if labelidx is not None:
                maxis_idx = tf.gather_nd(labelidx,idx)
            idx = idx[:,1:]
            
        else:
            if nD == 2:
                p = tf.squeeze(tf.squeeze(x,-1),0)
                if labelidx is not None:
                    labelidx = tf.squeeze(labelidx,-1)
                R = tf.meshgrid(list(range(0,labelidx.shape[1])),list(range(0,labelidx.shape[2])),indexing='ij')
                R = list(map(lambda x: tf.cast(tf.expand_dims(x,nD),dtype=tf.float32),R))
                X = tf.concat([1+0.0*p,p,p*R[0],p*R[1]],nD)
            else:
                p = tf.squeeze(x,0)
                R = tf.meshgrid(list(range(0,labelidx.shape[1])),list(range(0,labelidx.shape[2])),list(range(0,labelidx.shape[3])),indexing='ij')                
                R = list(map(lambda x: tf.cast(tf.expand_dims(x,nD),dtype=tf.float32),R))
                X = tf.concat([1+0.0*p,p,p*R[0],p*R[1],p*R[2]],nD)
                labelidx = tf.squeeze(labelidx,0)
            num_labels = tf.reduce_max(labelidx)
            accum = tf.scatter_nd(labelidx,X,[num_labels+1,nD+2])
            vol = accum[:,1:2]
            idx = accum[:,2:] / vol
            maxis_idx = tf.range(num_labels+1)
            maxis = tf.squeeze(vol,-1)
            
            
            
        print("number of local maxima: "  + str(maxis.shape[0]))
        colorhex = ['#ff0000','#00ff00','#0000ff','#ffff00','#ff00ff','#00ffff','#ff8000','#ff0080','#80ff80','#0080ff','#808080','#b9aa9b']
        sorted_ = tf.argsort(maxis,-1,'DESCENDING')
        for j in range(min(maxis.shape[0],maxpoints)):
            k = sorted_[j]
            p = tf.concat([idx[k,0:nD],[1]*(4-nD)],0).numpy()
            p = np.matmul(affine,p)
            p = p[0:3]
            
            score = maxis[k].numpy()
            if labelidx is not None:
                theidx = maxis_idx[k].numpy()-1
                
            else:
                theidx = labelnum

            if len(points_raw) > 0:
                if min([np.sum((a-p)**2) for a in points_raw]) < 4*size*size:
                    continue

            points_raw.append(p)
            
            name = 'L'+str(theidx)+' score:'+str(score)
            col = colorhex[theidx%len(colorhex)]
            if namemap is not None:
                name = namemap(theidx,score)
            if colormap is not None:
                col = colormap(theidx)
            if not np.isnan(p[0]):
                points.append({ 'coords': [float(p[0]),float(p[1]),float(p[2]),1],
                                'name': name,
                                'color': col,
                                'size': size                              
                    })
            points = sorted(points,key=lambda x:x['name'])
    
        print("number of local maxima after NMS: "  + str(len(points)))
            
        return points

    if idxMode:
        points = getLM(tf.cast(x[...,1:2],dtype=tf.float32),labelidx = x[...,0:1])
        
    else:
        points = []
        for s in range(0,x.shape[4]):
            points = points + getLM(x[...,s:s+1],labelnum=s)

    
        
    annotation = { "type":"pointset",
                   "state":{},
                   "name":"detections",
                   "points":points        
        }    
    
    return {"annotations":[annotation]}
            
        


def align_to_physical_coords(im):
    
    aff = im.affine
    idx = np.argmax(np.abs(aff[0:3,0:3]),axis=0)
    sg = np.ones(3);
    for i in range(3):
        sg[i] = np.sign(aff[idx[i],i])
        
    perm = np.zeros((4,4))
    perm[3,3] = 1;
    for k in range(3):
        perm[k,idx[k]] = 1 
    
    d = im.get_fdata()
    #d = im._dataobj.get_unscaled()
    
    idxinv = [0]*3
    for k in range(3):
        idxinv[idx[k]] = k
    if len(im.shape) > 3:
        for k in range(len(im.shape)-3):
            idxinv.append(k+3)
    
    
    d = np.transpose(d,idxinv)
    
    
    flip = np.zeros((4,4))
    flip[3,3] = 1;
    for k in range(3):
        flip[k,k] = sg[idxinv[k]]
        if sg[idxinv[k]] < 0:
            flip[k,3] = d.shape[k]-1
    
    for k in range(3):
        if sg[idxinv[k]] < 0:
          d = np.flip(d,axis=k)    
    
    newaff = np.matmul(np.matmul(aff,perm),flip)
    
    im.set_sform(newaff)
        
    return  nib.nifti1.Nifti1Image(d, newaff , header=im.header)


