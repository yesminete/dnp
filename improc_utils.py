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

def conv_gauss2D_fft(img,std):
  sz = img.shape
  pad = np.floor(std*2)
  s0 = np.int32(sz[1]+pad)
  s1 = np.int32(sz[2]+pad)
  X,Y = tf.meshgrid(list(range(0,s1)),list(range(0,s0)))
  X = tf.cast(X,dtype=tf.float32)
  Y = tf.cast(Y,dtype=tf.float32)
  X = X - 0.5*(sz[2]+pad)
  Y = Y - 0.5*(sz[1]+pad)
  gauss_kernel = tf.exp(-(X*X + Y*Y)/(2*std*std))
  gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
  gauss_kernel = tf.cast(gauss_kernel,dtype=tf.complex64)
  gauss_kernel = tf.signal.fftshift(gauss_kernel,[0,1])
  gfft = tf.signal.fft2d(gauss_kernel)  
  gfft = tf.expand_dims(gfft,0)
  pads = tf.cast([[0,0],[0,pad],[0,pad]],tf.int32)

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
  sz = img.shape
  pad = np.floor(std*2)
  s0 = np.int32(sz[1]+pad)
  s1 = np.int32(sz[2]+pad)
  s2 = np.int32(sz[3]+pad)
  X,Y,Z = tf.meshgrid(list(range(0,s0)),list(range(0,s1)),list(range(0,s2)),indexing='ij')
  X = tf.cast(X,dtype=tf.float32)
  Y = tf.cast(Y,dtype=tf.float32)
  Z = tf.cast(Z,dtype=tf.float32)
  X = X - 0.5*(sz[1]+pad)
  Y = Y - 0.5*(sz[2]+pad)
  Z = Z - 0.5*(sz[3]+pad)
  gauss_kernel = tf.exp(-(X*X + Y*Y + Z*Z)/(2*std*std))
  gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
  gauss_kernel = tf.cast(gauss_kernel,dtype=tf.complex64)
  gauss_kernel = tf.signal.fftshift(gauss_kernel,[0,1,2])
  gfft = tf.signal.fft3d(gauss_kernel)  
  gfft = tf.expand_dims(gfft,0)
  pads = tf.cast([[0,0],[0,pad],[0,pad],[0,pad]],tf.int32)

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



def gaussian3D(std):  
  size = tf.cast(tf.math.floor(2.5*std),tf.int32).numpy()+1
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

    off = 0.5
    
    if nD==2:
        sz = image.shape
        X,Y = np.meshgrid(np.arange(0,dest_shape[0]),np.arange(0,dest_shape[1]),indexing='ij')
        X = tf.cast(X,dtype=tf.float32)/dest_shape[0]*sz[1]
        Y = tf.cast(Y,dtype=tf.float32)/dest_shape[1]*sz[2]
        if edge_center:
            X = X + off
            Y = Y + off
        res = interp2lin(image,X,Y)
    if nD==3:
        sz = image.shape
        X,Y,Z = np.meshgrid(np.arange(0,dest_shape[0]),np.arange(0,dest_shape[1]),np.arange(0,dest_shape[2]),indexing='ij')
        X = tf.cast(X,dtype=tf.float32)/dest_shape[0]*sz[1]
        Y = tf.cast(Y,dtype=tf.float32)/dest_shape[1]*sz[2]
        Z = tf.cast(Z,dtype=tf.float32)/dest_shape[2]*sz[3]
        if edge_center:
            X = X + off
            Y = Y + off
            Z = Z + off
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







