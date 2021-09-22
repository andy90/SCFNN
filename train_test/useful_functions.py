import numpy as np
from parameters import *

def find_max(a):
    np.unravel_index(np.argmax(a), a.shape)

def forward_axis(a):  # assuming that the last two axis of a is nfolder, nconfig, which needs to be move to the front
    adim = len(a.shape)
    neworder = np.concatenate(([-2, -1], np.arange(adim-2)))
    new_a = np.transpose(a, axes=neworder)
    return new_a

def compress_dims(a, dim): # compress dim and dim+1 of a into a single dim
    #arr = numpy.zeros((3, 4, 5, 6, 7, 8))
    #new_arr = arr.reshape(*arr.shape[:2], -1, *arr.shape[-2:])
    #new_arr.shape
    # use the above example to implement compress_dims
    newa = a.reshape(*a.shape[:dim], -1, *a.shape[(dim+2):])
    return newa

def backward_axis(a):  # move the front two axis to the back
    adim = len(a.shape)
    neworder = np.concatenate((np.arange(2, adim), [0, 1]))
    new_a = np.transpose(a, axes=neworder)
    return new_a

def uncompress_dims(a, dim, n1): # uncompress the given dim into two dims (dim1, dim2). The number of elements in dim1 os m1
    # we can also use the example in compress_dims to implement this function
    newa = a.reshape(*a.shape[:dim], n1, -1, *a.shape[(dim + 1):])
    return newa

def get_totaldipole(Oxyz, Hxyz, wxyz):
   return qO * np.sum(Oxyz, axis=(0,3)) + qH * np.sum(Hxyz, axis=(0,3)) + qw * np.sum(wxyz, axis=(0,1,4))
