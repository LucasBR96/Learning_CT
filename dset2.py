#PYTHON STANDARD LIBRARY-----------------------------------------------------
import os
import glob
from collections import namedtuple
import functools
import csv
import random
import sys
import itertools
import math

#THIRD PARTY MODULES--------------------------------------------------------
import SimpleITK as sitk
import diskcache
import numpy
import torch
import torch.utils.data as dset

#MADE FOR THIS PROJECT-------------------------------------------------------
from Annotations import candidate_info , get_candidate_info_list
from ct_object import getCt

def split_canlst( is_val , data ):
    selected_cand = [ cand for i , cand in enumerate( data ) if i%10 != int( is_val ) ]
    random.shuffle( selected_cand )
    return selected_cand

def add_padding( box , desired_shape , val ):

    shape = box.shape
    diff = [ dim_des - dim for dim_des , dim in zip( desired_shape , shape ) ]
    if any( x < 0 for x in diff ):
        raise ValueError

    pad_tups = []
    for x in diff:
        tup = ( 0 , 0 )
        if x != 0:
            tup = ( 
                int( numpy.floor( x/2 ) ),
                int( numpy.ceil( x/2 ) ) 
                )
        pad_tups.append( tup )

    new_box = numpy.pad( box ,
    pad_tups , 'constant',
    constant_values = val )

    return new_box
def rotate_and_translate( shape , rot , trans = None ):

    if trans is None:
        trans = numpy.zeros( 2 )
        # trans = numpy.array( [ 5 , -5 ] )

    a , b = shape
    coord = itertools.product( range( a ) , range( b ) )
    coord = numpy.array( [ ( row , col ) for row , col in coord ] ) - numpy.array( [ a , b ] )/2
    new_coord = coord@rot + trans
    # coord_mat = numpy.array( [ ( row , col ) for row , col in coord ] ) - numpy.array( [ a , b ] )/2
    # new_coord = coord_mat@rot + trans

    coord = coord + numpy.array( [ a , b ] )/2
    new_coord = new_coord + numpy.array( [ a , b ] )/2
    pairs = []
    for ( x , y ) , ( nx , ny ) in zip( coord , new_coord ):

        c1 = ( 0 <= nx < a )
        c2 = ( 0 <= ny < b )
        if not ( c1 and c2 ):
            continue

        # nx = int( nx )
        # ny = int( ny )

        x = math.floor( x )
        y = math.floor( y )
        nx = math.floor( nx )
        ny = math.floor( ny )
        pairs.append( [ x , y , nx , ny ])
    return [ numpy.array( i ) for i in zip( *pairs ) ]

def rotation2dmatrix( ang , degree = False ):

    if degree:
        ang = numpy.deg2rad( ang )

    trans = numpy.array( [
        numpy.cos( ang ),
        -numpy.sin( ang ),
        numpy.sin( ang ),
        numpy.cos( ang )
    ] )

    return trans.reshape( 2 , 2 )  
    
def get_ct_heat( uid , index ):
    
    ct_slice = get_ct_slice( uid , index )
    heat_points = get_points( uid )
    heat_map = numpy.zeros( ct_slice.shape )
    for i , r , c in heat_points:
        if index != i: continue
        heat_map[ r , c ] = 1.
    return ct_slice , heat_map

cache_3 = diskcache.Cache()
@cache_3.memoize()
def get_ct_slice( uid , index ):
    
    ct = getCt( uid )
    return ct.Mat[ index ].copy()

cache_4 = diskcache.Cache()
@cache_4.memoize()
def get_points( uid ):
    
    cand_list = get_candidate_info_list().copy()
    ct = getCt( uid )
    points = []
    for cand in cand_list:
        if cand.series_uid != uid:
            continue
        points.extend( ct.highlight_points( cand.center_xyz , cand.diameter_mm ) )
    return points

@functools.lru_cache( 1 )
def get_heights():

    uid_set = set( [ cand.series_uid for cand in get_candidate_info_list() ] )
    ct_heights = []
    for uid in uid_set:
        ct = getCt( uid )
        ct_heights.append( ( uid , ct.Mat.shape[ 0 ] ) )
    return ct_heights
    
class seg_dset( dset.Dataset ):

    def __init__( self , is_val = False , augment = False ):

        self.is_val = is_val
        self.augment = augment and not( is_val )
        
        heights = get_heights()
        stacks = []
        for uid , h in heights:
            uid_lst = [ uid ]*h
            lst = list( zip( uid_lst , range( h ) ) )
            stacks.extend( lst )
        self.ct_slices = split_canlst( is_val , stacks )

    def __len__( self ):

        n = len( self.ct_slices )
        if self.augment:
            n *= 10
        return n

    def __getitem__( self , idx ):

        uid , index = self.ct_slices[ idx ]
        ct_slice , heat_map = get_ct_heat( uid , index )

        if self.augment:
            ct_slice , heat_map = self.aug_cand( ct_slice , heat_map )

        a , b = ct_slice.shape
        if not ( a == b == 512 ):
            ct_slice = add_padding( ct_slice , ( 512 , 512 ) , -1000 )    
            heat_map = add_padding( heat_map , ( 512 , 512 ) , 0 )

        ct_slice = torch.from_numpy( ct_slice )
        heat_map = torch.from_numpy( heat_map )
        return ct_slice , heat_map
    
    def aug_cand( self , ct_slice , heat_map ):
        
        ang = 60*numpy.random.normal()
        rot = rotation2dmatrix( ang , True )
        trans = 10*numpy.random.normal( size = 2 )
        x , y , nx , ny = rotate_and_translate( ct_slice.shape , rot , trans )

        ct_rot = ct_slice.min()*numpy.ones( ct_slice.shape )
        ct_rot[ nx ,ny ] = ct_slice[ x , y ] 
        hm_rot = heat_map.min()*numpy.ones( ct_slice.shape )
        hm_rot[ nx ,ny ] = heat_map[ x , y ]
        return ct_rot , hm_rot
