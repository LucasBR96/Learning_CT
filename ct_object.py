'''
The Ct-scans themselves will be dealt with by the Ct class, wicht will rpresent
the metadata and the body of the scan in the RAM.
'''

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
import numpy

#MADE FOR THIS PROJECT-------------------------------------------------------
from Annotations import candidate_info , get_candidate_info_list

#Especial Tuples --------------------------------------------------------
irc_tuple = namedtuple('IrcTuple', ['index', 'row', 'col'])
xyz_tuple = namedtuple('XyzTuple', ['x', 'y', 'z'])
sample_tuple = namedtuple('sample_tuple', [ 'box' , 'is_nod'])

def irc_to_xyz( coord_irc, origin_xyz, vxSize_xyz, direction_a ):
    cri_a = numpy.array(coord_irc)[::-1] # z , y , x -> x , y , z
    origin_a = numpy.array(origin_xyz)
    vxSize_a = numpy.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    return xyz_tuple(*coords_xyz)

def xyz_to_irc( coord_xyz, origin_irc, vxSize_xyz, direction_a ):

    coord_a = numpy.array( coord_xyz )
    origin_a = numpy.array(origin_irc)
    vxSize_a = numpy.array(vxSize_xyz)

    cri_a = ((coord_a - origin_a) @ numpy.linalg.inv(direction_a)) / vxSize_a
    return irc_tuple( *[ int( x ) for x in numpy.round( cri_a )[ : : -1] ] )

class Ct:
    def __init__( self, series_uid ):
        mhd_path = glob.glob(
        'E:\Datasets\CT\subset*\{}.mhd'.format(series_uid)
        )[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = numpy.array(sitk.GetArrayFromImage(ct_mhd), dtype= numpy.float32)
        self.Mat = numpy.clip( ct_a , -1000 , 1000 )
        self.series_uid = series_uid

        # ---------------------------------------------------------------------
        # These are to mapping xyy mm coord to array positions
        self.xyz_origin = xyz_tuple(*ct_mhd.GetOrigin() )
        self.vx_size = xyz_tuple(*ct_mhd.GetSpacing() )
        self.direction_xyz = numpy.array( ct_mhd.GetDirection() ).reshape( 3 , 3 )
        # ---------------------------------------------------------------------

        pass

        #box bound -> i , r , c 
    def get_irc( self , xyz_pos ):
        center_irc = xyz_to_irc( xyz_pos , self.xyz_origin , 
        self.vx_size , self.direction_xyz)
        return center_irc
    
    def get_xyz( self , irc_pos ):
        center_xyz = irc_to_xyz( irc_pos , self.xyz_origin , self.vx_size,
        self.direction_xyz )
        return center_xyz
        
    
    def highlight_points( self , center_xyz , diam , epsilon = .1 ):

        if diam == 0:
            diam = 10
        box_bound = [ math.ceil( diam/d ) for d in self.vx_size[::-1] ]
        center_irc = self.get_irc( center_xyz )
        limits = []
        for ax , side , ctr in zip( range( 3 ) , box_bound , center_irc ):

            lower_side = math.floor( side/2 )
            low = max( ctr - lower_side , 0 )
            upper_side = math.ceil( side/2 ) + 1
            hight = min( ctr + upper_side, self.Mat.shape[ ax ] )
            limits.append( ( low , hight ) )

        slice_list = [ slice( start , end , 1 ) for start , end in limits ]
        block = self.Mat[ slice_list[ 0 ] , slice_list[ 1 ] , slice_list[ 2 ] ]
        mu = block.mean()
        std = block.std()
        sigma = std if std != 0 else 1

        range_list = [ range( start , end ) for start , end in limits ]
        coord = itertools.product( *range_list )
        selected = []
        for i , j , k in coord:
            z = ( self.Mat[ i , j , k ] - mu )/sigma
            if abs( z ) < epsilon: selected.append( ( i , j , k ) )

        return selected

    
@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    '''
    The ct scan will never be called directly, but by this function, that will save a single scan in
    in the cache
    '''
    return Ct(series_uid)