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


#Functions for dealing with the Annotation protodata-------------------------
def balance_fun( data ):

    pos_list = [ x for x in data if x.is_nodule ]
    neg_list = [ x for x in data if not x.is_nodule ]

    return pos_list , neg_list 

def split_canlst( is_val , data ):
    selected_cand = [ cand for i , cand in enumerate( data ) if i%10 != int( is_val ) ]
    random.shuffle( selected_cand )
    return selected_cand

#Functions for maneging the 3d croppings of the scan-----------------------
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
        
cache_1 = diskcache.Cache()
@cache_1.memoize(typed=True)
def is_corrupted( center_xyz , series_uid ):

    ct = getCt(series_uid)
    center_irc = ct.get_irc(center_xyz)
    shape = ct.Mat.shape
    return not all( 0 <= x < y for x , y in zip( center_irc , shape ) )

cache_2 = diskcache.Cache()
@cache_2.memoize(typed=True )
def get_ct_box(series_uid, center_xyz, box ):

    ct = getCt(series_uid)
    center_irc = ct.get_irc(center_xyz)
    
    slice_list = []
    for axis, center_val in enumerate(center_irc):
        start_ndx = int(round(center_val - box[ axis ]/2))
        end_ndx = int(start_ndx + box[ axis ])
        slice_list.append(slice(start_ndx, end_ndx))
    ct_chunk = ct.Mat[tuple(slice_list)]

    return ct_chunk

#Functions for augmentation--------------------------------------------
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


class cand_dset( dset.Dataset ):

    '''
    This class is used when training the classification network. When a candidate is selected, 
    this dataset returns a 3d image of the candidate cropped from its ct_scan and if it is a 
    real Nodule.
    '''

    def __init__( self , is_val = False , balance = False , augment = False ):

        '''
        all argments boolean
        is_val -> used to validate the model
        balanced -> should the data be split and balanced to compensate the assymetric frequency of the
        nodule label
        augment -> should every sample be modified in order to increase the size of the dataset ( only valid 
        when is_val == False )

        '''
        super().__init__()
        
        self.is_val = is_val
        self.balance = balance
        self.augment = augment and not( is_val )

        data = get_candidate_info_list().copy()
        data = split_canlst( is_val , data )
        if self.balance:
            pos_list , neg_list = balance_fun( data )
            self.pos_list = numpy.array( pos_list )
            self.neg_list = numpy.array( neg_list )
        else:
            self.cand_list = numpy.array( data )

    def __len__( self ):
        
        if self.balance:
            #-----------------------------------------------------------------------------------------
            # if neg_list is larger and we want to call positive sample as frequeantly as negative ones,
            # therefore the actual lenght is twice of neg_list
            n = 2*len( self.neg_list )
            #-----------------------------------------------------------------------------------------
        else:
            n = len( self.cand_list )
        
        if self.augment:
            n *= 10

        return n

    def __getitem__( self , idx ):

        if self.balance:
            lst = self.pos_list if bool( idx%2 ) else self.neg_list
        else:
            lst = self.cand_list
        cand = lst[ idx%len( lst ) ]

        # remember that cand came from an array
        cand = candidate_info(*cand )

        #-------------------------------------------------------------------------------------------------
        # a corrupted candidate has its xyz_coordinate outside of the ct_scan boundaries, thus is not fit 
        # for training. Since we need to return something , self.corrupt_msg will return a tuple with generic
        # data tha should be ignored by the training app
        # if is_corrupted( cand.center_xyz , cand.series_uid ):
        #     arr1 = torch.zeros( 1 , 32 , 48 , 48 )
        #     arr2 = torch.zeros( 2 )
        #     return False , arr1 , arr2
        #--------------------------------------------------------------------------------------------------
        
        ct_chunk , label = self.format_cand( cand )
        if self.augment:
            ct_chunk = self.augm_cand( ct_chunk )
        
        arr1 = torch.from_numpy( ct_chunk ).to( torch.float32 ).unsqueeze( 0 ) #for the convolution layer
        arr2 = torch.tensor( label ).to( torch.float32 )
        # return True , arr1 , arr2
        return arr1 , arr2
        
    def format_cand( self , cand ):
        
        uid = cand.series_uid
        xyz = cand.center_xyz
        box = ( 32 , 48 , 48 )
        ct_chunk = get_ct_box( uid , xyz , box ).copy()

        #------------------------------------------------------------------------------------
        # for some unknown reason some ct_chunks come with smaller shapes than wanted. The code
        # bellow seeks to correct that. 
        if any( x != y for x , y in zip( ct_chunk.shape , box ) ):
            ct_chunk = add_padding( ct_chunk , box , -1000 )
        #------------------------------------------------------------------------------------
        label = int( cand.is_nodule )

        return ct_chunk , label

    def augm_cand( self , ct_chunk ):

        ang = 30*numpy.random.normal()
        rot = rotation2dmatrix( ang , True )
        trans = ( 5*numpy.random.normal( size = 2 ) ).astype( int )
        x , y , nx , ny = rotate_and_translate( ct_chunk[0].shape , rot , trans )

        ct_rot = ct_chunk.min()*numpy.ones( ct_chunk.shape )
        ct_rot[ : , nx , ny ] = ct_chunk[ : , x , y ]
        return ct_rot 


if __name__ == "__main__":

    # import matplotlib
    # import matplotlib.pyplot as pplot

    # A = get_candidate_info_list()[ 15 ]
    # ct = getCt( A.series_uid )
    # index , row , col = ct.get_irc( A.center_xyz )
    # ct_slice = ct.Mat[ index ].copy()
    
    # shape = ct_slice.shape
    # ct_rot = ct_slice.min()*numpy.ones( shape ) 
    # rot = rotation2dmatrix( 0 , True )
    # points = rotate_and_translate( shape , rot )
    # for x , y , nx , ny in points:
    #     ct_rot[ nx ,ny ] = ct_slice[ x , y ]

    # img , ( ax1 , ax2 ) = pplot.subplots( 1 , 2 )
    # ax1.imshow( ct_slice , cmap = "gray")
    # ax2.imshow( ct_rot , cmap = "gray")

    # pplot.show()
    # A = class_dset( balance = True , augment = True )
    # print( len( A ) )
    pass