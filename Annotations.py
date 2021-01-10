'''
As said before, the annotation part of the raw data have two parts, the set of suspects
and the set of confirmed suspects. This module has the purpose of unifying those sets in a
single list, that will be saved in cache memory, since it will be requested many times 
by both networks.

Each tuple of the raw data has the format:

Suspects  -> center_xyz, CT_scan_id
Confirmed -> center_xyz, CT_scan_id, diameter_mm

Each node in the confirmed set is in the suspect set, but this is not a vice versa situation.
Therefore, For every node in the suspect set, there is an equivalent candidate node:

Candidate -> is_nodule, diameter_mm, series_uid, center_xyz'

Where series_uid is the equivalent of CT_scan_id, "is_nodule" is a boolean that indicates 
if the suspect is in the confirmed set, and diameter_mm is the diameter of the nodule if 
the boolean is true. ( And equal to zero if false )
'''

#PYTHON STANDARD LIBRARY-----------------------------------------------------
import os
import glob
from collections import namedtuple
import functools
import csv

#THIRD PARTY MODULES--------------------------------------------------------
import numpy

def binary_search_tuple( seq , tup ):

    '''
    if seq is a sequence of sorted tuples and tup a new tuple, returs the position of where
    tup should be located in seq. Uses binary search to find the location 
    '''

    assert all( len( x ) == len( tup ) for x in seq )

    start = 0
    end = len( seq ) - 1
    while end > start:

        mid = ( start + end )//2
        if seq[ mid ] > tup:
            end = mid - 1 
        
        elif seq[ mid ] < tup:
            start = mid + 1 
        
        else:
            break
    
    return mid

def get_diameter_dict():

    '''
    this function will read every module listed in annotations.csv and return the following dict:

    nodule[ series_uid ] = ( diameter_list , coord_list )
    series_uid -> code of the CT scan
    diameter_list -> diameter of the nodule
    coord_list -> xyz position of the nodule
    '''

    diameter_dict = {}
    with open( "E:\Datasets\CT\\annotations.csv" , "r" ) as f:
        for row in list( csv.reader( f ) )[ 1: ]:
            series_uid = row[0]
            center_xyz = tuple([float(x) for x in row[1:4]])
            diameter_mm = float(row[4])
            diameter_dict[ series_uid ] = diameter_dict.get( series_uid , [] )
            diameter_dict[ series_uid ].append( ( diameter_mm , center_xyz ) )
    
    return diameter_dict


candidate_info = namedtuple( "Candidate_info" , 'is_nodule, diameter_mm, series_uid, center_xyz')
@functools.lru_cache(1)
def get_candidate_info_list( require_dsk = True ):
    mhd_list = glob.glob("E:\Datasets\CT\subset*\*.mhd")
    present_ondsk = {os.path.split(p)[-1][:-4] for p in mhd_list}

    nodule_dict = get_diameter_dict() #from annotations.csv
    candidate_lst = []
    with open( "E:\Datasets\CT\candidates.csv" , "r" ) as f:
        for row in list( csv.reader( f ) )[ 1: ]:
            
            series_uid = row[0]
            if require_dsk and not( series_uid in present_ondsk ):
                continue
            
            center_xyz = tuple([float(x) for x in row[1:4]])
            diameter_mm = 0.
            is_nod = bool( int( row[ 4 ] ) )

            if is_nod:
                p1 = numpy.array( center_xyz )
                for d , pt in nodule_dict[ series_uid ]:
                    p2 = numpy.array( pt )
                    dist = ( p1 - p2 )**2
                    if numpy.sqrt( dist.sum() ) <= d/4:
                        diameter_mm = d 
                        break
            candidate_lst.append( 
                candidate_info( is_nod , diameter_mm , series_uid , center_xyz ) 
                )
    candidate_lst.sort( reverse = True )
    return candidate_lst 




