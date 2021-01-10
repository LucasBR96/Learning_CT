import os
from collections import namedtuple
import pickle
import gc

import torch
import torch.utils as utils
import torch.nn as net
import numpy

from luna_model import LunaModel
from ct_mat import class_dset

class Luna_Trainer:

    progress_tup = namedtuple( 'cost_tuple' , [ 'iter' , 'train_cost' , 'test_cost' ] )
    rec_tup = namedtuple( 'precision_tuple' , [ 'iter' , 'pos_prec' , 'neg_prec' ] )

    def __init__( self , *args , **kwargs ):

        '''
        hosts , trains and evaluates a LunaModel
        '''

        self.device = self.init_device( 
            kwargs.get( "device" , "cuda" )
        )
        self.mod = self.init_model( 
            kwargs.get( "mod" , None)
            )
        self.opm = self.init_opm(
            kwargs.get( "opm_type" , torch.optim.Adam ),
            kwargs.get( "lr" , 1e-5 )
        )

        self.batch_size = kwargs.get( "batch_size" , 20 )
        self.threshold = numpy.clip( 
            kwargs.get( "threshold" , .5 ),
            .5 , .9 )

        self.it_num = 0
        self.init_iterator( False )
        self.init_iterator( True )
        
    def init_device( self , dev = "cuda" ):
        
        if not( dev in ( "cuda" , "cpu" ) ):
            return torch.device( "cpu" )

        if not( dev == "cuda" and torch.cuda.is_available() ):
            dev = "cpu"
        return torch.device( dev )

    def init_model( self , model = None ):
        
        if not isinstance( model , LunaModel ):
            model = None
        
        if model is None:
            model = LunaModel( ).to( self.device )
        
        return model

    def init_opm( self , opm_type , lr ):
        
        if not isinstance( opm_type , torch.optim.Optimizer ):
            opm_type = torch.optim.Adam
        
        return opm_type( self.mod.parameters() , lr = lr )
        
    def init_iterator( self , val = False ):

        d_set = class_dset( val , balance = True , augment = not val )
        batch_size = self.batch_size
        pin_mem = ( self.device.type == 'cuda' )

        d_loader = utils.data.DataLoader( 
            dataset = d_set,
            batch_size = batch_size,
            pin_memory = pin_mem,
            num_workers = 0
        )
        if val:
            self.val_loader = iter( d_loader )
        else:
            self.test_loader = iter( d_loader )
    
    def sample( self, val = False ):

        loader = self.val_loader if val else self.test_loader
        try:
            return next( loader )
        except StopIteration:
            self.init_iterator( val )
            return self.sample( val )
    
    def iter_run( self , epochs = 1000 ):

        self.mod.train()
        costs = []
        for i in range( epochs ):

            self.it_num += 1
            if bool( self.it_num%5 ): #training 99 every 100 epochs

                box , label = self.sample( False )
                loss = self.get_loss( box , label )
                self.opm.zero_grad()
                loss.backward()
                self.opm.step()
            
            else:
                self.opm.zero_grad()
                self.mod.eval()
                box , label = self.sample( True )
                eval_result = self.get_loss( box , label , is_val = True )
                eval_result = eval_result.detach().cpu().numpy()
                print( "Accr at it {}: {}%".format( 
                    self.it_num ,
                    numpy.round( eval_result, decimals = 2 ))
                )
                costs.append( eval_result )
                self.mod.train()

        return costs


    def get_loss( self , box_t , class_t , is_val = False ):

        box_g = box_t.to( self.device )
        class_g = class_t.to( self.device )
        pred = self.mod( box_g )
        if is_val:
            pred = torch.ceil( pred - .5 ).abs()
            matches = ( pred + class_g + 1 )%2
            result = 100*matches.mean()
        else:
            result = class_g*torch.log( pred ) + ( 1 - class_g )*torch.log( 1 - pred )
            result = -result.mean()
        return result
                    
if __name__ == "__main__":
    app = Luna_Trainer()
    app.iter_run( epochs = 30 )


    
    
