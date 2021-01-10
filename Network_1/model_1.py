import torch
import torch.nn as net
import torch.nn.functional as net_fun

class batch_norm( net.Module ):

    def __init__( self ):

        super().__init__()
        self.weight = net.Parameter( torch.rand(1) )
        self.bias = net.Parameter( torch.rand(1) )
    
    def forward( self , X ):

        X = X*self.weight + self.bias
        mu = X.mean()
        sigma = X.std()
        return ( X - mu )/sigma

class backbone_bloc( net.Module ):

    def __init__( self , c_in , c_out ):

        super().__init__()

        self.conv1 = net.Conv3d( c_in , c_out , kernel_size = 3 , padding = 1, bias = True )
        self.conv2 = net.Conv3d( c_out , c_out , kernel_size = 3 , padding = 1, bias = True )

    def forward( self , X ):

        X = self.conv1( X )
        X = net_fun.relu( X )
        X = self.conv2( X )
        X = net_fun.relu( X )
        return net_fun.max_pool3d( X , kernel_size = ( 2 , 2 , 2 ) )

class Fc_Classifier( net.Module ):

    def __init__ ( self , inpt , oupt , layers = None , act = net_fun.relu , prob_fun = net_fun.softmax ):

        super( Fc_Classifier , self ).__init__()

        self.inpt = inpt
        self.oupt = oupt

        if layers == None:
            layers = [] 

        if not hasattr( layers, "__iter__" ):
            raise ValueError( "if given, layers must be sequential" )
        
        if not isinstance( layers , list ):
            layers = list( *layers )
        
        layers = [ inpt ] + layers + [ oupt ]
        weight_list = net.ModuleList()
        # ( x1 , x2 , x3 , ... ) -> ( [ x1 , x2 ] , [ x2 , x3 ] , ... )
        for a , b in zip( layers[ : -1 ] , layers[ 1: ] ):
            weight_list.append( net.Linear( a , b ) )
        
        self.w_list = weight_list
        self.act = act
        self.prob_fun = prob_fun
    
    def forward( self , X ):

        if X.ndim == 1:
            X = X.unsqueeze( 0 )
        
        assert( X.shape[ 1 ] == self.inpt )

        y = X.clone()
        for i , w in enumerate( self.w_list ):
            z = w( y )
            # print( z )
            if i == len( self.w_list ) - 1:
                y = self.prob_fun( z , dim = 1 )
            else:
                y = self.act( z )
            # print( y )
        return y.squeeze()

class LunaModel(net.Module):

    def __init__( self ):
        super().__init__()
        
        #Tail-----------------------------------------
        self.tail_batchnorm = batch_norm()

        #Backbone------------------------------------
        self.block1 = backbone_bloc( 1 , 8 )
        self.block2 = backbone_bloc( 8 , 16 )
        self.block3 = backbone_bloc( 16 , 32 )
        self.block4 = backbone_bloc( 32 , 64 )

        #head-----------------------------------------
        self.network = Fc_Classifier( 1152 , 1 , [ 128 ] , prob_fun = net_fun.sigmoid )
    
    def forward( self , X ):

        X = self.tail_batchnorm( X )
        X = self.block1( X )
        X = self.block2( X )
        X = self.block3( X )
        X = self.block4( X )

        X_flat = X.view( X.size( 0 ) , -1 )
        return self.network( X_flat )