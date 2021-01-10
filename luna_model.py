import torch
import torch.nn as net
import torch.nn.functional as net_fun

class gaussian_MM( net.Module ):

    def __init__( self , n , k ):
        '''
        n -> gaussian size
        k -> number of gaussians
        '''
        super().__init__()
        self.n = n
        self.k = k

        self.mu = net.Parameter( torch.rand( k , n ) )
        self.sigma = net.Parameter( torch.rand( k , n , n ) )
        self.weights = net.Parameter( torch.random( k ) )
    
    def forward( self , X ):

        if X.ndim == 1:
            X = X.unsqueeze( 0 )
        
        diff = X.unsqueeze( 1 ) - self.mu
        #diff.shape = ( X.shape[ 0 ] , self.k , self.n )
        sigma_det = torch.cat( [ torch.det( sig ) for sig in self.sigma] )
        #sigma_det.shape = ( self.k , )
        z = torch.sum( diff**2 , axis = 2 )/sigma_det

        norm = torch.distributions.Normal( 0 , 1 )
        prob = torch.exp( norm.log_prob( z ) )
        #prob.shape = ( X.shape[ 0 ] , self.k )
        weights = net_fun.softmax( self.weights.unsqueeze( 0 ),
        dim = 1 )

        return torch.sum( prob*weights , axis = 1 )

class compressor_mod2d( net.Module ):

    def __init__( self , c_in , c_out ):
        super().__init__()
        
        self.conv1 = net.Conv2d( c_in , c_out , kernel_size = 3 , padding = 1, bias = True )
        self.conv2 = net.Conv2d( c_out , c_out , kernel_size = 3 , padding = 1, bias = True )

    def forward( self , X ):

        X = self.conv1( X )
        X = net_fun.relu( X )
        X = self.conv2( X )
        X = net_fun.relu( X )
        return net_fun.max_pool2d( X , kernel_size = ( 2 , 2 ) )        

class decompressor_mod2d( net.Module ):

    def __init__( self , c_in , c_out ):
        super().__init__()
        
        self.usp = net.Upsample( scale_factor = ( 2 , 2 ) )
        self.conv1 = net.Conv2d( c_in , c_out , kernel_size = 3 , padding = 1, bias = True )
        self.conv2 = net.Conv2d( c_out , c_out , kernel_size = 3 , padding = 1, bias = True )
    
    def forward( self , X ):

        X = self.usp( X )
        X = self.conv1( X )
        X = net_fun.relu( X )
        X = self.conv2( X )
        return net.ReLU( X )

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



class U_model( net.Module ):

    def __init__( self ):
        super().__init__()
        
        #Tail-----------------------------------------
        self.tail_batchnorm = batch_norm()

        #compress network-----------------------------
        self.compress_net = net.ModuleList(
            compressor_mod2d( 1 , 2 ) ,
            compressor_mod2d( 2 , 2 ) ,
            compressor_mod2d( 2 , 4 ) ,
            compressor_mod2d( 4 , 4 ) ,
            compressor_mod2d( 4 , 8 )
        )

        #decompress network-----------------------------
        self.decompress_net = net.ModuleList(
            decompressor_mod2d( 8 , 8 ) ,
            decompressor_mod2d( 8 , 4 ) ,
            decompressor_mod2d( 4 , 4 ) ,
            decompressor_mod2d( 4 , 2 ) ,
            decompressor_mod2d( 2 , 1 )
        )
    
    def foward( self , X ):

        X = self.tail_batchnorm( X )
        for mod in self.compress_net:
            X = mod( X )
        for mod in self.decompress_net:
            X = mod( X )
        return net_fun.sigmoid( X )