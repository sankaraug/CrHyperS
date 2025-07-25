import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout = dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth-2):
            self.skipcat.append(nn.Conv2d(num_channel+1, num_channel+1, [1, 2], 1, 0))

    def forward(self, x, mask = None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask = mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:           
                last_output.append(x)
                if nl > 1:             
                    x = self.skipcat[nl-2](torch.cat([x.unsqueeze(3), last_output[nl-2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, mask = mask)
                x = ff(x)
                nl += 1

        return x



"""Spatial Attention Block"""
class SpatialSELayer3D(nn.Module):

    
    def __init__(self, num_channels = 300):
        """:param num_channels: No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(1, 1, (1, 1 , num_channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        
        x =  input_tensor 
        x = self.conv(x)
        squeeze_tensor = self.sigmoid(x)
        # spatial excitation
        output_tensor =   torch.mul(input_tensor, squeeze_tensor)   #squeeze_tensor #
        return output_tensor
    
# model = SpatialSELayer3D(BAND).cuda()
# summary(model, (1, img_rows, img_cols, BAND))


# """SE_ECA block"""
# class SE_ECA_block(nn.Module):
#     def masks(self, input_tensor):
#         #generate mask
#         def ndvi(X):
#             ndvi = (X[:,:,:,:,199]-X[:,:,:,:,143])/(X[:,:,:,:,199]+X[:,:,:,:,143])
#             return ndvi
#         mask1 = ndvi(input_tensor)
#         mask1[ mask1 <= 0.6] = 0
#         mask1[ mask1 > 0.6] = 1
#         masks = mask1.unsqueeze(-1)        
#         return masks    
    
#     def __init__(self, num_channels =300, gamma =2, b =1):
#         """
#         num_channels: Number of channels of the input feature map
#         k: Adaptive selection of kernel size
#         """
#         super(SE_ECA_block, self).__init__()
#         self.num_channels = num_channels 
        
#         #finding the size of the 1D-CNN kernel
#         import math
#         t = int(abs((math.log(num_channels, 2) + b)/gamma))
#         k = t if t % 2 else t+1
        
#         """ Set K directly"""
#         #k = 7
        
#         self.avg_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, num_channels))
#         self.conv1D = nn.Conv1d( in_channels=1, out_channels=1, kernel_size= k, padding = int(k/2), bias = False)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, input_tensor):
#         """
#         feature descriptor on the global spatial information
#         """
#         batch_size, num_channels, H, W, D = input_tensor.size()
#         #y = torch.multiply(input_tensor, self.masks(input_tensor))
#         # Average along each channel
#         y = self.avg_pool(input_tensor)
#         y = torch.squeeze(y, dim=(2))
#         y = torch.squeeze(y, dim=(2))
       
#         # channel excitation with 1D CNN
#         y = self.conv1D (y)
#         y = self.relu(y)   # this is not used in the ECA net paper
#         y = self.sigmoid(y.unsqueeze(dim=2).unsqueeze(dim=2))
        
#         output_tensor = torch.mul(input_tensor, y)

#         return output_tensor


# """The back bone 3D CNN network implementation"""

"""The back bone 3D CNN network implementation"""

class CNN3D_network(nn.Module):
    def __init__(self):
        super(CNN3D_network, self).__init__()
        self.name = '3D_CNN'
        self.channels = 300 
        self.SSEblock = SpatialSELayer3D(self.channels)
       

        
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16,  kernel_size=(3, 3, 7))
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32,  kernel_size=(3, 3, 3))
        self.relu = nn.ReLU()   
        self.dense1 = nn.Linear(9344, 64)             
     
     
    def forward(self, input_tensor):
        
        x =  input_tensor   #self.SSEblocklock(input_tensor) 
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.contiguous().view(x.size(0), -1)
        out = self.dense1(x)
        
        return out
    

class ViT(nn.Module):

    def __init__(self, image_size, near_band, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1, dim_head = 16, dropout=0., emb_dropout=0., mode='ViT'):
        super().__init__()

        patch_dim = image_size ** 2 * near_band                   # Change can be made @Sankar
        self.SSEblock = SpatialSELayer3D()
        # self.ECAblock = SE_ECA_block()
        self.S3DCNN = CNN3D_network()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(2*dim),
            nn.Linear(2*dim, num_classes)
        )
        
        self.conv_1by2 = nn.Conv2d(in_channels=1, out_channels=1,  kernel_size=(1,2))
    

    def forward(self, input_tensor, mask = None):
        """ ### For  """
        #print(input_tensor.shape)    # torch.Size([32, 1, 5, 5, 300])
        input_tensor =  self.SSEblock(input_tensor) 
        y = self.S3DCNN(input_tensor)
       
        #input_tensor = input_tensor.permute(0, 4, 2, 3, 1)
        

        """ ### For SpectralFormer """
         
        x = input_tensor.view(input_tensor.shape[0], input_tensor.shape[1]*input_tensor.shape[4],  input_tensor.shape[2]*input_tensor.shape[3])
                           
        x = self.patch_to_embedding(x) #[b,n,dim]
        b, n, _ = x.shape
        #print("shape patch to emeding:" + str(x.shape))

        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b) #[b,1,dim]
        x = torch.cat((cls_tokens, x), dim = 1) #[b,n+1,dim]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x, mask)
        
        # classification: using cls_token output
        x = self.to_latent(x[:,0])
        #print("x shape:" + str(x.shape))
        # MLP classification layer
        z =   torch.cat((x, y), dim=1)  # # x+y
     
        # z = z.view(z.shape[0], 1,  64, 2) 
        # z = self.conv_1by2(z)
        # z = z.view(z.shape[0], z.shape[2]) 
 
        return self.mlp_head(z)







