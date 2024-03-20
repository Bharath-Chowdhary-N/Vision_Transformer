import einops

from torchsummary import summary

import torch
from torch import nn
import torchvision
import torch.optim as optim
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop, RandomRotation

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# image = (H,W,C)
# flattened_2d_patch = (N,P^2,C) where P^2 = H*W/N

patch_size=16
latent_size = 768 # Hidden size D
n_channels = 3
num_heads = 12
num_encoders = 12 
dropout = 0.1
num_classes=10
size=224

epochs=10
base_lr = 10e-3
weight_decay = 0.03
batch_size = 1

# Linear Layer implementation
class InputEmbedding(nn.Module):
    def __init__(self,patch_size=patch_size, n_channels=n_channels, device=device, latent_size=latent_size, batch_size=batch_size):
        super(InputEmbedding, self).__init__()
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.device = device
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.input_size = self.patch_size*self.patch_size*self.n_channels
        self.linearProjection = nn.Linear(self.input_size, self.latent_size)
        print("linear_projection:{}".format(self.linearProjection.weight.shape))
        self.class_token   = nn.Parameter(torch.randn(self.batch_size,1,self.latent_size).to(device))
        #self.pos_embedding = nn.Parameter(torch.randn(self.batch_size,1,self.latent_size).to(device))

    def forward(self, input_data):
        input_data = input_data.to(self.device)

        patches = einops.rearrange(input_data, 'b c (h h1) (w w1) -> b (h w) (h1 w1 c)', h1=self.patch_size, w1=self.patch_size)

        print(input_data.size())
        print(patches.size())

        linear_projection = self.linearProjection(patches).to(self.device)
        
        linear_projection = torch.cat((self.class_token, linear_projection), dim=1)
        
        s1, s2, s3 = linear_projection.shape

        self.pos_embedding = nn.Parameter(torch.randn(self.batch_size,s2,self.latent_size).to(device))

        print("pos embedding shape",self.pos_embedding.shape)

        self.linear_projection = linear_projection + self.pos_embedding

        return self.linear_projection

class EncoderBlock(nn.Module):
    def __init__(self, latent_size=latent_size, num_heads=num_heads, dropout=dropout, device=device):
        super(EncoderBlock, self).__init__()
        self.latent_size = latent_size
        self.num_heads = num_heads
        self.device= device
        self.dropout = dropout
        
        self.multihead = nn.MultiheadAttention(self.latent_size, self.num_heads, self.dropout)
        
        self.enc_MLP = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size*4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size*4, self.latent_size),
            nn.Dropout(self.dropout)
        )

        self.norm = nn.LayerNorm(self.latent_size)
    
    def forward(self, embedded_patches):
        firstnorm_out = self.norm(embedded_patches)
        attention_out = self.multihead(firstnorm_out, firstnorm_out, firstnorm_out)[0]

        first_added =embedded_patches + attention_out
        secondnorm_out = self.norm(first_added)
        ff_out = self.enc_MLP(secondnorm_out)
        output = ff_out + first_added
        
        return output
        
     
test_input = torch.randn(1,3,224,224)
test_class = InputEmbedding().to(device)
embed_test = test_class(test_input)

test_encoder = EncoderBlock().to(device)
test_encoder(embed_test)