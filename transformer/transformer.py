from util import mask_

import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    '''
    This class is the self-attention unit of the transformer.
    This implements the very basic self-attention mechanism
    '''
    def __init__(self, embeddings, heads=8, mask=False):
        '''
        @params: embeddings -> This will represent the length of the embeddings.
        @params: heads -> This represents multi-head attention mechanism. 
        @params: mask -> 
        '''
        
        super().__init__()
        
        self.emb   = embeddings
        self.head  = heads
        self.mask  = mask
        
        self.toKeys     = nn.Linear(embeddings, embeddings * heads, bias=False)
        self.toQueries  = nn.Linear(embeddings, embeddings * heads, bias=False)
        self.toValues   = nn.Linear(embeddings, embeddings * heads, bias=False)
        
        self.unifyheads = nn.Linear(heads * embeddings, embeddings)

    def forward(self, x):
        b, t, embedSize = x.size()
        h = self.head

        keys    = self.toKeys(x).view(b, t, h, embedSize)
        queries = self.toQueries(x).view(b, t, h, embedSize)
        values  = self.toValues(x).view(b, t, h, embedSize)

        keys    = keys.transpose(1,2).contiguous.view(b*h,t,e)
        queries = queries.transpose(1,2).contiguous().view(b*h, t,e)
        values  = values.transpose(1,2).contiguous().view(b*h, t, e)

        # Getting the dot product of Queries & Keys 
        dotProd = torch.bnm(queries, keys.transpose(1,2))
        dotProd = dotProd / math.sqrt(embedSize)

        if self.mask:
            mask_(dotProd, maskval=float('-inf'), mask_diagonal=False)

        dotProd = F.softmax(dotProd, dim=2)

        if self.mask == 'first':
            dotProd = dotProd.clone()
            dotProd[:, :1, :] = 0.0

        output = torch.bnm(dotProd, values).view(b,h,t,e)

        output = output.transpose(1,2).contiguous().view(b,t,h*e)

        return self.unifyheads(output)