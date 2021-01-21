import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class User2Vec(nn.Module):

    def __init__(self, user_id, word_embeddings, margin=1):
        super(User2Vec, self).__init__()        
        self.user_id = user_id
        self.E = nn.Embedding.from_pretrained(word_embeddings, freeze=True)
        self.emb_dimension = word_embeddings.shape[1]
        self.U = nn.Embedding(1, self.emb_dimension)
        self.margin = margin
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.U.weight.data, -initrange, initrange)
                

    def forward(self, doc, neg_samples):
        #document embeddings
        emb_doc = self.E(doc)
        #negative samples embeddings
        emb_neg = self.E(neg_samples)
        #user embedding
        emb_user = self.U(torch.tensor([0]))                
        #document word probabilities
        logits = emb_doc @ emb_user.T
        #negative sample probabilities
        neg_logits = emb_neg @ emb_user.T
        #loss: max(0, margin - pos + neg)             
        zero_tensor = torch.tensor([0]).float().expand_as(logits)        
        loss = torch.max(zero_tensor, (self.margin - (logits + neg_logits)))
        return loss.mean()
    
    def doc_logproba(self, doc):
        #document embeddings
        emb_doc = self.E(doc)        
        #user embedding
        emb_user = self.U(torch.tensor([0]))                
        #document word probabilities
        logits = emb_doc @ emb_user.T        
        probs = F.logsigmoid(logits.squeeze())        
        return torch.mean(probs)        

    def save_embedding(self, output): 
        with open(output+self.user_id+".txt","w") as fo:
            embedding = self.U.weight.cpu().data.numpy()[0]                
            fo.write('%d %d\n' % (1, self.emb_dimension))            
            e = ' '.join(map(lambda x: str(x), embedding))
            fo.write('%s %s\n' % (self.user_id, e))
