import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from numpy.random import RandomState

SHUFFLE_SEED=10

class User2Vec(nn.Module):

    def __init__(self, user_id, word_embeddings, outpath, margin=1, initial_lr=0.001, epochs=10, device=None):
        super(User2Vec, self).__init__()        
        self.user_id = user_id
        self.outpath = outpath
        self.E = nn.Embedding.from_pretrained(word_embeddings, freeze=True)
        self.emb_dimension = word_embeddings.shape[1]
        self.U = nn.Embedding(1, self.emb_dimension)
        self.margin = margin
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.U.weight.data, -initrange, initrange)
        self.initial_lr = initial_lr
        self.epochs = epochs
        if not device:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device               

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
        loss = torch.max(zero_tensor, (self.margin - logits + neg_logits))
        return loss.mean()
    
    def doc_logproba(self, doc):
        #document embeddings
        emb_doc = self.E(doc)        
        #user embedding
        emb_user = self.U(torch.tensor([0]))                
        #document word probabilities
        logits = emb_doc @ emb_user.T        
        probs = torch.sigmoid(logits.squeeze())        
        return torch.mean(probs)        

    def fit(self, train_x, neg_samples, val_x):        
        # ipdb.set_trace()         
        self.to(self.device)
        rng = RandomState(SHUFFLE_SEED)       
        optimizer = optim.Adam(self.parameters(), lr=self.initial_lr)

        val_prob=0
        best_val_prob=0    
        n_val_drops=0   
        MAX_VAL_DROPS=5
        loss_margin = 0.005      

        for e in range(self.epochs):        
            idx = rng.permutation(len(train_x))
            train_x_shuff = [train_x[i] for i in idx]
            neg_samples_shuff = [neg_samples[i] for i in idx]           
            running_loss = 0.0
            for x, neg_x in zip(train_x_shuff, neg_samples_shuff):    
                x_ = torch.from_numpy(x).long().to(self.device)
                neg_x_ = torch.tensor(neg_x).long().to(self.device)            
                optimizer.zero_grad()
                loss = self.forward(x_, neg_x_ )
                loss.backward()
                optimizer.step()                
                running_loss += loss.item() 
            avg_loss = round(running_loss/len(train_x),4)
            val_prob = 0
            for v in val_x:
                v_ = torch.from_numpy(v).long().to(self.device)
                val_prob += self.doc_logproba(v_).item()
            val_prob = round(val_prob/len(val_x),4)
            print("epoch: {} | loss: {} | val avg prob: {} ".format(e, avg_loss, val_prob))                
            if val_prob > best_val_prob:    
                n_val_drops=0            
                best_val_prob = val_prob
                self.save_embedding()
                #save best model
                # print("[updating best model]")
                # torch.save(self.model.state_dict(), tmp_model_fname)
            elif val_prob < (best_val_prob - loss_margin):                
                n_val_drops+=1
                if n_val_drops == MAX_VAL_DROPS:
                    print("[early stopping: {} epochs]".format(e))
                    break


    def save_embedding(self): 
        with open(self.outpath+self.user_id+".txt","w") as fo:
            embedding = self.U.weight.cpu().data.numpy()[0]                
            fo.write('%d %d\n' % (1, self.emb_dimension))            
            e = ' '.join(map(lambda x: str(x), embedding))
            fo.write('%s %s\n' % (self.user_id, e))
