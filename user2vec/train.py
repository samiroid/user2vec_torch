import argparse
import numpy as np
from numpy.random import RandomState
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from user2vec import User2Vec 
from pathlib import Path
import os
import pickle
import ipdb 
SHUFFLE_SEED=10

def train(model, train_x, val_x, neg_samples, initial_lr, epochs, device):        
    # ipdb.set_trace() 
    model.to(device)    
    rng = RandomState(SHUFFLE_SEED)       
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    for e in range(epochs):        
        idx = rng.permutation(len(train_x))
        train_x_shuff = [train_x[i] for i in idx]
        neg_samples_shuff = [neg_samples[i] for i in idx]           
        running_loss = 0.0
        for x, neg_x in zip(train_x_shuff, neg_samples_shuff):    
            x_ = torch.from_numpy(x).long().to(device)
            neg_x_ = torch.tensor(neg_x).long().to(device)            
            optimizer.zero_grad()
            loss = model.forward(x_, neg_x_ )
            loss.backward()
            optimizer.step()                
            running_loss += loss.item() 
        avg_loss = round(running_loss/len(train_x),4)
        val_prob = 0
        for v in val_x:
            v_ = torch.from_numpy(v).long().to(device)
            val_prob += model.doc_logproba(v_).item()
        val_prob = round(val_prob/len(val_x),4)
        print("epoch: {} | loss: {} | val log prob: {} ".format(e, avg_loss, val_prob))
    return model

def main(path, output, epochs=20, initial_lr=0.001):
    E = np.load(path+"word_emb.npy")    
    E = torch.from_numpy(E.astype(np.float32)) 
    user_data = Path(path+"/users/").iterdir()

    for j, user in enumerate(user_data):
        with open(user, "rb") as fi:
            user_id, train_x, val_x, neg_samples = pickle.load(fi)
        print("{} | tr: {} | ts: {}".format(user_id,len(train_x), len(val_x)))
        model = User2Vec(user_id, E.T)    
        model = train(model, train_x, val_x, neg_samples, initial_lr, epochs, "cpu")
        model.save_embedding(output)

def cmdline_args():
    parser = argparse.ArgumentParser(description="Train User2Vec")
    parser.add_argument('-input', type=str, required=True, help='input folder')    
    parser.add_argument('-output', type=str, required=True, help='path of the output')    
    return parser.parse_args()	

if __name__ == "__main__" :    
    args = cmdline_args()
    print("[input: {} | output: {}]".format(os.path.basename(args.input), args.output))

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))   

    main(args.input, args.output)
    
