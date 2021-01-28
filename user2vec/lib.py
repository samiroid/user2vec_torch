import argparse
# import codecs
import pickle
from collections import Counter
from tadat.core import embeddings
import numpy as np
import os
import time
from ipdb import set_trace
import torch
from pathlib import Path
from model import User2Vec
import shutil
import random

MIN_DOC_LEN=2
MIN_DOCS = 50

class NegativeSampler():

    def __init__(self, vocab, word_count, warp=0.75, n_neg_samples=5):
        '''
        Store count for the range of indices in the dictionary
        '''
        self.n_neg_samples = n_neg_samples
        index2word = {i:w for w,i in vocab.items()}	
        # set_trace()
        max_index = max(index2word.keys())
        counts = []
        for n in range(max_index):
            if n in index2word:
                counts.append(word_count[index2word[n]]**warp)
            else:    
                counts.append(0)
        counts = np.array(counts)
        norm_counts = counts/sum(counts)
        scaling = int(np.ceil(1./min(norm_counts[norm_counts>0])))
        scaled_counts = (norm_counts*scaling).astype(int)
        self.cumsum = scaled_counts.cumsum()   

    def sample(self, size):        
        total_size = np.prod(size)
        random_ints = np.random.randint(self.cumsum[-1], size=total_size)
        data_y_neg = np.searchsorted(self.cumsum, random_ints).astype('int32')
        return data_y_neg.reshape(size) 

    def sample_filtered(self, exclude, size):        
        total_size = np.prod(size)
        random_ints = np.random.randint(self.cumsum[-1], size=total_size)
        data_y_neg = np.searchsorted(self.cumsum, random_ints).astype('int32')
        #filter out words that should be excluded 
        filtered = [x for x in data_y_neg.tolist() if x not in exclude][:total_size]
        data_y_neg = np.array(filtered)
        return data_y_neg.reshape(size) 


def get_vocabulary(inpath, min_word_freq=5, max_vocab_size=None):    
    print(" > extracting vocabulary")
    word_counter = Counter()
    n_docs=0	
    # doc_lens = []
    with open(inpath) as fid:	
        for line in fid:			
            #discard first token (user id)            
            message = line.split()[1:]            
            word_counter.update(message)				
            n_docs+=1
            # doc_lens+=[len(message)]
    #keep only words that occur at least min_word_freq times
    wc = {w:c for w,c in word_counter.items() if c>min_word_freq} 
    #keep only the max_vocab_size most frequent words
    tw = sorted(wc.items(), key=lambda x:x[1],reverse=True)
    vocab = {w[0]:i for i,w in enumerate(tw[:max_vocab_size])}	

    return vocab, word_counter

def extract_word_embeddings(embeddings_path, vocab, encoding="latin-1"):    
    print(" > loading word embeddings")
    full_E, full_wrd2idx = embeddings.read_embeddings(embeddings_path, vocab, encoding)
    ooevs = embeddings.get_OOEVs(full_E, full_wrd2idx)
    #keep only words with pre-trained embeddings    
    for w in ooevs:
        del vocab[w]	
    vocab_redux = {w:i for i,w in enumerate(vocab.keys())}	    
    #generate the embedding matrix
    emb_size = full_E.shape[0]
    E = np.zeros((int(emb_size), len(vocab_redux)))   
    for wrd,idx in vocab_redux.items(): 
        E[:, idx] = full_E[:,vocab[wrd]]	       
    
    return E, vocab_redux

def save_user(user_id, user_txt, negative_sampler, rng, outpath, n_samples, split=0.8):    
    if len(user_txt) > MIN_DOCS:
        print("> saving user: {}  ({})".format(user_id,len(user_txt)))
        #shuffle the data
        shuf_idx = np.arange(len(user_txt))
        rng.shuffle(shuf_idx)
        user_txt_shuf = [user_txt[i] for i in shuf_idx]
        split_idx = int(len(user_txt)*split)
        user_txt_train = user_txt_shuf[:split_idx]
        validation_samples = user_txt_shuf[split_idx:]

        pos_samples = []
        neg_samples = []
        for x in user_txt_train:        
            #replicate each sample by the number of negative samples so that the number
            # of samples is the same as the number of negative samples
            x_rep = np.tile(x,(n_samples,1))
            #calculate negative samples 
            neg_sample = negative_sampler.sample((n_samples, len(x)))    
            pos_samples.append(x_rep)
            neg_samples.append(neg_sample)            
        
        with open(outpath+user_id, "wb") as fo:        
            pickle.dump([user_id, pos_samples, neg_samples, validation_samples], fo)
    else:
        print("> IGNORED user: {}  ({})".format(user_id,len(user_txt)))

def build_data(inpath, outpath, embeddings_path, emb_encoding="latin-1", 
                min_word_freq=5, max_vocab_size=None, random_seed=123, n_neg_samples=10, reset=False):
    pkl_path=outpath+"pkl/"
    users_path=pkl_path+"users/"  

    if reset:
        shutil.rmtree(pkl_path, ignore_errors=True)
        shutil.rmtree(users_path, ignore_errors=True)

    if not os.path.exists(os.path.dirname(users_path)):
        os.makedirs(os.path.dirname(users_path))   

    vocab = None
    word_counts = None
    try:
        with open(pkl_path+"vocab.pkl", "rb") as fi:        
            vocab, word_counts = pickle.load(fi)        
            print("[found cached vocabulary]")
    except FileNotFoundError:
        pass

    if not vocab:
        #compute vocabulary
        vocab, word_counts = get_vocabulary(inpath, min_word_freq=min_word_freq,max_vocab_size=max_vocab_size)        
        #extract word embeddings
        E, vocab_redux = extract_word_embeddings(embeddings_path, vocab, encoding=emb_encoding)
        #vocab_redux has only words for which an embedding was found
        print("[vocab size: {} > {}]".format(len(vocab),len(vocab_redux)))
        vocab = vocab_redux
        with open(pkl_path+"word_emb.npy", "wb") as f:
            np.save(f, E)    
        with open(pkl_path+"vocab.pkl", "wb") as f:
            pickle.dump([vocab_redux, word_counts], f, pickle.HIGHEST_PROTOCOL)
        
    #negative sampler    
    sampler = NegativeSampler(vocab, word_counts, warp=0.75)
    rng = np.random.RandomState(random_seed)    
    with open(inpath) as fi:
        #peek at the first line to get the first user
        curr_user, doc = fi.readline().replace("\"", "").replace("'","").split("\t")
        #read file from the start
        fi.seek(0,0)
        user_docs = []
        for line in fi:                        
            user, doc = line.replace("\"", "").replace("'","").split("\t")            
            #if we reach a new user, save the current one
            if user!= curr_user:                
                save_user(curr_user, user_docs, sampler, rng, users_path, n_neg_samples)
                #reset current user
                curr_user = user
                user_docs = []  
            doc = doc.split(" ")            
            doc_idx = [vocab[w] for w in doc if w in vocab]			    
            if len(doc_idx) < MIN_DOC_LEN: continue
            #accumulate all texts
            user_docs.append(np.array(doc_idx, dtype=np.int32).reshape(1,-1))        
        #save last user
        save_user(curr_user, user_docs, sampler, rng, users_path, n_neg_samples)

def train_model(path,  epochs=20, initial_lr=0.001, margin=1, reset=False):
    txt_path = path+"/txt/"    
    if reset:
        shutil.rmtree(txt_path, ignore_errors=True)
    
    if not os.path.exists(os.path.dirname(txt_path)):
        os.makedirs(os.path.dirname(txt_path))   

    E = np.load(path+"/pkl/word_emb.npy")    
    E = torch.from_numpy(E.astype(np.float32)) 
    user_data = list(Path(path+"/pkl/users/").iterdir())
    random.shuffle(user_data)
    cache = set([os.path.basename(f).replace(".txt","") for f in Path(txt_path).iterdir()])
    
    for j, user_fname in enumerate(user_data):
        user = os.path.basename(user_fname) 
        if user in cache:
            print("cached embedding: {}".format(user))
            continue        
        with open(user_fname, "rb") as fi:
            user_id, pos_samples, neg_samples, val_samples = pickle.load(fi)
        
        print("{} | tr: {} | ts: {}".format(user_id,len(pos_samples), len(val_samples)))
        model = User2Vec(user_id, E.T, txt_path, margin=margin, initial_lr=initial_lr, epochs=epochs)    
        model.fit(pos_samples, neg_samples, val_samples)


# def cmdline_args():
#     parser = argparse.ArgumentParser(description="Build Training Data")
#     parser.add_argument('-input', type=str, required=True, help='train file(s)')
#     parser.add_argument('-emb', type=str, required=True, help='path to word embeddings')
#     parser.add_argument('-output', type=str, required=True, help='path of the output')
#     parser.add_argument('-vocab_size', type=int, help='max size of vocabulary')
#     parser.add_argument('-min_word_freq', type=int, help='ignore words that occur less than min_word_freq times',default=5)
#     parser.add_argument('-seed', type=int, default=1234, help='random number generator seed')    
#     parser.add_argument('-neg_samples', type=int, help='number of negative samples', default=10)
#     return parser.parse_args()	

# def main(args):
#     if not os.path.exists(os.path.dirname(args.output)):
#         os.makedirs(os.path.dirname(args.output))   
    
#     vocab = None
#     word_counter = None
#     try:
#         with open(args.output+"/vocab.pkl", "rb") as fi:        
#             z = pickle.load(fi)
#         if z:
#             print("[vocabulary found]")
#             vocab, word_counter, max_doclen = z
#     except FileNotFoundError:
#         pass

#     if not vocab:
#         print(" > extracting vocabulary")
#         vocab, word_counter, max_doclen = get_vocabulary(args.input, args.min_word_freq, 
#                                                         args.vocab_size)        
#         print(" > loading word embeddings")
#         E, vocab_redux = extract_word_embeddings(args.emb, vocab, encoding="latin-1")    
#         print("[vocabulary size: {} > {} | max_doclen: {}]".format(len(vocab),
#                                                                     len(vocab_redux),
#                                                                     max_doclen))
#         with open(args.output+"/word_emb.npy", "wb") as f:
#             np.save(f, E)    
#         with open(args.output+"/vocab.pkl", "wb") as f:
#             pickle.dump([vocab_redux, word_counter, max_doclen], f, pickle.HIGHEST_PROTOCOL)
#         vocab = vocab_redux
#     print(" > reading users")
#     if not os.path.exists(os.path.dirname(args.output+"/users/")):
#         os.makedirs(os.path.dirname(args.output+"/users/"))   

#     build_data(args.input, vocab, word_counter, args.seed, args.neg_samples, 
#              args.output+"/users/")


# if __name__ == "__main__" :    
#     arg = cmdline_args()
#     print("[input: {} | word vectors: {} | max vocab_size: {} | min_word_freq: {} | output: {}]".format(os.path.basename(arg.input), 
#                             os.path.basename(arg.emb), 
#                             arg.vocab_size, 
#                             arg.min_word_freq, 
#                             arg.output))
#     main(arg)
    