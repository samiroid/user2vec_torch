import argparse
# import codecs
import pickle
from collections import Counter
from tadat.core import embeddings
import numpy as np
import os
import time
MIN_DOC_LEN=4
from ipdb import set_trace

class negative_sampler():

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


def get_wordcounts(path, min_word_freq=5, max_vocab_size=None):
    word_counter = Counter()
    n_docs=0	
    doc_lens = []
    with open(args.input) as fid:	
        for line in fid:			
            #discard first token (user id)            
            message = line.split()[1:]            
            word_counter.update(message)				
            n_docs+=1
            doc_lens+=[len(message)]
    #keep only words that occur at least min_word_freq times
    wc = {w:c for w,c in word_counter.items() if c>min_word_freq} 
    #keep only the max_vocab_size most frequent words
    tw = sorted(wc.items(), key=lambda x:x[1],reverse=True)
    vocab = {w[0]:i for i,w in enumerate(tw[:max_vocab_size])}	

    return vocab, word_counter, max(doc_lens)

def extract_word_embeddings(embeddings_path, vocab, encoding="latin-1"):
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

def save_user(user_id, user_txt, neg_sampler, rng, outpath, n_neg_samples, max_doclen, split=0.9):
    print("> saving user: {}".format(user_id))
    #shuffle the data
    shuf_idx = np.arange(len(user_txt))
    rng.shuffle(shuf_idx)
    user_txt_shuf = [user_txt[i] for i in shuf_idx]
    split_idx = int(len(user_txt)*split)
    user_txt_train = user_txt_shuf[:split_idx]
    user_txt_test = user_txt_shuf[split_idx:]

    train = []
    neg_samples = []
    for x in user_txt_train:        
        # x_arr = np.array(x, dtype=np.int32).reshape(1,-1)        
        #to include multiple negative samples per instance we will replicate samples so that 
        #the number of samples is the same as the number of negative samples
        #replicate each sample by the number of negative samples
        x_rep = np.tile(x,(n_neg_samples,1))
        #calculate negative samples (of max_doclen size)        
        neg_sample = neg_sampler.sample((n_neg_samples, len(x)))    
        train.append(x_rep)
        neg_samples.append(neg_sample)
        
    
    # neg_samples = [neg_sampler.sample(len(txt), n_neg_samples) for txt in user_txt_train]    
    with open(outpath+user_id, "wb") as fo:        
        pickle.dump([user_id, train, user_txt_test, neg_samples], fo)

def get_users(path, vocab, word_counts, random_seed, n_neg_samples, max_doclen, outpath):
    #negative sampler    
    sampler = negative_sampler(vocab, word_counts, warp=0.75)
    rng = np.random.RandomState(random_seed)    
    with open(path) as fi:
        #peek at the first line to get the first user
        curr_user, doc = fi.readline().replace("\"", "").replace("'","").split("\t")
        #read file from the start
        fi.seek(0,0)
        user_docs = []
        for line in fi:                        
            user, doc = line.replace("\"", "").replace("'","").split("\t")            
            #if we reach a new user, save the current one
            if user!= curr_user:                
                save_user(curr_user, user_docs, sampler, rng, outpath, n_neg_samples, max_doclen)
                #reset current user
                curr_user = user
                user_docs = []  
            doc = doc.split(" ")            
            doc_idx = [vocab[w] for w in doc if w in vocab]			    
            if len(doc_idx) < MIN_DOC_LEN: continue
            #accumulate all texts
            user_docs.append(np.array(doc_idx, dtype=np.int32).reshape(1,-1))        
        #save last user
        save_user(curr_user, user_docs, sampler, rng, outpath, n_neg_samples, max_doclen)

def cmdline_args():
    parser = argparse.ArgumentParser(description="Build Training Data")
    parser.add_argument('-input', type=str, required=True, help='train file(s)')
    parser.add_argument('-emb', type=str, required=True, help='path to word embeddings')
    parser.add_argument('-output', type=str, required=True, help='path of the output')
    parser.add_argument('-vocab_size', type=int, help='max size of vocabulary')
    parser.add_argument('-min_word_freq', type=int, help='ignore words that occur less than min_word_freq times',default=5)
    parser.add_argument('-seed', type=int, default=1234, help='random number generator seed')    
    parser.add_argument('-neg_samples', type=int, help='number of negative samples', default=10)
    return parser.parse_args()	

if __name__ == "__main__" :    
    args = cmdline_args()
    print("[input: {} | word vectors: {} | max vocab_size: {} | min_word_freq: {} | output: {}]".format(os.path.basename(args.input), 
                            os.path.basename(args.emb), 
                            args.vocab_size, 
                            args.min_word_freq, 
                            args.output))

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))   
    
    vocab = None
    word_counter = None
    try:
        with open(args.output+"/vocab.pkl", "rb") as fi:        
            z = pickle.load(fi)
        if z:
            print("[vocabulary found]")
            vocab, word_counter, max_doclen = z
    except FileNotFoundError:
        pass

    if not vocab:
        print(" > extracting vocabulary")
        vocab, word_counter, max_doclen = get_wordcounts(args.input, args.min_word_freq, 
                                                        args.vocab_size)        
        print(" > loading word embeddings")
        E, vocab_redux = extract_word_embeddings(args.emb, vocab, encoding="latin-1")    
        print("[vocabulary size: {} > {} | max_doclen: {}]".format(len(vocab),
                                                                    len(vocab_redux),
                                                                    max_doclen))
        with open(args.output+"/word_emb.npy", "wb") as f:
            np.save(f, E)    
        with open(args.output+"/vocab.pkl", "wb") as f:
            pickle.dump([vocab_redux, word_counter, max_doclen], f, pickle.HIGHEST_PROTOCOL)
        vocab = vocab_redux
    print(" > reading users")
    if not os.path.exists(os.path.dirname(args.output+"/users/")):
        os.makedirs(os.path.dirname(args.output+"/users/"))   

    get_users(args.input, vocab, word_counter, args.seed, args.neg_samples, 
            max_doclen, args.output+"/users/")
