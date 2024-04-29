import gzip
import numpy as np
import re
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

#tokenizer = ByteLevelBPETokenizer()
tokenizer = ByteLevelBPETokenizer.from_file(vocab_filename='bl_bpe/vocab.json', merges_filename='bl_bpe/merges.txt' )

file = gzip.open("./data/enwik8.gz")
data = file.read()
print('enwik8 tokenization in progress...')
data = np.concatenate([ np.array(tokenizer.encode(s).ids, dtype=np.int16 ) for s in tqdm(re.split('([^\t\ \n\r]+[\ \n\t\r])', str(data) )) ])
print('saving to ./data/enwik8.npy ...')
np.save('./data/enwik8.npy', data )
print('done.')
