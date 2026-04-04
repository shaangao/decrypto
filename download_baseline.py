import sys
import os
sys.path.append(os.getcwd())
from src.agents.embedding_baseline import preprocess_word2vec, load_glove_embeddings

data_dir = os.path.abspath('embedding_models')
os.makedirs(data_dir, exist_ok=True)
print(f'Starting downloads into {data_dir}...')

print('\n--- Downloading GloVe ---')
load_glove_embeddings(os.path.join(data_dir, 'glove.840B.300d.txt'), data_dir)

print('\n--- Downloading Word2Vec ---')
preprocess_word2vec(data_dir)
print('\nDone!')