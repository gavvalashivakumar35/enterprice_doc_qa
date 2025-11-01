import os, pickle
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
def build_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    return index, embs
def save_index(index, texts, index_path='data/faiss_docs.index', meta_path='data/faiss_docs_meta.pkl'):
    os.makedirs(os.path.dirname(index_path) or '.', exist_ok=True)
    faiss.write_index(index, index_path)
    with open(meta_path, 'wb') as f:
        pickle.dump(texts, f)
def load_index(index_path='data/faiss_docs.index', meta_path='data/faiss_docs_meta.pkl'):
    if not Path(index_path).exists():
        return None, None
    index = faiss.read_index(index_path)
    with open(meta_path, 'rb') as f:
        texts = pickle.load(f)
    return index, texts
