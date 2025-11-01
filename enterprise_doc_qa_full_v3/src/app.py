import streamlit as st, os, pickle, faiss, time
from pathlib import Path
from document_loader import load_file
from vector_store import build_embeddings, save_index, load_index
from query_engine import generate_answer
st.set_page_config(page_title='Enterprise Document Q&A')
st.title('Enterprise Document Q&A (PDF/DOCX)')
DATA_DIR = Path('data/sample_docs')
if st.button('Index documents'):
    files = list(DATA_DIR.rglob('*.*'))
    texts = []
    file_map = []
    for f in files:
        text = load_file(str(f))
        if text and len(text.strip())>0:
            # simple chunking by 1000 chars
            for i in range(0, len(text), 1000):
                chunk = text[i:i+1000]
                texts.append(chunk)
                file_map.append(str(f.name))
    if texts:
        index, embs = build_embeddings(texts)
        save_index(index, list(zip(file_map, texts)), 'data/faiss_docs.index', 'data/faiss_docs_meta.pkl')
        st.success(f'Indexed {len(texts)} chunks from {len(files)} files')
    else:
        st.warning('No text extracted from documents')
else:
    st.write('Click "Index documents" to build embeddings from data/sample_docs')
q = st.text_input('Ask a question based on the uploaded documents:')
if st.button('Ask') and q:
    idx, meta = load_index('data/faiss_docs.index', 'data/faiss_docs_meta.pkl')
    if idx is None:
        st.warning('Index not found. Please click "Index documents" first.')
    else:
        t0 = time.time()
        # load texts and search
        with open('data/faiss_docs_meta.pkl','rb') as f:
            import pickle
            meta_list = pickle.load(f)
        # meta_list is list of (filename, text)
        texts = [t for (_,t) in meta_list]
        import numpy as np
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        q_emb = encoder.encode([q], convert_to_numpy=True)
        D,I = idx.search(q_emb,5)
        contexts = [texts[i] for i in I[0]]
        ans = generate_answer(q, contexts)
        st.subheader('Answer')
        st.write(ans)
        st.subheader('Retrieved contexts (filename and excerpt)')
        for i in I[0]:
            fn, txt = meta_list[i]
            st.write(f'**{fn}**: {txt[:400]}...')
        st.caption(f'Retrieval+LLM time: {time.time()-t0:.2f}s')