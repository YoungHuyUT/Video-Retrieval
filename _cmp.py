import numpy as np
from pathlib import Path
from aic2026.embeddings import OpenCLIPTextEmbedder
from aic2026.ingestion import load_manifest
from aic2026.retrieval.factory import load_index_for_query

emb = OpenCLIPTextEmbedder(pretrained='openai')
records = load_manifest(Path('data/processed/derived_manifest.jsonl'))
index = load_index_for_query(Path('data/processed/derived_features.npy'), records, backend='auto')

queries = ['mot con trau', 'Image of a cow', 'a buffalo', 'Hinh anh con trau']
embeds = {q: emb.encode(q) for q in queries}

# cosine matrix
def cos(a,b):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b); return float(a@b)
lines = []
lines.append('=== Cosine similarity ===')
for q in queries:
    row = '  '.join(f'{cos(embeds[q],embeds[o]):.2f}' for o in queries)
    lines.append(f'{q:22} | {row}')

lines.append('')
lines.append('=== Top-3 video for each query ===')
for q in queries:
    ids, scores = index.search(embeds[q], 3)
    top = [(records[int(i)].video_id, round(float(sc),3)) for i,sc in zip(ids[:3], scores[:3])]
    lines.append(f'{q:22} -> {top}')

open('D:/MinhHuy/AIC2026/_cmp_out.txt','w',encoding='utf-8').write('\n'.join(lines))
