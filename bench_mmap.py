import time, gc, numpy as np, psutil, os
proc = psutil.Process(os.getpid())

def rss():
    gc.collect()
    return proc.memory_info().rss / 1e6  # MB

t0 = time.time()
from aic2026.ingestion import load_manifest
from aic2026.retrieval import RetrievalPipeline
from aic2026.retrieval.index import VectorIndex
from aic2026.reranking.lexical import build_object_adjustment_matrices

manifest = load_manifest('data/processed/official_manifest.jsonl')
rss_after_manifest = rss()

# MMAP load (new default)
idx = VectorIndex.from_npy('data/processed/official_features.npy', mmap=True)
rss_after_index = rss()
pipe = RetrievalPipeline(idx, manifest, frames_per_video=3)
rss_after_pipeline = rss()

print('load manifest+mmp: %.1fs  RSS after index=%.0fMB  after pipeline=%.0fMB' % (time.time()-t0, rss_after_index, rss_after_pipeline))

rng = np.random.default_rng(0)
events = ['a person enters the room','sits down','starts speaking']
ev = rng.standard_normal((3, idx.vectors.shape[1])).astype(np.float32)

all_videos = set(pipe._video_embeddings.keys())
t1 = time.time()
mats = build_object_adjustment_matrices(events=events, candidate_videos=all_videos,
    video_to_manifest_indices=pipe._video_to_manifest_indices, manifest=manifest, weight=0.08)
rss_after_mat = rss()
print('build_object_adjustment_matrices: %.2fs  RSS=%.0fMB' % (time.time()-t1, rss_after_mat))

t1 = time.time()
cands = pipe.retrieve_trake(event_embeddings=ev, top_videos=100, prefilter_frames_per_event=500,
    coarse_top_k=200, object_adjustment_matrices=mats, preferred_prefixes=['L26'])
rss_after_trake = rss()
print('retrieve_trake: %.2fs  RSS=%.0fMB' % (time.time()-t1, rss_after_trake))

c = cands[0]
print('top: %s  frames=%s  score=%.4f' % (c.video_id, c.event_frames, c.score))
assert len(c.event_frames) == 3, 'FORMAT BROKEN'
print('FORMAT OK (3 frames)')
print('TOTAL per-query peak RSS ~= %.0f MB' % rss_after_trake)
