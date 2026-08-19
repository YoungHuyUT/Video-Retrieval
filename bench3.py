import time, numpy as np
from aic2026.ingestion import load_manifest
from aic2026.retrieval import RetrievalPipeline
from aic2026.retrieval.index import VectorIndex
from aic2026.reranking.lexical import build_object_adjustment_matrices

t0=time.time()
manifest = load_manifest('data/processed/official_manifest.jsonl')
vectors = np.load('data/processed/official_features.npy')
idx = VectorIndex(vectors)
pipe = RetrievalPipeline(idx, manifest, frames_per_video=3)
print('load: %.1fs  videos=%d  frames=%d' % (time.time()-t0, len(pipe._video_embeddings), vectors.shape[0]))

rng = np.random.default_rng(0)
events = ['a person enters the room','sits down','starts speaking']
ev = rng.standard_normal((3, vectors.shape[1])).astype(np.float32)

# Replicate tools.retrieve_trake EXACTLY (without CLIP load — encoder mocked)
all_videos = set(pipe._video_embeddings.keys())
t1=time.time()
mats = build_object_adjustment_matrices(events=events, candidate_videos=all_videos,
    video_to_manifest_indices=pipe._video_to_manifest_indices, manifest=manifest, weight=0.08)
print('build_object_adjustment_matrices: %.2fs (%d videos)' % (time.time()-t1, len(mats)))

t1=time.time()
cands = pipe.retrieve_trake(event_embeddings=ev, top_videos=100, prefilter_frames_per_event=500,
    coarse_top_k=200, object_adjustment_matrices=mats, preferred_prefixes=['L26'])
print('retrieve_trake: %.2fs -> %d videos' % (time.time()-t1, len(cands)))

# SECOND query (matrices rebuilt, but label/concept caches warm)
t1=time.time()
mats2 = build_object_adjustment_matrices(events=events, candidate_videos=all_videos,
    video_to_manifest_indices=pipe._video_to_manifest_indices, manifest=manifest, weight=0.08)
print('2nd build_object_adjustment_matrices (caches warm): %.2fs' % (time.time()-t1))

c = cands[0]
print('top: %s  frames=%s  score=%.4f' % (c.video_id, c.event_frames, c.score))
assert len(c.event_frames)==3
print('FORMAT OK (3 frames, not KIS single)')
print('TOTAL per-query (excl one-time CLIP/Florence load): ~%.1fs' % (time.time()-t0 - t0 + (mats.__sizeof__()*0)/1))
