import time, gc, ctypes, numpy as np, psutil, os
from ctypes import wintypes

# Windows private working set (không tính file page cache)
class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]

def private_mb():
    gc.collect()
    p = PROCESS_MEMORY_COUNTERS_EX()
    p.cb = ctypes.sizeof(p)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(),
        ctypes.byref(p), ctypes.sizeof(p))
    return p.PrivateUsage / 1e6

proc = psutil.Process(os.getpid())
def rss():
    gc.collect(); return proc.memory_info().rss / 1e6

from aic2026.ingestion import load_manifest
from aic2026.retrieval import RetrievalPipeline
from aic2026.retrieval.index import VectorIndex
from aic2026.reranking.lexical import build_object_adjustment_matrices

manifest = load_manifest('data/processed/official_manifest.jsonl')
idx = VectorIndex.from_npy('data/processed/official_features.npy', mmap=True)
pipe = RetrievalPipeline(idx, manifest, frames_per_video=3)
print('after mmap index+pipeline: RSS=%.0fMB  PRIVATE=%.0fMB' % (rss(), private_mb()))

rng = np.random.default_rng(0)
events = ['a person enters the room','sits down','starts speaking']
ev = rng.standard_normal((3, idx.vectors.shape[1])).astype(np.float32)
all_videos = set(pipe._video_embeddings.keys())

mats = build_object_adjustment_matrices(events=events, candidate_videos=all_videos,
    video_to_manifest_indices=pipe._video_to_manifest_indices, manifest=manifest, weight=0.08)
print('after build matrices:       RSS=%.0fMB  PRIVATE=%.0fMB' % (rss(), private_mb()))

cands = pipe.retrieve_trake(event_embeddings=ev, top_videos=100, prefilter_frames_per_event=500,
    coarse_top_k=200, object_adjustment_matrices=mats, preferred_prefixes=['L26'])
print('after retrieve_trake:       RSS=%.0fMB  PRIVATE=%.0fMB' % (rss(), private_mb()))
assert len(cands[0].event_frames) == 3
print('FORMAT OK')
