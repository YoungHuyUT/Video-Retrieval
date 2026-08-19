import json, os, tempfile, builtins
import scripts.colab_ocr_l25 as ocr

tmp = tempfile.mkdtemp()
manifest_in = os.path.join(tmp, "in.jsonl")
manifest_out = os.path.join(tmp, "out.jsonl")

def make_recs(n, done_idx=()):
    recs = []
    for i in range(n):
        r = {
            "vector_id": i, "video_id": "L25_V001", "frame_id": i,
            "keyframe_path": f"/fake/L25_V001/{i:03d}.jpg",
            "object_labels": ["person"],  # Faster-RCNN label, must be preserved
        }
        if i in done_idx:
            r["ocr_done"] = True
            r["object_labels"] = ["person", "ocrtextOLD"]
        recs.append(r)
    return recs

def write_recs(path, recs):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Fake PaddleOCR
class FakePage:
    def __init__(self, t): self._t = t
    def json(self): return {"res": {"rec_texts": [self._t]}}
class FakeExtractor:
    def __init__(self): self.calls = 0
    def predict(self, kp):
        self.calls += 1
        i = int(kp.split("/")[-1].split(".")[0])
        return [FakePage(f"ocrtext{i}")]

fake = FakeExtractor()
def fake_extractor(*a, **k):
    # Reuse the shared instance so we can count predict() calls across main().
    return fake

real_import = builtins.__import__
def fake_import(name, *a, **k):
    if name == "paddleocr":
        class M:
            PaddleOCR = fake_extractor
        return M
    return real_import(name, *a, **k)
builtins.__import__ = fake_import
# Force the fake paddleocr module into sys.modules so any cached import inside
# main() returns our fake instead of a real (or missing) paddleocr.
import sys
sys.modules["paddleocr"] = type(sys)("paddleocr")
sys.modules["paddleocr"].PaddleOCR = fake_extractor

ocr.KEYFRAMES_DIR = "/fake"
ocr.MANIFEST_IN = manifest_in
ocr.MANIFEST_OUT = manifest_out
ocr.VIDEO_PREFIX = "L25"
ocr.USE_GPU = False

# Make keyframe files exist on disk at the exact paths used in records so
# resolve_kp() (a nested function inside main) returns a real existing path
# instead of treating them as "no image" and skipping OCR.
for i in range(6):
    p = os.path.join(tmp, "L25_V001", f"{i:03d}.jpg")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(b"\xff\xd8\xff\xe0dummy")
ocr.KEYFRAMES_DIR = tmp  # rglob will find the L25_V001/*.jpg above

# ---- Test 1: SKIP already-ocr_done frames (resume semantics) ----
fake.calls = 0
write_recs(manifest_in, make_recs(4, done_idx={0, 1}))  # frames 0,1 already done
# ensure no stale MANIFEST_OUT
if os.path.exists(manifest_out): os.remove(manifest_out)
ocr.BATCH_SIZE = 4
ocr.main()
assert fake.calls == 2, f"expected 2 OCR calls (skip 2 done), got {fake.calls}"
final = [json.loads(l) for l in open(manifest_out, encoding="utf-8")]
assert all("person" in r["object_labels"] for r in final), "Faster label lost!"
assert any("ocrtext2" in r["object_labels"] for r in final), "new OCR not written"
assert any("ocrtextOLD" in r["object_labels"] for r in final), "prior OCR label lost on skip"
print("TEST1 skip-done + preserve Faster: PASS")

# ---- Test 2: CHECKPOINT writes every BATCH_SIZE frames ----
# Patch write_manifest to count calls
orig_write = ocr.write_manifest
counts = {"n": 0}
def counting_write(records, path):
    counts["n"] += 1
    orig_write(records, path)
ocr.write_manifest = counting_write
fake.calls = 0
write_recs(manifest_in, make_recs(6))  # fresh, none done
if os.path.exists(manifest_out): os.remove(manifest_out)
ocr.BATCH_SIZE = 2  # expect checkpoints at 2,4,6 + final = 4 writes
ocr.main()
# main writes checkpoint at done==2,4,6 and a final write at [5/6] => 4 calls
assert counts["n"] == 4, f"expected 4 writes (3 checkpoint + 1 final), got {counts['n']}"
ckpt = [json.loads(l) for l in open(manifest_out, encoding="utf-8")]
assert len([r for r in ckpt if r.get("ocr_done")]) == 6, "not all done"
assert all("person" in r["object_labels"] for r in ckpt)
print("TEST2 checkpoint cadence (every BATCH_SIZE) + final: PASS")

builtins.__import__ = real_import
print("ALL OCR CHECKPOINT/RESUME TESTS PASSED")
