"""Supervisor: keep SigLIP2 embedding running despite machine failures.

Runs embed_siglip2_full.py in a loop. If it exits (crash, kill, sleep-wake),
the supervisor relaunches it immediately — resume logic continues from the
last checkpoint, so no work is lost.

Each crash loses at most `save_every` frames (1000) of work.

Usage:
    python scripts/embed_supervisor.py [--batch-size 64]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

WORKDIR = Path(__file__).parent.parent
EMBED_SCRIPT = WORKDIR / "scripts" / "embed_siglip2_full.py"
LOG = WORKDIR / "data" / "processed" / "siglip2" / "supervisor.log"


def main():
    parser = argparse.ArgumentParser(description="SigLIP2 embedding supervisor")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-restarts", type=int, default=9999, help="Give up after N restarts")
    args = parser.parse_args()

    restart = 0
    while restart < args.max_restarts:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{ts}] Launch #{restart + 1} (batch={args.batch_size})"
        print(msg, flush=True)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        cmd = [
            sys.executable, str(EMBED_SCRIPT),
            "--limit", "0",
            "--batch-size", str(args.batch_size),
            "--resume",
        ]
        try:
            proc = subprocess.run(cmd, cwd=str(WORKDIR))
            code = proc.returncode
        except Exception as e:
            code = -1
            print(f"[{ts}] EXCEPTION: {e}", flush=True)

        ts2 = time.strftime("%Y-%m-%d %H:%M:%S")
        if code == 0:
            print(f"[{ts2}] DONE (exit 0). Embedding complete.", flush=True)
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(f"[{ts2}] COMPLETE\n")
            break
        else:
            print(f"[{ts2}] Job exited code={code}. Restarting in 5s...", flush=True)
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(f"[{ts2}] RESTART after exit={code}\n")
            restart += 1
            time.sleep(5)


if __name__ == "__main__":
    main()
