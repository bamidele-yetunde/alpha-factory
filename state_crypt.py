"""Encrypt / decrypt the tool's private files so they can live in a public repository.

Everything that reveals your alphas (expressions, results, passes, logs) is packed into one file,
state.enc, encrypted with the key in the STATE_KEY environment variable (a GitHub secret).
Without the key, state.enc is unreadable.

Usage:
  python state_crypt.py newkey          # print a new key (store it as the STATE_KEY secret)
  python state_crypt.py pack            # private files -> state.enc
  python state_crypt.py unpack          # state.enc -> private files
  python state_crypt.py pull            # download the latest state from GitHub and unpack it

The key is read from STATE_KEY, or from a local .state_key file (never committed).
"""
import io
import os
import sys
import tarfile
from pathlib import Path

from cryptography.fernet import Fernet

HERE = Path(__file__).parent
STATE = HERE / "state.enc"
PRIVATE_FILES = ["results.csv", "checks.csv", "submissions.csv", "submitted_alphas.csv", "fields.csv",
                 "READY_TO_SUBMIT.csv", "insights.md", "activity_log.txt", "sweep_state.json",
                 "run_output.txt", "notified.json", "pnl_cache.json.gz"]


def key():
    k = os.environ.get("STATE_KEY")
    if not k and (HERE / ".state_key").exists():
        k = (HERE / ".state_key").read_text(encoding="utf-8").strip()
    if not k:
        sys.exit("STATE_KEY is not set (and no .state_key file).")
    return Fernet(k.encode())


def pull():
    """Download the latest encrypted state from GitHub and unpack it here (to view results locally)."""
    import subprocess
    subprocess.run(["git", "fetch", "-q", "origin", "state"], cwd=HERE, check=True)
    data = subprocess.run(["git", "show", "origin/state:state.enc"], cwd=HERE, check=True,
                          capture_output=True).stdout
    STATE.write_bytes(data)
    unpack()


def pack():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in PRIVATE_FILES:
            if (HERE / name).exists():
                tar.add(HERE / name, arcname=name)
    STATE.write_bytes(key().encrypt(buf.getvalue()))
    print(f"packed {STATE.name} ({STATE.stat().st_size // 1024} KB)")


def unpack():
    if not STATE.exists():
        print("no state.enc yet - starting fresh")
        return
    data = key().decrypt(STATE.read_bytes())
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(HERE, filter="data")
    print("unpacked state")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "newkey":
        print(Fernet.generate_key().decode())
    elif command == "pack":
        pack()
    elif command == "unpack":
        unpack()
    elif command == "pull":
        pull()
    else:
        sys.exit(__doc__)
