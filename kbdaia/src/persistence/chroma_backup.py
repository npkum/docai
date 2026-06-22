# Version 0.2 -  MATCHING kmai.py: DATA_DIR="data_enterprise"

import os, zipfile, hashlib, json
from filelock import FileLock
from .github_client import get_or_create_release, upload_asset

# MATCHING kmai.py: DATA_DIR="data_enterprise", CHROMA_DIR="data_enterprise/chroma_db"
CHROMA_DIR = "data_enterprise/chroma_db"
LOCK = FileLock("/tmp/chroma.lock")

def dir_hash(path):
    h = hashlib.sha256()
    for root, _, files in os.walk(path):
        for f in sorted(files):
            with open(os.path.join(root, f), "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()

def backup_chroma():
    print(f"[Backup] Checking for Chroma DB at {CHROMA_DIR}...")
    if not os.path.exists(CHROMA_DIR):
        print("[Backup] Chroma directory not found. Skipping.")
        return

    with LOCK:
        h = dir_hash(CHROMA_DIR)
        zip_path = "/tmp/chroma_backup.zip"

        with zipfile.ZipFile(zip_path, "w") as z:
            for root, _, files in os.walk(CHROMA_DIR):
                for f in files:
                    full = os.path.join(root, f)
                    # Maintain structure relative to CHROMA_DIR
                    z.write(full, arcname=os.path.relpath(full, start=os.path.dirname(CHROMA_DIR)))

        release = get_or_create_release()
        upload_asset(release, zip_path, "chroma_backup.zip")
        upload_asset(release, json.dumps({"hash": h}).encode(), "chroma_hash.json")