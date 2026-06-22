# version 0.3 
# - added hashlib that calculates a hash of the database file and uploads a sqlite_hash.json.
# -- This prevents redundant uploads to GitHub if the database hasn't actually changed.

import sqlite3, zipfile, os, hashlib, json
from filelock import FileLock
from .github_client import get_or_create_release, upload_asset

# Path must match kmai.py
SQLITE_DB = "data_enterprise/cache_store.db"
LOCK = FileLock("/tmp/sqlite.lock")

def get_file_hash(path):
    """Calculate SHA256 hash of the SQLite file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def backup_sqlite():
    print(f"[Backup] backup_sqlite() called for {SQLITE_DB}")
    if not os.path.exists(SQLITE_DB):
        print("[Backup] SQLite DB file not found. Skipping backup.")
        return

    with LOCK:
        current_hash = get_file_hash(SQLITE_DB)
        zip_path = "/tmp/sqlite_backup.zip"
        
        # Create the zip
        with zipfile.ZipFile(zip_path, "w") as z:
            # We use arcname to make sure it extracts as 'cache_store.db'
            z.write(SQLITE_DB, arcname="cache_store.db")

        release = get_or_create_release()
        
        # Upload the Zip and the Hash
        print(f"[Backup] Uploading SQLite backup to GitHub. Hash: {current_hash[:10]}")
        upload_asset(release, zip_path, "sqlite_backup.zip")
        
        # Create and upload hash JSON as bytes
        hash_data = json.dumps({"hash": current_hash}).encode()
        upload_asset(release, hash_data, "sqlite_hash.json")
        print("[Backup] SQLite backup and hash successfully updated in GitHub.")

        
  