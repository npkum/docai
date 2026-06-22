# version 0.3
# - fixed chroma DB restore to match DATA_DIR in main application

import os, zipfile, sqlite3
from .github_client import get_or_create_release, download_asset

# Use the same DATA_DIR defined in kmai.py
DATA_DIR = "data_enterprise"

def restore_chroma_once():
    # HF Space restart check: only run once per container life
    if os.path.exists("/tmp/.chroma_restored"):
        return

    print("[Restore] Starting Chroma DB restore from GitHub...")
    # Ensure the data_enterprise directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    release = get_or_create_release()
    zip_temp = "/tmp/chroma_restore.zip"
    
    if download_asset(release, "chroma_backup.zip", zip_temp):
        with zipfile.ZipFile(zip_temp, 'r') as z:
            # zip contains 'chroma_db/...'
            # Extracting into 'data_enterprise' results in 'data_enterprise/chroma_db/...'
            z.extractall(DATA_DIR)
        print(f"[Restore] Chroma DB successfully extracted into {DATA_DIR}")
    else:
        print("[Restore] No Chroma backup found in GitHub Release.")

    # Mark as restored so it doesn't run again until next HF restart
    open("/tmp/.chroma_restored", "w").close()

def restore_sqlite_once():
    # HF Space restart check
    if os.path.exists("/tmp/.sqlite_restored"):
        return

    print("[Restore] Starting SQLite Cache restore from GitHub...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    release = get_or_create_release()
    zip_temp = "/tmp/sqlite_restore.zip"
    
    if download_asset(release, "sqlite_backup.zip", zip_temp):
        with zipfile.ZipFile(zip_temp, 'r') as z:
            # zip contains 'cache_store.db'
            # Extracting into 'data_enterprise' results in 'data_enterprise/cache_store.db'
            z.extractall(DATA_DIR)
        
        db_path = os.path.join(DATA_DIR, "cache_store.db")
        if os.path.exists(db_path):
            try:
                # Optional: Clean up sequence to prevent ID conflicts
                conn = sqlite3.connect(db_path)
                conn.execute("DELETE FROM sqlite_sequence WHERE name='qa_cache';")
                conn.commit()
                conn.close()
                print(f"[Restore] SQLite DB restored and sequence cleaned.")
            except Exception as e:
                print(f"[Restore] SQLite sequence cleanup skipped: {e}")
    else:
        print("[Restore] No SQLite backup found in GitHub Release.")

    # Mark as restored
    open("/tmp/.sqlite_restored", "w").close()