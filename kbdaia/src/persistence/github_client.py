# version 0.3
# - Fixed function delete_asset_if_exists()

import os
import requests

# These are fetched from HuggingFace Secrets/Environment Variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO = os.getenv("GITHUB_REPO")
RELEASE_TAG = os.getenv("GITHUB_RELEASE_TAG", "kmai-dbbkups")

API = "https://api.github.com"
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

def get_or_create_release():
    """Fetches the release object. Creates it if it doesn't exist."""
    url = f"{API}/repos/{REPO}/releases/tags/{RELEASE_TAG}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code == 200:
        return r.json()

    # If release doesn't exist, create it
    create_url = f"{API}/repos/{REPO}/releases"
    payload = {
        "tag_name": RELEASE_TAG,
        "name": "KMAI Persistent Backup",
        "draft": False,
        "prerelease": False
    }
    r = requests.post(create_url, headers=HEADERS, json=payload)
    r.raise_for_status()
    return r.json()

def delete_asset_if_exists(release, asset_name):
    """Safely finds and deletes an existing asset by name."""
    # Ensure release has an assets list
    assets = release.get("assets", [])
    for asset in assets:
        if asset["name"] == asset_name:
            asset_id = asset["id"]
            url = f"{API}/repos/{REPO}/releases/assets/{asset_id}"
            
            # Perform the delete
            resp = requests.delete(url, headers=HEADERS)
            if resp.status_code == 204:
                print(f"[GitHub] Successfully deleted existing asset: {asset_name}")
            else:
                print(f"[GitHub] Failed to delete {asset_name}: {resp.status_code}")
            return # Exit after finding and deleting

def upload_asset(release, data_or_path, asset_name):
    """
    Uploads a file or raw bytes to the GitHub release.
    data_or_path: can be a string (file path) or bytes (for the hash json)
    """
    # 1. Clean up old version first
    delete_asset_if_exists(release, asset_name)

    # 2. Prepare the Upload URL (Crucial fix: fetch it from release first)
    # The URL looks like: https://uploads.github.com/.../assets{?name,label}
    if "upload_url" not in release:
        print("[GitHub] Error: No upload_url found in release object.")
        return
        
    upload_url = release["upload_url"].split("{")[0]

    # 3. Determine content
    if isinstance(data_or_path, str):
        if not os.path.exists(data_or_path):
            print(f"[GitHub] File not found: {data_or_path}")
            return
        with open(data_or_path, "rb") as f:
            content = f.read()
    else:
        # Assume it is bytes (like our hash.json)
        content = data_or_path

    # 4. Upload
    print(f"[GitHub] Uploading {asset_name} ({len(content)} bytes)...")
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Content-Type": "application/octet-stream"
    }
    
    r = requests.post(
        upload_url,
        headers=headers,
        params={"name": asset_name},
        data=content
    )
    
    if r.status_code == 201:
        print(f"[GitHub] Successfully uploaded: {asset_name}")
    else:
        print(f"[GitHub] Upload failed for {asset_name}: {r.status_code} - {r.text}")

def download_asset(release, asset_name, out_path):
    """Downloads an asset from GitHub release to a local path."""
    for asset in release.get("assets", []):
        if asset["name"] == asset_name:
            r = requests.get(asset["url"], headers={
                **HEADERS,
                "Accept": "application/octet-stream"
            })
            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(r.content)
                return True
    return False