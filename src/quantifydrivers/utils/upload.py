

import requests
import os
import sys
import urllib.request
import json

# --- CONFIGURATION ---
TOKEN = "zT4pBwBGQ03Fhg0veTEoLqc126S72AKGbXMdKRiBMTfdAC1fUg7Pn0BCYS5K"
DEPOSITION_ID = "18402053"
FILE_PATH = "climate_data_new.tar.gz"


def upload_file():
    headers = {"Authorization": f"Bearer {TOKEN}"}

    # 1. Get the Bucket URL
    print("Fetching deposition details...")
    req = urllib.request.Request(
        f"https://zenodo.org/api/deposit/depositions/{DEPOSITION_ID}",
        headers=headers
    )

    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            bucket_url = data["links"]["bucket"]
    except urllib.error.HTTPError as e:
        print(f"Error getting deposition: {e}")
        sys.exit(1)

    # 2. Upload the file (Streaming to save RAM)
    file_name = os.path.basename(FILE_PATH)
    upload_url = f"{bucket_url}/{file_name}"
    print(f"Uploading {file_name} to {upload_url}...")

    # We open the file and pass the file object directly to urllib
    # This reads it in chunks instead of loading 10GB into RAM
    with open(FILE_PATH, "rb") as f:
        req = urllib.request.Request(upload_url, data=f, headers=headers, method="PUT")

        try:
            with urllib.request.urlopen(req) as response:
                print("✅ Upload Successful!")
        except urllib.error.HTTPError as e:
            print(f"❌ Upload Failed: {e}")
            print(e.read().decode())


if __name__ == "__main__":
    upload_file()