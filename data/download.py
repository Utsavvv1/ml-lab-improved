import os
import requests
import gzip
import shutil

def download_file(url, dest_path):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print(f"Saved to {dest_path}")

def extract_gzip(file_path, dest_path):
    print(f"Extracting {file_path}...")
    with gzip.open(file_path, 'rb') as f_in:
        with open(dest_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Extracted to {dest_path}")

def prepare_multi30k():
    data_dir = "data/multi30k"
    os.makedirs(data_dir, exist_ok=True)
    
    # URLs for Multi30k (using a reliable mirror or github raw if available, 
    # but often these are hosted on servers that go down. 
    # Using raw raw raw text from a github repo is often easiest for small datasets.)
    # Here using the bentrevett/pytorch-seq2seq mirror which is popular for this tutorial loop.
    
    urls = {
        "train.en": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz",
        "val.en": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz",
        "test.en": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt_task1_test2016.tar.gz"
    }
    
    # Actually, the easiest plaintext version:
    base_url = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw"
    files = [
        "train.en", "train.de",
        "val.en", "val.de",
        "test_2016_flickr.en", "test_2016_flickr.de"
    ]
    
    for filename in files:
        url = f"{base_url}/{filename}.gz"
        local_gz = os.path.join(data_dir, filename + ".gz")
        local_raw = os.path.join(data_dir, filename)
        
        if not os.path.exists(local_raw):
            try:
                download_file(url, local_gz)
                extract_gzip(local_gz, local_raw)
                os.remove(local_gz)
            except Exception as e:
                print(f"Failed to download {url}: {e}")

    print("Multi30k dataset ready in data/multi30k/")

if __name__ == "__main__":
    prepare_multi30k()
