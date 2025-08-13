import os
import gzip
import struct
import numpy as np
import urllib.request
from pathlib import Path

MNIST_URLS = {
    'train_images': 'https://cseweb.ucsd.edu/~weijian/static/datasets/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'https://cseweb.ucsd.edu/~weijian/static/datasets/mnist/train-labels-idx1-ubyte.gz',
    'test_images': 'https://cseweb.ucsd.edu/~weijian/static/datasets/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'https://cseweb.ucsd.edu/~weijian/static/datasets/mnist/t10k-labels-idx1-ubyte.gz'
}

def download_and_convert():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading the MNIST datasets...")
    
    for name, url in MNIST_URLS.items():
        filename = raw_dir / os.path.basename(url)
        if not filename.exists():
            print(f"  Downloading {filename.name}...")
            urllib.request.urlretrieve(url, filename)
        else:
            print(f"  {filename.name} has existed.")
    
    print("Transferring the data format...")
    
    # 读取并转换数据
    train_images = read_images(raw_dir / "train-images-idx3-ubyte.gz")
    train_labels = read_labels(raw_dir / "train-labels-idx1-ubyte.gz")
    test_images = read_images(raw_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = read_labels(raw_dir / "t10k-labels-idx1-ubyte.gz")
    
    # 保存为二进制格式
    save_binary(train_images, processed_dir / "train_images.bin")
    save_binary(train_labels, processed_dir / "train_labels.bin")
    save_binary(test_images, processed_dir / "test_images.bin")
    save_binary(test_labels, processed_dir / "test_labels.bin")
    
    print("The data is all set!")

def read_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols).astype(np.float32) / 255.0
    return images

def read_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)
    return labels

def save_binary(data, filename):
    with open(filename, 'wb') as f:
        f.write(data.tobytes())
    print(f"  Saving: {filename}")

if __name__ == "__main__":
    download_and_convert()