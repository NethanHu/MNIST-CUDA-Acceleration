# CUDA MNIST Neural Network

学习目标：基于 CUDA 实现的 MNIST 手写数字识别神经网络项目，包含四种不同的实现方式：PyTorch 经典实现方式、CPU 简单实现、CUDA 简单实现和 cuBLAS 矩阵乘法优化实现。

> 神经网络为 MLP 结构，只有一个隐藏层以及 10 个输出类别，最后使用交叉熵 Loss。

## 项目结构

```
.
├── README.md                  # 项目说明文档
├── pyproject.toml             # Python项目配置
├── data/                      # 数据集目录
│   ├── processed/             # 预处理后的二进制数据集
│   │   ├── test_images.bin
│   │   ├── test_labels.bin
│   │   ├── train_images.bin
│   └── └── train_labels.bin
│   └── raw/                   # 原始MNIST数据（省略内部的.gz文件）
├── scripts/                   # 工具脚本
│   ├── torch_impl.py          # PyTorch版本实现（最简单）
│   └── download_mnist.py      # MNIST数据下载脚本
└── src/                       # 源代码
    ├── nn_struct.h            # 神经网络结构定义
    ├── cpu_naive_impl.c       # 基于 CPU 简单实现
    ├── cuda_naive_impl.cu     # 基于 CUDA 简单实现
    └── cuda_cublas_impl.cu    # 基于 cuBLAS 的优化实现
```

## 快速开始

### 1. 环境准备

确保你的系统已安装：
- **CUDA Toolkit** (>=11.0)
- **Python** (>=3.8)
- **UV 包管理器**
- **GCC/G++ 编译器**

安装UV包管理器（如果尚未安装）：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 项目依赖安装

使用UV同步项目依赖：
```bash
uv sync
source .venv/bin/activate
```
> Note：如果想要运行PyTorch的训练流程，请手动安装`torch`, `torchvision`等依赖项。这里为了精简期间不选择下载。

### 3. 下载和预处理数据集

运行数据下载脚本：
```bash
mkdir data
mkdir data/raw
mkdir data/processed
python scripts/download_mnist.py
```

这个脚本会：
- 自动下载MNIST数据集到`data/raw/`
- 解压并转换为二进制格式存储到`data/processed/`
- 数据归一化到[0,1]范围

### 4. 编译项目

进入到src文件夹下对三个源码进行分别编译：
```bash
cd src
gcc -o naive_cpu cpu_naive_impl.c -lm
nvcc -o naive_cuda cuda_naive_impl.cu
nvcc -o cublas_cuda cuda_cublas_impl.cu -lcublas
```
> Note：可以选择在源码中修改超参数。

### 5. 运行训练

编译完成后，我们可以运行文件进行训练，查看训练时间和训练精度。

```bash
# 【可选】使用PyTorch进行训练
python ../scripts/torch_impl.py
./naive_cpu
./naive_cuda
./cublas_cuda
```
