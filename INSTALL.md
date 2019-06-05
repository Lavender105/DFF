## Requirements:

Pytorch 1.0

cuda 9.0

cudnn v7.3

nccl v2.3.7

## Step-by-step installation
### 1. Install Pytorch from source
(1) download pytorch and checkout to version ed02619
```
mkdir pytorch-master
cd pytorch-master
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
git checkout ed02619
git submodule init
git submodule update --recursive
```

(2) create a clean conda env
```
conda create -n dff-master python=3.6
conda activate dff-master
```

(3) install pytorch
 In this part, we assume you are in the directory `pytorch-master/pytorch`.
```
# for conda user
export PATH="anaconda root directory/envs/dff-master/bin:/usr/local/bin:/usr/bin:$PATH"

# install path
export CMAKE_PREFIX_PATH="anaconda root directory /envs/dff-master/"

# Install basic dependencies
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -c mingfeima mkldnn

export CUDA_HOME="cuda root directory"

export CUDNN_INCLUDE_DIR="cudnn root directory/include"
export CUDNN_LIB_DIR="cudnn root directory/lib64"

export NCCL_ROOT_DIR="nccl root directory"
export NCCL_LIB_DIR="nccl root directory/lib"
export NCCL_INCLUDE_DIR="nccl root directory/include"

# Current setting does not support NNPACK
export TORCH_CUDA_ARCH_LIST="3.5;5.0;5.2;6.0;6.1;7.0+PTX"

make clean
rm ./torch/lib/build -fr
python setup.py install
```

### 2. Install pytorch-encoding
(1) Install dependencies
```
conda activate dff-master
conda install -c conda-forge ninja
pip install requests
pip install tqdm
pip install scikit-image
```

(2) Clone the DFF repository and install pytorch-encoding
```
git clone --recursive https://github.com/Lavender105/DFF.git
cd pytorch-encoding
python setup.py install
```
