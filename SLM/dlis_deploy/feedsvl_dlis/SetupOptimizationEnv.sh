#!/bin/bash

python_path=$(which python)
suffix=/bin/python
conda_env=${python_path:0:${#python_path}-${#suffix}}

echo "conda env path: $conda_env"

python_name=$(ls $conda_env/lib | grep python | grep -v lib)
packages_dir=$conda_env/lib/$python_name/site-packages
cuda_folder=$(ls $packages_dir/dlis_model_opt | grep cuda)
cudnn_folder=$(ls $packages_dir/dlis_model_opt | grep cudnn)
tensorrt_folder=$(ls $packages_dir/dlis_model_opt | grep tensorrt)

echo "packages dir: $packages_dir"

cuda_libs=$packages_dir/dlis_model_opt/$cuda_folder/lib64
cudnn_libs=$packages_dir/dlis_model_opt/$cudnn_folder/lib64
tensorrt_libs=$packages_dir/dlis_model_opt/$tensorrt_folder/lib64
onnxruntime_libs=$packages_dir/onnxruntime/capi
deepgpu_dir=$packages_dir/dlis_model_opt/deepgpuv2

export PATH=$deepgpu_dir:$PATH
export LD_LIBRARY_PATH=$cuda_libs:$cudnn_libs:$tensorrt_libs:$onnxruntime_libs:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$deepgpu_dir:$deepgpu_dir/deepgpu_plugins:$LD_LIBRARY_PATH

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"