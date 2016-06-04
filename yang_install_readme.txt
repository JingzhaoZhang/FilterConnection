require install cuda toolkit, and cudnn tool kit (at the location specified below)


export LD_LIBRARY_PATH="/usr/local/cuda-7.5/lib64/":"/home/yang/local/my_software/cuda/lib64":$LD_LIBRARY_PATH


cd matconvnet/matlab 

vl_compilenn('enableGpu', true, ...
               
'cudaRoot', '/usr/local/cuda-7.5/', ...
               
'cudaMethod', 'nvcc', ...
               
'enableCudnn', 'true', ...
               
'cudnnRoot', '/home/yang/local/my_software/cudnn_v4', ...
               
'ImageLibraryCompileFlags', {'-I/opt/libjpeg-turbo/include'}, ...
               
'ImageLibraryLinkFlags', {'-L/opt/libjpeg-turbo/lib64'}, ...
               
'Verbose', 0)



before each use, call:

run <MatConvNet>/matlab/vl_setupnn

then 
test:
vl_testnn
vl_testnn('gpu', true)


### make using Makefile

# basic usage
make ARCH=glnxa64 MATLABROOT=/usr/local/MATLAB/R2016a



# full usage

make -j ARCH=glnxa64 MATLABROOT=/usr/local/MATLAB/R2016a \
     
ENABLE_GPU=yes CUDAROOT=/usr/local/cuda-7.5/ CUDAMETHOD=nvcc\
     
ENABLE_CUDNN=yes CUDNNROOT=/home/yang/local/my_software/cudnn_v4 \
     
ENABLE_IMREADJPEG=yes \
     
IMAGELIB_CFLAGS=-I/opt/libjpeg-turbo/include \
     
IMAGELIB_LDFLAGS='-L/opt/libjpeg-turbo/lib64 -ljpeg' \
     
VERBOSE=0

changes made:
Makefile: search Yang
add two files: vl_imreadjpeg_gpu.cpp/cu
add a read function: cnn_imagenet_get_batch.m.gpu
