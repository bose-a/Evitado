#/bin/bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
echo $TF_INC
echo $TF_LIB

# For TF 1.X (Google Colab only)
# TL="usr/local/lib/python3.7/dist-packages/tensorflow/libtensorflow_framework.so.2"
# PYPATH="/usr/local/lib/python3.7/dist-packages/tensorflow"

# For TF 2.X (Google Colab only)
TL="tensorflow-1.15.2/python3.7/tensorflow_core/libtensorflow_framework.so.1"
PYPATH="tensorflow-1.15.2/python3.7/tensorflow_core"
ROOT_TL="tensorflow-1.15.2/python3.7/tensorflow_core/include/tensorflow"

# NVCC Compile
cd pointnet2/tf_ops/grouping/
nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# Go to the root directory first (Important!)
cd ~/../
SCRIPT_DIR="content/drive/MyDrive/pointnet2/tf_ops/grouping"

# TF1.2
# g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /home/adi/anaconda3/envs/tf-gpu/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-10.1/include -lcudart -L /usr/local/cuda-10.1/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
g++ -std=c++11 ${SCRIPT_DIR}/tf_grouping.cpp ${SCRIPT_DIR}/tf_grouping_g.cu.o -o ${SCRIPT_DIR}/tf_grouping_so.so -shared -fPIC -I ${ROOT_TL} -I ${PYPATH}/include -I /usr/local/cuda-10.1/include -I ${PYPATH}/include/external/nsync/public -lcudart -L /usr/local/cuda-10.1/lib64/ -L${TL} -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -I$TF_INC/external/nsync/public -L ${TL} -ltensorflow_framework # -v
