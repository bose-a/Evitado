

### The following document outlines cutting edge research for 3D point cloud classification.

---

### Setup for Linux Ubuntu 20.0.4

- **To use the PointNet framework, we need `tensorflow-gpu`  for reasonable training times**

- **Install the latest *tested* driver for your graphics card:** 

    ```
    Software & Updates > Additional Drivers > Select NVIDIA driver > Apply
    ```

  - Make sure you reboot your computer, then run `nvidia-smi` to ensure the driver is working correctly

  - This driver version may be overridden by later steps—but that's OK, we just need to be able to use the `nvidia-smi` command for now.

    - If unable to install the latest driver through the default app, run `sudo apt get <driver>` and reboot.

- **If using an NVIDIA GPU that has a compute capability >= 3.5, install the latest stable versions of CUDA, cuDNN, and TensorFlow normally. We'll use virtual environments with `Anaconda`** to make versioning easier.

  - Execute the following commands to install Cuda from the Ubuntu Repository:

    ```bash
    sudo apt get update
    sudo apt install nvidia-cuda-toolkit
    ```

  - After following the provided instructions, check that your system detects the Cuda install by running `nvidia-smi`; also check that you have a working Cuda Toolkit install by running `nvcc -V`

  - If  `nvcc` cannot be detected, you might have to run the following commands to properly set your environment variables:

    ```bash
    export PATH=/usr/local/cuda-<version>/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-<version>/lib64:$LD_LIBRARY_PATH`
    source ~/.bashrc # Reloads .bashrc settings
    ```

  - Then, install the latest version of TensorFlow with `pip install tensorflow`. 

  - For more detailed requirements and instructions, refer to https://www.tensorflow.org/install/gpu

- **Otherwise, if using an unsupported CUDA architecture (i.e compute 3.0), you can either build TensorFlow from source, or install an older version of TensorFlow & CUDA.**

  - Install Python & TensorFlow package dependencies

    ```bash
    sudo apt install python3-dev python3-pip
    ```

  - Install TensorFlow pip package dependencies. Since we're using a virtual environment, omit the `--user` argument. 

    ```bash
    pip install -U pip numpy wheel
    pip install -U keras_preprocessing --no-deps
    ```

  - Install Bazel. We'll install it manually, but you can also use `Bazelisk` to install it.

    ```bash
    sudo apt install curl gnupg
    curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
    sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
    echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
    
    sudo apt update && sudo apt install bazel
    sudo apt update && sudo apt full-upgrade
    bazel --version # Check if Bazel was correctly installed
    ```

  - 



