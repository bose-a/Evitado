# Setup for Linux Ubuntu 20.0.4

To use the PointNet framework, we first need to configure TensorFlow so that it uses our dedicated graphics card. Personally, I scavenged an old Linux system running on NVIDIA's GeForce GTX 750M. The PointNet library's backbone is a deep neural network, and training these models is an extremely performance intensive process. Using my old graphics card for this work is much faster than a quad-core, hyperthreaded CPU—simply because GPUs are meant for this kind of computation. This is why it is necessary to follow the steps below. Although tedious, it's definitely worth it!

---

## **Install the latest tested driver for your graphics card:** 

```
Software & Updates > Additional Drivers > Select NVIDIA driver > Apply
```

- Make sure you reboot your computer, then run `nvidia-smi` to ensure the driver is working correctly

- This driver version may be overridden by later steps—but that's OK, we just need to be able to use the `nvidia-smi` command for now.

- If unable to install the latest driver through the default app, run:

  ```bash
  sudo apt get <driver>
  sudo reboot
  ```

- If `nvidia-smi` still doesn't work, it may be because you have a firmware password for your Linux operating system. Run:

  ```bash
  sudo update-secureboot-policy --enroll-key
  sudo reboot # Make sure to go through the 'Enroll MOK key' process.
  ```

---

## If using an NVIDIA GPU that has a compute capability >= 3.5, install the latest stable versions of CUDA, cuDNN, and TensorFlow normally. 

For now, we won't use virtual environments. Later on, we'll see how to use the `Anaconda` package & environment manager to make versioning way easier.

Execute the following commands to install Cuda from the Ubuntu Repository:

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

- Note that different versions of CUDA may install to different locations, so you may have to change the above commands appropriately.

- Lastly, install the latest version of TensorFlow with `pip install tensorflow`. 

- For more detailed requirements and instructions, refer to https://www.tensorflow.org/install/gpu

---

## **If using an unsupported CUDA architecture (i.e compute 3.0), either build TensorFlow from source, or install an older version of TensorFlow & CUDA.**

### First, we'll go through building from source:

- Install Python & TensorFlow package dependencies

  ```bash
  sudo apt install python3-dev python3-pip
  ```

- Install TensorFlow pip package dependencies. Since we're using a virtual environment, omit the `--user` argument. 

  ```bash
  pip install -U pip numpy wheel
  pip install -U keras_preprocessing --no-deps
  ```

- Install Bazel, Google's proprietary build system (`CMake`, `make` and `ninja` are standard alternatives). We'll install it manually, but you can also use `Bazelisk` to install it.

  ```bash
  sudo apt install curl gnupg
  curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
  sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
  echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
  
  sudo apt update && sudo apt install bazel
  sudo apt update && sudo apt full-upgrade
  bazel --version # Check if Bazel was correctly installed
  ```

- Download the TensorFlow source code

  ```bash
  git clone https://github.com/tensorflow/tensorflow.git
  cd tensorflow
  ```

- Checkout to a specific release branch of TensorFlow. We need to do this because our GPU's compute capability is not compatible with the latest versions of CUDA, and the latest versions TensorFlow only support the recent versions of CUDA. Using the following links and table, find a CUDA version that supports your GPU's compute capability. and then the corresponding TensorFlow version. 

  ***TensorFlow & CUDA compatibility***
  https://www.tensorflow.org/install/source

  ***CUDA & cuDNN compatibility***
  https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html

  ***CUDA & compute capability compatibility; see table below***
  https://tech.amikelive.com/node-930/cuda-compatibility-of-nvidia-display-gpu-drivers/ 

  | CUDA Version      | Minimum Compute Capability | Default Compute Capability |
  | ----------------- | -------------------------- | -------------------------- |
  | CUDA 10.0         | 3.0                        | 3.0                        |
  | CUDA 9.2 update 1 | 3.0                        | 3.0                        |
  | CUDA 9.2          | 3.0                        | 3.0                        |
  | CUDA 9.1          | 3.0                        | 3.0                        |
  | CUDA 9.0          | 3.0                        | 3.0                        |
  | CUDA 8.0 GA2      | 2.0                        | 2.0                        |
  | CUDA 8.0          | 2.0                        | 2.0                        |
  | CUDA 7.5          | 2.0                        | 2.0                        |
  | CUDA 7.0          | 2.0                        | 2.0                        |
  | CUDA 6.5          | 1.1                        | 2.0                        |
  | CUDA 6.0          | 1.0                        | 1.0                        |
  | CUDA 5.5          | 1.0                        | 1.0                        |
  | CUDA 5.0          | 1.0                        | 1.0                        |
  | CUDA 4.2          | 1.0                        | 1.0                        |
  | CUDA 4.1          | 1.0                        | 1.0                        |
  | CUDA 4.0          | 1.0                        | 1.0                        |
  | CUDA 3.2          | 1.0                        | 1.0                        |
  | CUDA 3.1          | 1.0                        | 1.0                        |
  | CUDA 3.0          | 1.0                        | 1.0                        |

- Since I'm using a GeForce GTX 750M (Kepler family), I chose to install CUDA 9.0, cuDNN 7.6.5, and TensorFlow 1.12:

  ```bash
  git checkout r1.12
  ```

- Then, we want to configure our custom TensorFlow build. 

  ```bash
  ./configure # Use this, since we aren't using virtual environments. 
  python configure.py # If we were using a virtual environment, we'd use this line instead
  ```

- Make sure you select `Y` for CUDA support, and then type your desired CUDA compatibility. 

- After running the above command, there should be a `.tf_configure.bazelrc` file in your current directory. You may need to use `ls -a` to see it there, since it's a hidden file. 

- We need to make some tweaks here to ensure the custom build will successfully compile. First, open the file with the default `nano` text editor. This is handy for quickly editing files from the command line itself.    

  ```bash
  nano .tf_configure.bazelrc
  ```

- Add the following lines to the file and save your changes.

  ```bash
  build --define=with_xla_support=false
  build --action_env TF_ENABLE_XLA=0
  ```

- In the above command, we excluded support for XLA (stands for Accelerated Linear Algebra). This is compiler for linear algebraic computation that accelerates TensorFlow models. It often improves runtime significantly, so it's always preferred to have it if possible. Although I disabled it in order to get my build to work, the following link outlines a similar build from source method that successfully uses XLA with an even older graphics card: 

  https://stackoverflow.com/questions/39023581/tensorflow-cuda-compute-capability-3-0-the-minimum-required-cuda-capability-is 

- Since I'm building TensorFlow 1.12, I'm going to specify the `--config=v1` flag. I'm also going to constrain the amount of RAM that `baze` can use, because it tends to be very taxing. My system crashed on first build when I didn't do this! I allocated half of my 8 GB RAM.

  ```bash
  bazel build --config=v1 [--config=option] //tensorflow/tools/pip_package:build_pip_package --local_ram_resources=4096
  ```

- As a side note, we can also specify more options either through the command line or by directly editing the `.tf_configure.bazelrc` file like before. We might want to specify other CUDA options, like the following lines if the build doesn't work the first time:

  ```bash
  build:opt --copt=-DTF_EXTRA_CUDA_CAPABILITIES=3.0
  build --config=cuda # Either keep both config flags or delete existing.
  ```

- As the TensorFlow website states, "The [build] command creates an executable named `build_pip_package`—this is the program that builds the `pip` package. Run the executable as shown below to build a `.whl` package in the `/tmp/tensorflow_pkg` directory.

  ```bash
  ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg # If building from release branch
  
  ./bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag /tmp/tensorflow_pkg # If building from master
  ```

- Finally, use `pip install` to install the package. Note that the "filename of the generated `.whl` file depends on the TensorFlow version and your platform". Use `ls -a` in the appropriate directory to find the exact name if you're having trouble. 

  ```bash
  pip install /tmp/tensorflow_pkg/tensorflow-<version>-<tags>.whl
  ```

- That's it. If all goes well, you should see some sort of a success message. I also wanted to note that people on developer forums are often willing to share their own prebuilt wheels (the `.whl` file that took forever to generate). If you get lucky, someone might have a publicly available wheel that works for you. You could skip all the steps before the `pip install`!

### We can also create a virtual environment with Conda

- This makes it easy to handle multiple projects with vastly different sets of dependencies. In other words, we can have separate programming environments for separate purposes. For our case in particular, we're installing older versions of TensorFlow and CUDA so that we can use our graphics card for machine learning. Using `conda ` allows me to create an environment called `ml-env` , specifically for this use case. At the same time, I can have a `base` environment for all other general programming uses. Maybe this `base` environment will have the latest version of TensorFlow instead. This is why I love virtual environments—they're simple and powerful!

- Anyways, here's how to install the correct modules for an unsupported CUDA architecture. Note that I've included the versions that I need in the below commands, but yours may be different.

  ```bash
  conda create -n ml-env
  conda activate ml-env
  conda install tensorflow-gpu=1.12
  conda install cudatoolkit=9.0
  ```

- Then, double check that your GPU is being used

  ```python
  python # Open the python interpreter
  >>> import tensorflow as tf
  >>> tf.Session()
  ```

  If correct, you'll see something around the lines of `Created TensorFlow device -> physical GPU` outputted. 

- Note that when we were building from source, we didn't use virtual environments. If we wanted to, we'd have to install all the packages to our active environment.

# Point Cloud Classification

- Cloned repository at https://github.com/charlesq34/pointnet.git
- Modified `provider.py` to include `--no check-certificate` flag for downloading the ModelNet40 training data. Without this flag, running `train.py` failed because of a certificate error. 
- Began training the given model with a batch size of 8. It took 5 epoch's to achieve an eval accuracy of 70% and roughly 2 hours to do so on a GTX 750M. 
- Evaluated the performance of the incomplete model by running `python evaluate.py --visu`. Surprisingly, classifiying the airplane had an accuracy of 0.97! This suggests that we don't need the full ModelNet40 training set if we only want to classify something specific.  
- General tips about neural networks:
  - The number of epoch's is not that significant—determine your tolerance for error, and then iterate for as many epoch's as needed until you reach that threshold. For example, it took me 5 epoch's to achieve ~70% accuracy. If 70% was the minimum accuracy I desired, I could just stop there.
  - Look out for overfitting. For the first few iterations, mean error begins to decrease. However, in some cases, training the model for too long may result in overfitting. You'll notice the error start to increase again; this is a warning sign for overfitting. 
  - Epoch: one learning cycle where the classifier sees the whole training dataset. 
  - Does more neurons -> more accuracy? 
  - If you get an OOM (Out of Memory) error while trying to train your model, reduce your batch size and/or image dimensions. In general,  the 'standard' batch size is 32 (Usually powers of 2).
- Which implementation of the PointNet libary do we need?
  1. Classification -> Identify which plane (`PointNet`, `PointNet++`)
  2. Part Segmentation -> Visualize the plane's different parts, i.e this is the wing, this is the body, etc. 
  3. Semantic Segmentation -> Is there a plane present in the first place?
     - `VoxelNet`: based on PointNet, efficiently handles LiDar point clouds
     - `Frustrum PointNets`: 3D object detection w/ RGB-D data
- Outstanding Questions
  1. All of these models are traineed on the ModelNet40 or some other large dataset. Do we need to give our model all of this data? Does it need to be given a boat, couch, etc?
  2. Will we be placing the sensor on the wing and then classifying? Or are we giving our model a birds-eye view of the plane?



