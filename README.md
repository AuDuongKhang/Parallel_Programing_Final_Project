## Features and Functionality

Our project creates an basic MLP using **Cuda C/C++**. We have 2 main versions of it, **CPU** and **GPU**. In **GPU** version, we try to optimize it 
having a better performance.


## How to Run the Code

### We are using Colab Notebook to run all the project, so follow these code to run project, each step is in 1 code cell

#### Step 1: Set up the Environment
1. `!wget https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64 -O cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb`

   `!dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb`

   `!apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub`

   `!apt-get update`

   `!apt-get install cuda-9.2`

2. `!apt-get install -y cmake`

#### Step 2: Mount Google Drive to the Colab Environment
1. `from google.colab import drive`

   `drive.mount('/content/drive')`

2. `dir = "path/to/your/directory"`

   `%cd $dir/build`

#### Step 3: Running with Cmake
1. `!make clean`

   `!cmake ..`

   `!make`

   `!./Project`

#### Parameter to run the file
Parameter to run file:

*argv[1]*: mode (default mode = 0). There are 9 modes:
- Mode = 0: CPU Implement.
- Mode = 1: GPU basic Implement.
- Mode = 2: GPU Implement with shared memory.
- Mode = 3: GPU Implement with fushed shared memory and unrolling.
- Mode = 4: GPU Implement with constant memory.
- Mode = 5: GPU Implement with restrict and unrolling.
- Mode = 6: GPU Implement with reduction tree.
- Mode = 7: GPU Implement with reduction atomic.
- Mode = 8: GPU Implement with Strassen algorithm (advanced matrix multiplication).
- Mode = 9: GPU Implement with multiple kernel for different layer sizes. (Not already developed)
- Mode = 10: GPU Implement with FP16 arithmetic.

*argv[2]*: number of epochs (default epochs = 20).

*argv[3]*: number of batch size (default batch size = 64).

*argv[4]*: learning rate (default learning rate = 0.01).

***Example***: 
1. `!./Project 1` will run **GPU** basic implement.
2. `!./Project 1 10` will run **GPU** basic implement with 10 epochs.
3. `!./Project 1 10 32` will run **GPU** basic implement with 10 epochs and batch size = 32.
4. `!./Project 1 10 32 0.001` will run **GPU** basic implement with 10 epochs, batch size = 32 and learning rate = 0.001.
