# Matterport3D Simulator
### Prerequisites

- Ubuntu 16.04
- Nvidia GPU with driver >= 384

### Clone Repo

Clone the Matterport3DSimulator repository:
```
# Make sure to clone with --recursive
```
If you didn't clone with the `--recursive` flag, then you'll need to manually clone the pybind submodule from the top-level directory:
```
git submodule update --init --recursive
```
### Building the Simulator
We don't use docker to build the Simulator.
1. install required packages
```
sudo apt-get update && sudo apt-get install -y wget doxygen curl libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python-opencv python-setuptools python-dev
```
2. make sure you have cmake>=3.12, otherwise go to `https://cmake.org` for information.
3. Run 
```
mkdir build && cd build
cmake -DPYTHON_EXECUTABLE:FILEPATH=$(which python) ..
make
cd ../
```
### Dataset Download

### Dataset Preprocessing
