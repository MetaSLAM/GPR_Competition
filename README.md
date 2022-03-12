# GPR Competition datasets

Dataset for the General Place Recognition Competition. You can find more details in the competition website: [http://gprcompetition.com/](http://gprcompetition.com/)

## Datasets

We provide two datasets for evaluating place recognition or global localization methods. They are

- **Pittsburgh City-scale Dataset**: This dataset aims for the Large-scale 3D Localization (LiDAR→LiDAR) competition track. It has LiDAR point clouds and ground truth poses for 55 trajectories, collected in Pittsburgh. Each trajectory is divided into several submaps, and a submap has size 50m*50m with the distance between every two submaps being 2m. [Download](https://xxxx).
    
    ![pitts_large-scale](docs/data_pics/Pittsburgh_City-scale_Dataset.png)

    In this dataset, we include:
    * Point cloud submaps (size 50m*50m, every 2m along the trajectory).
    * Ground truth poses of submaps (6DoF)

    You can find the **sample** training data `gpr_pitts_sample.tar.gz` and testing/query data `gpr_pitts_query_sample.tar.gz` [here](https://sandbox.zenodo.org/record/1033096).

- **CMU Cross-Domain Dataset**: This dataset focuses on visual localization for UGVs using omnidirectional cameras within outdoor campus-type environments. We collected 80 real-world UAV sequences using a rover robot equipped with a 360 camera, a Velodyne VLP-16 LiDAR, a RealSense VIO and an Xsens MTI IMU. These consisted of 10 different trajectories. For each trajectory, we traversed 8 times, including forward(start point to endpoint)/backward(endpoint to start point) directions and day-light (2pm to 4:30pm)/dawn-light (6am to 7am or 5pm to 6pm). 8-times includes two forward sequences and two backward sequences during day-light and two forward and two backward sequences during dawn-light.Each trajectory is at least overlapped at one junction with the others,and some trajectories even have multiple junctions. This feature enables the dataset to be used in tasks such as LiDAR place recognition and multi-map fusion. [Download](https://xxxx).  
    
    ![CMU_Lifelong](docs/data_pics/cmu_lifelong.png)


    In this dataset, we include:
    * High resolution (1024x512) omnidirectional imagery, captured at 15fps. Timestamps are synchronized with the rest of the system.
    * Timestamped IMU (linear accelerations and angular velocities)
    * Timestamped VLP-16 LiDAR generated point cloud data
    * Timestamped RealSense generated odometry data

Relative ground truth for each sequence compared with the corresponding selected reference sequence is provided.

Datasets are *pre-processed* and you can easily manage the data with our tools. For more information about dataset, please refer to [dataset description](./docs/dataset_description.md).

## Install

The easiest way to install our tools is by using pip. We recommend the use of virtual environment such as `conda` to keep a clean software environment.

```bash
~$ git clone https://github.com/MetaSLAM/GPR_Competition.git
~$ conda create --name GPR python=3.7
~$ conda activate GPR
(GPR) ~$ cd GPR_Competition
(GPR) ~/GPR_Competition$ pip install -r requirements.txt
(GPR) ~/GPR_Competition$ python setup.py install
```

## Modules

Our package organizes different functions in sub-modules. You may have a better understanding of the `gpr` package with this table:

module | description
:--:   |--
`gpr`|common definations
`gpr.dataloader`|load dataset from disk, get images, point clouds, poses, etc.
`gpr.evaluation`|evaluate your method, such as recall@N, accuracy, PR curve
`gpr.tools`|utility, such as feature extraction, point cloud projection

## Quick Start

To quickly use this package for your place recognition task, we provide the test templates (both python scripts and jupyter notebooks) within the folder `tests/`. For **Pittsburgh City-scale Dataset** datasets, we can start a quick evaluation.

First, download the sample data [here](https://sandbox.zenodo.org/record/1033096). Decompress `gpr_pitts_sample.tar.gz` to **PATH_TO_DATA**. Then you can test it with the following code:

```python
import numpy as np
from gpr.dataloader import PittsLoader
from gpr.evaluation import get_recall
from gpr.tools import HogFeature, lidar_trans

# * Test Data Loader, change to your datafolder
pitts_loader = PittsLoader('PATH_TO_DATA')

# * Point cloud conversion and feature extractor
lidar_to_sph = lidar_trans()  # for lidar projections
hog_fea = HogFeature()

# feature extraction
feature_ref = []
for idx in tqdm(range(len(pitts_loader)), desc='comp. fea.'):
    pcd_ref = pitts_loader[idx]['pcd']

    # You can use your own method to extract feature
    sph_img = lidar_to_sph.sph_projection(pcd_ref)
    sph_img = (sph_img * 255).astype(np.uint8)
    feature_ref.append(hog_fea.infer_data(sph_img))

# evaluate recall
feature_ref = np.array(feature_ref)
topN_recall, one_percent_recall = get_recall(feature_ref, feature_ref)
```

For more about the data loader, visualization and evaluation, please refer to [loading_data.md](./docs/loading_data.md) and the jupyter notebook [test_pitts.ipynb](./tests/test_pitts.ipynb).
