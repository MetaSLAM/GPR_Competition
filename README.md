# GPR_Competition_datasets
Dataset for MetaSLAM Challenge. The website: [http://gprcompetition.com/](http://gprcompetition.com/)

# Datasets
We provide four (more in the future) datasets for evaluating place recognition or global localization methods. They are
- **CMU Day/Night UGV**: 
- **CMU Helicopter Terrain**:
- **CMU UAV Localization**:
- **Pittsburgh Large Scale**: It has LiDAR point clouds and ground truth poses for 55 trajectories. Each trajectory is divided into several submaps, and a submap has size 50m*50m with the distance between every two submaps being 2m. [Download](https://xxxx).

(Maybe we can put a summary table here~)

Datasets are provided in the [kapture format](https://github.com/naver/kapture). You can easily manage the data with our tools. For more information about dataset, please refer to [dataset description](./docs/dataset_description.md).


# Software
Tools, such as data loader and evaluation metrics, are also provided in Python. The minimum Python version is 3.6. For package dependencies, see [requirements.txt](./requirements.txt).

## Install
The easiest way to install our tools is by using pip. We recommend the use of virtual environment such as `conda` to keep a clean software environment.

First, clone this repository:
```bash
~$ git clone https://github.com/MetaSLAM/MetaSLAM_datasets.git
```

Then, set up the enviroment. Here we create a new virtual environment via conda. You can skip the first two steps if you already have one (but still need to install requirements):
```bash
~$ conda create --name MetaSLAM python=3.6
~$ source activate MetaSLAM
(MetaSLAM) ~$ cd MetaSLAM_datasets
(MetaSLAM) ~/MetaSLAM_datasets$ pip install -r requirements.txt
```

To make this tool available for Python to import, usually we have *two* ways. You need to choose **one**:
- Add this repo to the variable $PYTHONPATH in the file .bashrc (recommended):
    ```bash
    ~$ vim .bashrc

    # Add the following lines to the end of file, then save and quit.
    export PYTHONPATH=~/MetaSLAM_datasets:$PYTHONPATH
    ```
- Or add this repo in your code:
    ```python
    import sys
    sys.path.insert(0, '~/MetaSLAM_datasets')
    ```

Finally, you have successfully installed our tools. You can have a test with `import metaslam` in the python interpreter.


# Tutorial
We provide detailed documents about the usage of this package, which can be found in the `docs` folder.

## Modules
Our package organizes different functions in sub-modules. You may have a better understanding of the `metaslam` package with this table:

module | description   
:--:   |--
`metaslam`|some common operations
`metaslam.dataloader`|load dataset from disk, get images, point clouds, poses, etc.
`metaslam.evaluation`|evaluate your method, such as recall@N, accuracy, PR curve
`metaslam.utils`|utility, such as rotation, translation, transformation

## Quick Start
### Load data
Assume you have download the `Pittsburgh Large Scale dataset`, then you can load it and get the point cloud data:
```python
import metaslam

dataset_path = 'PATH_TO_THE_DATASET'
dataset = metaslam.load_dataset(dataset_path) # List[traj1, traj2, ...]

traj0 = dataset[0] # select the first trajectory
frame_id = 88      # get the data of 88th frame
pcd = traj0.get_point_cloud(frame_id) # open3d.geometry.PointCloud
```
For more about the data loader, please refer to [loading_data.md](./docs/loading_data.md).
