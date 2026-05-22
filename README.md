# Code for paper: "When Bayes met Longuet-Higgins"

* https://github.com/sfikas/ECCV2024-Relative-Pose-on-a-Scene-with-Moving-Objects

### Installation

* Testing on Python 3.8.10.

~~* First create a virtual environment, followed by ``pip3 install --upgrade pip``.~~
* Then:
```
python3 -m venv venv
pip3 install --upgrade pip
pip install normflows ipykernel matplotlib tqdm opencv-python
```

### Types of experiments

* I: Experiments on synthetic point clouds.
  * Two sets of point clouds, one for each scene. Most of the points remain stationary, some of them move; these correspond to object motion.
  * The two point clouds are projected via two "P" matrices to two sets of projected image points; 
  an amount of noise is added on the coordinates of the points; these image points are the input to our model.
  * We assume that point correspondences are *exactly known* -- this corresponds to a matcher that is 100% correct.


### Notes on files in this repository

* ```bayesrpe.py```: Basic implementation files.
* ```experiments_01_Synthetic_Point_Clouds.ipynb```: Notebook testing the