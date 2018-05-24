# Super Convergence Experiments

## Dependencies

- Python 3.6.3
- PyTorch 0.3.0
- torchvision 0.2.0
- Numpy 1.13.3
- Matplotlib 2.1.0
- Jupyter notebooks
- Cuda & CUDNN for running on a NVIDIA GPU


## Folder structure

- data/ - Folder where CIFAR-10 will be downloaded
- learning_curves/ - Folder where learning curves will be saved as numpy arrays
- notebooks/
  - learning_rate_finder.ipynb - Jupyter notebook for exploring the baseline learning rate
  - plotting.ipynb - Notebook for plotting the saved learning curve numpy arrays
- plots/ - Folder where the plots will be saved
- src/
  - plots.py - Functionality for plotting learning curves, learning rates and momentums
  - resnet.py - Class for building different ResNet architectures
  - super_convergence.py - Training script for running multiple experiments on super convergence
  - train_adam.py - Training script running the baseline Adam optimized model
  - train_utils.py - Utilities for training models normally and following the one-cycle proposition

## How to run

Warning: The following models take a long time to train even on a GPU.

For running the baseline run:

```
$ cd src && python train_adam.py
```

For running all the experiments run:

```
$ cd src && python super_convergence.py
```

For futher insight into the way of choosing the baseline learning rate, please
refer to the ```learning_rate_finder.ipynb``` Jupyter notebook.

For the plots used in the report, please refer to the ```plotting.ipynb``` Jupyter notebook.
