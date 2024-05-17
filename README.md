# Domain generalization counting via Graph-based Meta-Learning and Feature Disentanglement

![visualization](./samples_imgs/visualization.png)

## 1.Environment Setup

Python library dependencies.

```text
python 3.9.12
torch==1.11.0
torchvision==0.12.0
pandas
scikit-learn
```

## Train

- train.py

```shell
python train.py
```

If you want to replace the dataset, you can use the variable data-dir = 'dataset path' in the train.py file. There are also a number of parameters that can be modified to suit your needs, such as learning rate, epochs, etc.

## Test

Download our trained model, .

**If you want to use our model quickly, you can use the test.py file directly and modify the path to the model in the file to get our results quickly.**

- test.py

```shell
python test.py
```

If you want to replace the dataset, you can use the variable data-dir = 'dataset path' in the test.py file.
