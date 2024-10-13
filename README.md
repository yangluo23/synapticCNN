# synapticCNN

This is the implement of `synapticCNN` based on Pytorch framework.

## Installation

All experiments were conducted on the NVIDIA Geforce RTX3090, in a system environment of CentOS 7.9, with Pytorch version 1.8.0, and Python 3.8.0.

```shell
pip install -r requirements.txt
```

## NN pre-training

The `train.py` script is used to train the models. You can specify the optimizer, dataset, number of epochs, learning rate, batch size, model name and some training recipes.
The detailed parameter setting can be found in the `train.py` file, or by running
```shell
python train.py --help
```


The following are examples of how to train the models:

  * train `Basic Conv-Fc` CNNs:
    We trained these types of models using the MNIST dataset, offering three models with different depths: `cnn3`, `cnn5`, `cnn7`.
    ```python
    python train.py -o sgdm --model cnn3
    ```
  * train `VGG` models: 
    We trained these types of models using the CIFAR10 dataset, offering five models with different depths: `VGG6`, `VGG8`, `VGG11`, `VGG13`, `VGG16`.
    ```python
    python train.py -o sgdm -d cifar -e 200 --lr 1e-2 -b 128 --eval-bsz 256 --model vgg11
    ```
  * train `ResNet` models:
    We trained these types of models using the CIFAR100 dataset, offering four models with different depths: `resnet18`, `resnet34`, `resnet50`.
    ```python
    python train.py -o sgdm -d cifar100 -e 200 --lr 1e-1 -b 128 --eval-bsz 256 --model res18
    ```

Here, we provide our pre-trained model weights for [download](https://drive.google.com/drive/folders/17k_LpRBk1XXbCuuIeb9SqsnidVvys8qb?usp=drive_link):

| Model Name | Dataset  | ckpt     | Acc@1 |
|------------|----------|----------|-------|
| BasicCNN3  | MNIST    | [ckpt](https://drive.google.com/file/d/1f0cUTTXS9Q4QBno9lmzHYS9vBdXOmZH9/view?usp=drive_link) | 96.17 |
| BasicCNN5  | MNIST    | [ckpt](https://drive.google.com/file/d/1wYdDM7vpxpRNroSownvMQ9vj6Tg8j_is/view?usp=drive_link) | 95.84 |
| BasicCNN7  | MNIST    | [ckpt](https://drive.google.com/file/d/1Vy8fZN6lOZcUFLJ-FhMU8vz-SrYUtzcn/view?usp=drive_link) | 94.97 |
| Vgg6       | CIFAR10  | [ckpt](https://drive.google.com/file/d/1j2k0q4M2-rIuILXR2IVsMteMKTg5wd9-/view?usp=drive_link) | 84.46 |
| Vgg8       | CIFAR10  | [ckpt](https://drive.google.com/file/d/1FVgDS9yuoXsJoawbzVuDx_xw7JP4VN5i/view?usp=drive_link) | 87.00 |
| Vgg11      | CIFAR10  | [ckpt](https://drive.google.com/file/d/1KEIKgZEBuCF1ezS0BLAzVbKy3DXVdctP/view?usp=drive_link) | 87.36 |
| Vgg13      | CIFAR10  | [ckpt](https://drive.google.com/file/d/1ygF0OQSH26IZSlrIviXYRZVf2OjLPsvR/view?usp=drive_link) | 87.48 |
| Vgg16      | CIFAR10  | [ckpt](https://drive.google.com/file/d/1yQVUQXgLshMze8OcbRllfO6YH4QSouF7/view?usp=drive_link) | 88.12 |
| ResNet18   | CIFAR100 | [ckpt](https://drive.google.com/file/d/1MXcKZCo6HNsLSVlHJFHVhjaxEp_VbHpe/view?usp=drive_link) | 71.31 |
| ResNet34   | CIFAR100 | [ckpt](https://drive.google.com/file/d/1nPYUvNFfqOLw1Koa2E_pBe9LXlvuKeVz/view?usp=drive_link) | 71.94 |
| ResNet50   | CIFAR100 | [ckpt](https://drive.google.com/file/d/1luJhgdxH7RuxEtDKiwR_pO3eUXOe8fpZ/view?usp=drive_link) | 72.51 |


## NN Testing

After models pre-trained on corresponding datasets, the performance of the models can be evaluated by running `test.py`. 

Here is an example:

```shell
python test.py --ckpt ./ckpt/mnist/cnn3.pkl \
               -d mnist \
               -- model cnn3
```

NOTE: The `--ckpt` parameter specifies the path to the pre-trained model weights, you may replace it with your own local path.
The `--model` parameter specifies the model name, and the `-d` parameter specifies the dataset name.
The detailed parameter setting can be found in the `test.py` file, or by running
```shell
python test.py --help
```

## NN Quantization/Mapping

The provided codebook is linearly scaled, and the scaling factor and bias factor are obtained using a simulated annealing algorithm for search optimization. 
Then, the trained model weights are mapped.

By using the `quantize_test.py` script, you can map the model weights and save the model weights.
The detailed parameter setting can be found by running
```shell
python quantize_test.py --help
```

Here is an example:
```shell
python quantize_test.py -d mnit \
                        --model cnn3 \
                        --codebook ./data/codebook.xlsx --ckpt ./ckpt/mnist/cnn3.pkl
```

You may evaluate the performance of the mapped model by re-running `test.py` with new weights in the above step.
We also provide the mapped model weights for [download](https://drive.google.com/drive/folders/1jFzDd-M4396lDewDWDbLq7a06Hq10sFn?usp=drive_link):

| Model Name | Dataset  | ckpt     | Acc@1 |
|------------|----------|----------|-------|
| BasicCNN3  | MNIST    | [ckpt](https://drive.google.com/file/d/1fcKzj68n65HLeJ2-ZEVDOkK8geqG03Bj/view?usp=drive_link) | 95.17 |
| BasicCNN5  | MNIST    | [ckpt](https://drive.google.com/file/d/1edxgAKGBAXl4akPEANdMZHZV20bV3FBF/view?usp=drive_link) | 93.76 |
| BasicCNN7  | MNIST    | [ckpt](https://drive.google.com/file/d/1HIXToVJ2QSARYYOIfn0mVGDQTIpYRQaL/view?usp=drive_link) | 94.17 |
| Vgg6       | CIFAR10  | [ckpt](https://drive.google.com/file/d/1ZANg1AC76StfUpdx68ZHYbr91vAbz3f_/view?usp=drive_link) | 82.69 |
| Vgg8       | CIFAR10  | [ckpt](https://drive.google.com/file/d/1NRUN0HUbh04AReC3ed6JqDx_w6dnAg5Z/view?usp=drive_link) | 85.93 |
| Vgg11      | CIFAR10  | [ckpt](https://drive.google.com/file/d/1WLBSaVXG5jt6VM1hNhIupxe-iICbip2X/view?usp=drive_link) | 86.60 |
| Vgg13      | CIFAR10  | [ckpt](https://drive.google.com/file/d/1xmeAN5xytvfytMKW0L4S8mxhkHTHT_a7/view?usp=drive_link) | 86.30 |
| Vgg16      | CIFAR10  | [ckpt](https://drive.google.com/file/d/1gA0C-W5yHXY5dfkOWp_no1vKxhdh4QcS/view?usp=drive_link) | 86.57 |
| ResNet18   | CIFAR100 | [ckpt](https://drive.google.com/file/d/1snZx-fU_AcfAh5MBoh6mtM7d7xuMZUmX/view?usp=drive_link) | 68.20 |
| ResNet34   | CIFAR100 | [ckpt](https://drive.google.com/file/d/1MXtG7W98LkOjawnWakvsPB89-KsnzaCK/view?usp=drive_link) | 69.02 |
| ResNet50   | CIFAR100 | [ckpt](https://drive.google.com/file/d/1WbYnqGqVRj04iPMbfztoirmjcpeao6jM/view?usp=drive_link) | 69.47 |