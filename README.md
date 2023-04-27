# Pytorch-cifar100_nota

practice on cifar100 using pytorch

## Requirements

This is my experiment eviroument
- python3.6
- pytorch1.6.0+cu101
- tensorboard 2.2.2(optional)


## Usage

### 1. enter directory
```bash
$ cd pytorch-cifar100_nota
```

### 2. dataset
I will use cifar100 dataset from torchvision since it's more convenient, but I also
kept the sample code for writing your own dataset module in dataset folder, as an
example for people don't know how to write it.

### 3. run tensorbard(optional)
Install tensorboard
```bash
$ pip install tensorboard
$ mkdir runs
Run tensorboard
$ tensorboard --logdir='runs' --port=6006 --host='localhost'
```

### 4. train the model
You need to specify the net you want to train using arg -net

```bash
# use gpu to train vgg16
$ python train.py -net vgg16 -gpu
```

sometimes, you might want to use warmup training by set ```-warm``` to 1 or 2, to prevent network
diverge during early training phase.

The supported net args are:
```
vgg16
repvgg
mobilenetv2

or
saved model path
```
The pretrained models are from [here](https://github.com/chenyaofo/pytorch-cifar-models).


### 5. test the model
Test the model using test.py
```bash
$ python test.py -net path_to_fx_model_file
```

### 6. compress the model
Visit [netspresso.ai](https://netspresso.ai/) and compress the model. You can get step by step guide from [here](https://docs.netspresso.ai/docs/mc-step1-prepare-model).

### 7. fine-tune the model
You may need to set learning rate using ```-lr```
```bash
$ python train.py -net path_to_compressed_model_file -lr 0.001
```