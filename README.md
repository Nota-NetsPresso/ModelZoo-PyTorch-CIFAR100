<div align="center">
  <p>
    <a align="center" target="_blank">
      <img width="100%" src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/banner/Pytorch_Cifar%20model_banner.png"></a>
  </p>

</div>

# <div align="center">NetsPresso tutorial for PyTorch_CIFAR_Models compression</div>
## Order of the tutorial
[0. Sign up](#0-sign-up) </br>
[1. Install](#1-install) </br>
[2. Prepare the dataset](#2-prepare-the-dataset) </br>
[3. (Optioanl) Run tensorboard](#3-optional-run-tensorbard) </br>
[4. Training](#4-training) </br>
[5. Testing](#5-testing) </br>
[6. Model compression with NetsPresso Python Package](#6-model-compression-with-netspresso-python-package)</br>
[7. Fine-tuning the compressed model](#7-fine-tuning-the-compressed-model)</br>

## 0. Sign up
To get started with the NetsPresso Python package, you will need to sign up either at <a href="https://netspresso.ai/" target="_blank">NetsPresso</a> or <a href="https://py.netspresso.ai/?utm_source=git_cifar&utm_medium=page&utm_campaign=py_launch" target="_blank">PyNetsPresso</a>.
</br>

## 1. Install
Clone repo, including [**PyTorch >=1.11, < 2.0**](https://pytorch.org/get-started/locally/).
```bash
git clone https://github.com/Nota-NetsPresso/pytorch-cifar-models_nota.git  # clone
cd pytorch-cifar-models_nota
```
</br>

## 2. Prepare the dataset
We will use cifar100 dataset from torchvision since it's more convenient, but we also
kept the sample code for writing your dataset module in dataset folder, as an
example for people don't know how to write it.
</br>

## 3. (Optional) Run tensorbard
Install tensorboard
```bash
pip install tensorboard
mkdir runs
Run tensorboard
tensorboard --logdir='runs' --port=6006 --host='localhost'
```
</br>

## 4. Training
You need to specify the net you want to train using arg -net

```bash
# use gpu to train vgg16
python train.py -net vgg16 -gpu
```
Sometimes, you might want to use warmup training by set ```-warm``` to 1 or 2, to prevent network
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
</br>

## 5. Testing
Test the model using test.py
```bash
python test.py -net path_to_fx_model_file
```
</br>

## 6. Model compression with NetsPresso Python Package<br/>
Upload & compress your 'model-epoch-fx.pt' by using NetsPresso Python Package
### 6_1. Install NetsPresso Python Package
```bash
pip install netspresso
```
### 6_2. Upload & Compress
First, import the packages and set a NetsPresso username and password.
```python
from netspresso.compressor import ModelCompressor, Task, Framework, CompressionMethod, RecommendationMethod


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"
compressor = ModelCompressor(email=EMAIL, password=PASSWORD)
```
Second, upload 'model-epoch-fx.pt', which is the model converted to torchfx in step 4, with the following code.
```python
# Upload Model
UPLOAD_MODEL_NAME = "PyTorch_CIFAR_model"
TASK = Task.IMAGE_CLASSIFICATION
FRAMEWORK = Framework.PYTORCH
UPLOAD_MODEL_PATH = "./model-epoch-fx.pt"
INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [32, 32]}]
model = compressor.upload_model(
    model_name=UPLOAD_MODEL_NAME,
    task=TASK,
    framework=FRAMEWORK,
    file_path=UPLOAD_MODEL_PATH,
    input_shapes=INPUT_SHAPES,
)
```
Finally, you can compress the uploaded model with the desired options through the following code.
```python
# Recommendation Compression
COMPRESSED_MODEL_NAME = "test_l2norm"
COMPRESSION_METHOD = CompressionMethod.PR_L2
RECOMMENDATION_METHOD = RecommendationMethod.SLAMP
RECOMMENDATION_RATIO = 0.6
OUTPUT_PATH = "./compressed_model.pt"
compressed_model = compressor.recommendation_compression(
    model_id=model.model_id,
    model_name=COMPRESSED_MODEL_NAME,
    compression_method=COMPRESSION_METHOD,
    recommendation_method=RECOMMENDATION_METHOD,
    recommendation_ratio=RECOMMENDATION_RATIO,
    output_path=OUTPUT_PATH,
)
```

<details>
<summary>Click to check 'Full upload & compress code'</summary>

```bash
pip install netspresso
```

```python
from netspresso.compressor import ModelCompressor, Task, Framework, CompressionMethod, RecommendationMethod


EMAIL = "YOUR_EMAIL"
PASSWORD = "YOUR_PASSWORD"
compressor = ModelCompressor(email=EMAIL, password=PASSWORD)

# Upload Model
UPLOAD_MODEL_NAME = "PyTorch_CIFAR_model"
TASK = Task.IMAGE_CLASSIFICATION
FRAMEWORK = Framework.PYTORCH
UPLOAD_MODEL_PATH = "./model-epoch-fx.pt"
INPUT_SHAPES = [{"batch": 1, "channel": 3, "dimension": [32, 32]}]
model = compressor.upload_model(
    model_name=UPLOAD_MODEL_NAME,
    task=TASK,
    framework=FRAMEWORK,
    file_path=UPLOAD_MODEL_PATH,
    input_shapes=INPUT_SHAPES,
)

# Recommendation Compression
COMPRESSED_MODEL_NAME = "test_l2norm"
COMPRESSION_METHOD = CompressionMethod.PR_L2
RECOMMENDATION_METHOD = RecommendationMethod.SLAMP
RECOMMENDATION_RATIO = 0.6
OUTPUT_PATH = "./compressed_model.pt"
compressed_model = compressor.recommendation_compression(
    model_id=model.model_id,
    model_name=COMPRESSED_MODEL_NAME,
    compression_method=COMPRESSION_METHOD,
    recommendation_method=RECOMMENDATION_METHOD,
    recommendation_ratio=RECOMMENDATION_RATIO,
    output_path=OUTPUT_PATH,
)
```

</details>

More commands can be found in the official NetsPresso Python Package docs: https://nota-netspresso.github.io/netspresso-python-docs/build/html/index.html <br/>

Alternatively, you can do the same as above through the GUI on our website: https://console.netspresso.ai/models<br/><br/>

## 7. Fine-tuning the compressed model</br>
You may need to set learning rate using ```-lr```
```bash
$ python train.py -net path_to_compressed_model_file -lr 0.001
```
Now you can use the compressed model however you like! </br></br>

## <div align="center">Contact</div>

Join our <a href="https://github.com/orgs/Nota-NetsPresso/discussions">Discussion Forum</a> for providing feedback or sharing your use cases, and if you want to talk more with Nota, please contact us <a href="https://www.nota.ai/contact-us">here</a>.</br>
Or you can also do it via email(contact@nota.ai) or phone(+82 2-555-8659)!

<br>
<div align="center">
  <a href="https://github.com/Nota-NetsPresso" style="text-decoration:none;">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/github_white.png">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/github.png">
      <img alt="github" src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/github.png" width="3%">
    </picture>
  </a>
  <img src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.facebook.com/NotaAI" style="text-decoration:none;">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/facebook_white.png">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/facebook.png">
      <img alt="facebook" src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/facebook.png" width="3%">
    </picture>
  </a>
  <img src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/logo-transparent.png" width="3%" alt="" />
  <a href="https://twitter.com/nota_ai" style="text-decoration:none;">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/twitter_white.png">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/twitter.png">
      <img alt="twitter" src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/twitter.png" width="3%">
    </picture>
  </a>
  <img src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.youtube.com/channel/UCeewYFAqb2EqwEXZCfH9DVQ" style="text-decoration:none;">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/youtube_white.png">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/youtube.png">
      <img alt="youtube" src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/youtube.png" width="3%">
    </picture>
  </a>
  <img src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.linkedin.com/company/nota-incorporated" style="text-decoration:none;">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/linkedin_white.png">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/linkedin.png">
      <img alt="youtube" src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/common/linkedin.png" width="3%">
    </picture>
  </a>
</div>
<br>
