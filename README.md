# Solution for ChaLearn Face Anti-spoofing Attack Detection Challenge @ CVPR2019 by a.parkin (VisionLabs)

The solution uses DLAS (Deep Layers Aggregation Solution) architecture models for each of 3 sources (RGB, Depth, IR)

## Picture of the architecture

Сreating the conda environment and installing the required libraries

```
conda create --name python3 --file spec-file.txt;
conda activate python3;
pip install -r requirements.txt
```


## Train
Used pretrained models for face or gender recognition

|Exp. Name|Model architecture|Train description|Link|Google Drive|
|:---:|:------------:|:-------------:|:--------:|:---------:|
|exp1_2stage|resnet caffe34|CASIA, sphere loss|[MCS2018](https://github.com/AlexanderParkin/MCS2018.Baseline)|link|
|exp2|resnet caffe34|Gender classifier on pretrained weights|./attributes_trainer|link|
|exp3b|IR50|MSCeleb, arcface|[face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch#Model-Zoo)|link|
|exp3c|IR50|asia(private) dataset, arcface|[face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch#Model-Zoo)|link|


### Step 1 (can be skipped)
Download all pretrained models (exp1_2stage, exp3b, exp3c links) and challenge train/val/test data

### Step 2 (can be skipped)
Download AFAD-Lite and train a model for gender recognition task

### Step 3 (can be skipped)

Train models:

* exp1
* exp2
* exp3b
* exp3c

or run ```train.sh```

## Inference
### Step 1 (can be skipped)
#### Step 1.1
Change data_root path in ```datasets/init_dataloader.py:23```
#### Step 1.2
Run all prepaired models from ```data/opts/``` and 
use ```inference.py``` or ```inference.sh```

### Step 2
ensemble all results

```
python ensemble.py
```
