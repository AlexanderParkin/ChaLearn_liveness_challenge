# Solution for ChaLearn Face Anti-spoofing Attack Detection Challenge @ CVPR2019 by a.parkin (VisionLabs)

![Alt text](fact_sheets/figures/net.jpg?raw=true "Network Architecture")

Our method uses a modified network architecture in [1]. As shown on image, the RGB, Depth and IR inputs are processed by separate streams followed by the concatenation and fully-connected layers. Differently from [1] we use aggregation blocks (Agg res2, ...) to aggregate outputs from multiple layers of the network. We pre-train network weights on four different tasks for face recognition and gender recognition. We then fine- tune these networks separately on the training set of the CASIA-SURF face anti-spoofing dataset. To increase the robustness to various attacks, we ensemble networks trained on three training folds and with two initial seeds. Results of our models evaluated separately and in combination are illustrated in table.

| NN1 | NN1a | NN2 | NN3 | NN4 | seed | Val trp@fpr=10e-4 | Test trp@fpr=10e-4 |
|:-----:|:------:|:-----:|:-----:|:-----:|:------:|:-------------------:|:--------------------:|
|:heavy_check_mark:|      |     |     |     |      | 0.9943            |                    |
|     |   :heavy_check_mark:  |     |     |     |      | 0.9987            |                    |
|     |      | :heavy_check_mark:   |     |     |      | 0.9870            |                    |
|     |      |     | :heavy_check_mark:   |     |      | 0.9963            |                    |
|     |      |     |     | :heavy_check_mark:   |      | 0.9933            |                    |
| :heavy_check_mark:   |      | :heavy_check_mark:   |     |     |      | 0.9963            |                    |
| :heavy_check_mark:   |      | :heavy_check_mark:   | :heavy_check_mark:   |     |      | 0.9983            |                    |
| :heavy_check_mark:   |      | :heavy_check_mark:   | :heavy_check_mark:   |     | :heavy_check_mark:    | 0.9997            |                    |
| :heavy_check_mark:   |      | :heavy_check_mark:  | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:    | 1.0000            |                    |
|     | :heavy_check_mark:    | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:   | :heavy_check_mark:    | **1.0000**|**0.9988**|


## References
[1] Shifeng Zhang, Xiaobo Wang, Ajian Liu, Chenxu Zhao, Jun Wan, Ser- gio Escalera, Hailin Shi, Zezheng Wang, Stan Z. Li, ”CASIA-SURF: A Dataset and Benchmark for Large-scale Multi-modal Face Anti-spoofing”, arXiv, 2018.

## Environment
Сreating the conda environment and installing the required libraries

```
conda create --name python3 --file spec-file.txt;
conda activate python3;
pip install -r requirements.txt
```


## Train
Used pretrained models for face or gender recognition

|Exp. Name|Model architecture|Train description|Architecture|Weights|
|:---:|:------------:|:-------------:|:--------:|:---------:|
|exp1_2stage|resnet caffe34|CASIA, sphere loss|[MCS2018](https://github.com/AlexanderParkin/MCS2018.Baseline)|[Google Drive](https://drive.google.com/open?id=1dnfh7rSrGV9_ROQ6TcRU6O1Kn0ZP5kEQ)|
|exp2|resnet caffe34|Gender classifier on AFAD-Lite|./attributes_trainer|[Google Drive](https://drive.google.com/file/d/1-FhqBZ14qNQANQrs-7jYlBgaFy9bC3zN)|
|exp3b|IR50|MSCeleb, arcface|[face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch#Model-Zoo)|[Google Drive](https://drive.google.com/file/d/1-OCl0xt0f4eBwzKWYW5sErV_snHlZMBa)|
|exp3c|IR50|asia(private) dataset, arcface|[face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch#Model-Zoo)|[Google Drive](https://drive.google.com/open?id=1-DFXeauUKY0O5-1KWQ0-Ojyu0Nzpf84H)|


### Step 1 (can be skipped)
Download all pretrained models ([Google Drive](https://drive.google.com/file/d/1-FAmtxFTXJBl-G20W_naEM78PfaOXTpz/view?usp=sharing)) and challenge train/val/test data

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
