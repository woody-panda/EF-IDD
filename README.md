The official codebase for Exemplar-Free Incremental Deepfake Detection
<p align="center">
  <img src="https://github.com/woody-panda/EF-IDD/blob/main/figures/setting.png" width=75% height=75%>
</p>

### Introduction
This repo contains the official PyTorch implementation of the paper: Exemplar-Free Incremental Deepfake Detection.

- [x] 起床
* [ ] 洗漱
+ [ ] 吃早餐

### Data preparation and preprocessing

- Download the datasets: [FF++](https://github.com/ondyari/FaceForensics), [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics), [DFDC-P](https://ai.meta.com/datasets/dfdc/), [DFFD](https://cvlab.cse.msu.edu/dffd-dataset.html), [FFIW](https://github.com/tfzhou/FFIW), [OpenForensics](https://sites.google.com/view/ltnghia/research/openforensics), [ForgeryNIR](https://github.com/AEP-WYK/forgerynir), [ForgeryNet](https://yinanhe.github.io/projects/forgerynet.html)

- Data preprocessing. For the all datasets, we use [RetainFace](https://github.com/biubug6/Pytorch_Retinaface) to align the faces. Use the code in directory ./preprocessing to get the preprocessed data.
