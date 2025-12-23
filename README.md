# CLIP4Cir-MoE

### Composed Image Retrieval with CLIP-based Multimodal Fusion and Mixture of Experts

This repository provides an implementation for **Composed Image Retrieval (CIR)** based on CLIP features, extended with an enhanced **Combiner module incorporating Mixture of Experts (MoE)** for adaptive multimodal fusion.

The system retrieves target images given:
- a **reference image**, and
- a **modification text description**,

by learning a joint representation that integrates visual and textual information.

---

## About the Project

### Overview

Given a query composed of a reference image and a relative caption, the Composed Image Retrieval task aims to retrieve images that are visually similar to the reference image while reflecting the semantic modifications described by the text.

This project builds upon CLIP visual and textual encoders and follows a **two-stage training pipeline**:
1. **Task-oriented fine-tuning of CLIP encoders**
2. **Training of a dedicated Combiner network** for multimodal fusion

The Combiner has been **extended with a Mixture of Experts (MoE) mechanism**, enabling adaptive routing and improved modeling of diverse image–text interaction patterns.

---

## Composed Image Retrieval Task

![](images/cir-overview.png)

---

## CLIP Task-Oriented Fine-Tuning

![](images/clip-fine-tuning.png)

**Stage 1 – CLIP fine-tuning**

In the first stage, both CLIP encoders are fine-tuned to reduce the gap between large-scale pretraining and the composed image retrieval task.

- Image and text features are extracted
- Features are combined via element-wise summation
- A contrastive loss aligns query features with target image features

Both CLIP encoders are updated during this stage.

---

## Combiner Training

![](images/combiner-training.png)

**Stage 2 – Combiner training**

- CLIP encoders are frozen
- A Combiner network is trained from scratch
- Contrastive learning aligns combined representations with target images

---

## Combiner Architecture (with MoE)

![](images/Combiner-architecture.png)

The Combiner network includes:
- Projection layers for text and image embeddings
- Cross-modal attention
- Mixture of Experts (MoE) for adaptive fusion
- Element-wise gating and residual connections

---

## Installation

```bash
git clone https://github.com/ABaldrati/CLIP4Cir
cd CLIP4Cir
conda create -n clip4cir -y python=3.8
conda activate clip4cir
conda install -y -c pytorch pytorch=1.11.0 torchvision=0.12.0
conda install -y -c anaconda pandas=1.4.2
pip install comet-ml==3.21.0
pip install git+https://github.com/openai/CLIP.git
```

### Data Preparation

To properly work with the codebase FashionIQ and CIRR datasets should have the following structure:

```
project_base_path
└───  fashionIQ_dataset
      └─── captions
            | cap.dress.test.json
            | cap.dress.train.json
            | cap.dress.val.json
            | ...
            
      └───  images
            | B00006M009.jpg
            | B00006M00B.jpg
            | B00006M6IH.jpg
            | ...
            
      └─── image_splits
            | split.dress.test.json
            | split.dress.train.json
            | split.dress.val.json
            | ...

└───  cirr_dataset  
       └─── train
            └─── 0
                | train-10108-0-img0.png
                | train-10108-0-img1.png
                | train-10108-1-img0.png
                | ...
                
            └─── 1
                | train-10056-0-img0.png
                | train-10056-0-img1.png
                | train-10056-1-img0.png
                | ...
                
            ...
            
       └─── dev
            | dev-0-0-img0.png
            | dev-0-0-img1.png
            | dev-0-1-img0.png
            | ...
       
       └─── test1
            | test1-0-0-img0.png
            | test1-0-0-img1.png
            | test1-0-1-img0.png 
            | ...
       
       └─── cirr
            └─── captions
                | cap.rc2.test1.json
                | cap.rc2.train.json
                | cap.rc2.val.json
                
            └─── image_splits
                | split.rc2.test1.json
                | split.rc2.train.json
                | split.rc2.val.json
```

---

### CLIP fine-tuning

To fine-tune the CLIP model on FashionIQ or CIRR dataset run the following command with the desired hyper-parameters:

```sh
python src/clip_fine_tune.py \
   --dataset {'CIRR' or 'FashionIQ'} \
   --api-key {Comet-api-key} \
   --workspace {Comet-workspace} \
   --experiment-name {Comet-experiment-name} \
   --num-epochs 100 \
   --clip-model-name RN50x4 \
   --encoder both \
   --learning-rate 2e-6 \
   --batch-size 128 \
   --transform targetpad \
   --target-ratio 1.25  \
   --save-training \
   --save-best \
   --validation-frequency 1 
```

### Combiner training

To train the Combiner model on FashionIQ or CIRR dataset run the following command with the desired hyper-parameters:

```sh
python src/combiner_train.py \
   --dataset {'CIRR' or 'FashionIQ'} \
   --api-key {Comet-api-key} \
   --workspace {Comet-workspace} \
   --experiment-name {Comet-experiment-name} \
   --projection-dim 2560 \
   --hidden-dim 5120 \
   --num-epochs 300 \
   --clip-model-name RN50x4 \
   --clip-model-path {path-to-fine-tuned-CLIP} \
   --combiner-lr 2e-5 \
   --batch-size 4096 \
   --clip-bs 32 \
   --transform targetpad \
   --target-ratio 1.25 \
   --save-training \
   --save-best \
   --validation-frequency 1
```

### Validation

To compute the metrics on the validation set run the following command

```shell
python src/validate.py 
   --dataset {'CIRR' or 'FashionIQ'} \
   --combining-function {'combiner' or 'sum'} \
   --combiner-path {path to trained Combiner} \
   --projection-dim 2560 \
   --hidden-dim 5120 \
   --clip-model-name RN50x4 \
   --clip-model-path {path-to-fine-tuned-CLIP} \
   --target-ratio 1.25 \
   --transform targetpad
```

### Test

To generate the prediction files to be submitted on CIRR evaluation server run the following command:

```shell
python src/cirr_test_submission.py 
   --submission-name {file name of the submission} \
   --combining-function {'combiner' or 'sum'} \
   --combiner-path {path to trained Combiner} \
   --projection-dim 2560 \
   --hidden-dim 5120 \
   --clip-model-name RN50x4 \
   --clip-model-path {path-to-fine-tuned-CLIP} \
   --target-ratio 1.25 \
   --transform targetpad
```
