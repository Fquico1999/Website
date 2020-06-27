---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Accurate Image Super-Resolution Using Very Deep Convolutional Networks"
subtitle: ""
summary: "Using a Deep CNN to achieve highly accurate single-image super-resolution"
authors: [Jiwon Kim, Jung Kwon Lee, Kyoung Mu Lee]
tags: []
categories: []
date: 2020-06-26T10:37:12-07:00
lastmod: 2020-06-26T10:37:12-07:00
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

Find the paper [here](https://arxiv.org/abs/1511.04587)

## At a Glance

- The paper proposes an effective model for single image super resolution that is highly accurate.
- Increasing the model depth increases overall accuracy.
- Contextual information over large regions is built up by cascading multiple smaller filters.
- Convergence speed is maximized by learning only residuals, and using large learning rates with adjustable gradient clipping.
- May be usefull in denoising and compression artifact removal

## Introduction

The goal of the paper is to introduce a single image super resolution (SISR) model that addresses some of the limitations of a previously proposed framework, the SRCNN.

The advantages of using CNNs for super resolution is that they provide an effective end-to-end solution, whereas past work required hand-engineered features.

The paper lists three limitations of SRCNNs and how VDSR can address these:
- *SRCNN is context dependent in small images* - Information in a small patch does not hold enough information for detail recovery. VDSR addresses this by cascading small filters to capture large region information.

- *Training for deep CNNs is slow* - VDSR addresses this by only learning *residuals* - the difference between the Low Resolution (LR) and High Resolution (HR) images. This works because the LR and HR images share the same information to a very large extent. Additionally, very large learning rates are used during training, with adjustable gradient clipping.

- *SRCNN only works for a single scale* - A single VDSR model is adequate for multi-scale-factor super resolution.

## Proposed Method

### Proposed Network

The network takes in an interpolated LR (ILR) image of shape $w \times h \times 3$ and predicts the residual image ($ w \times h \times 1$) which is then added onto the ILR to yield the HR image ($w \times h \times 3$).

The network is comprised of $L$ layers where all but $l=1,20$ (first and last) follow ZEROPAD -> CONV($3\times 3, 64 \text{ filters}$) -> RELU. The first layer operates on the input and the last layer consists of ZEROPAD -> CONV($3\times 3, 1 \text{ filter}$) to output the desired residual image.

The purpose of zero-padding before each convolution is to preserve the size of the feature maps. One issue with deep CNNs is that the convolution operation reduces the size of the feature map. Pixels on the border cannot be inferred properly, so usually SISR methods crop the boundary out which is fine for shallow models, but for deep CNNs it is unfeasible. Zero-padding addresses this issue, and is reported to work well.

$L$ is specified to be $20$ in the paper's training description.

### Training

The Loss function was the mean squared error averaged over the training set: $\frac{1}{2}  || \pmb{y} - \pmb{\hat{y}}||^2$, where $\pmb{y}$ is the HR image corresponding to the input LR image, and $\pmb{\hat{y}}$ is the model predicted HR image.

#### Residual Learning

The residual image is defined as $\pmb{r}=\pmb{y}-\pmb{x}$. Most values are likely to be small or zero, which is desirable when training. Since we want the network to predict the residual $\pmb{r}$, the loss function can be rewritten as $\frac{1}{2}  || \pmb{r} - \pmb{\hat{y}}||^2$. However, in the actual network training, the loss is the $L_2$ norm betweeen the reconstructed image $\pmb{r}+\pmb{x}$ and the ground truth $\pmb{y}$.

Mini-batch Gradient Descent was used with a momentum optimizer (I assume, as the paper references momentum $\beta = 0.9$, could also be the Adam optimizer) and a weight decay of $0.0001$ (weight decay means adding a regularizing term to the loss, $\mathcal{L} = \frac{1}{2}  || \pmb{y} - \pmb{\hat{y}}||^2 + \gamma L_2, \gamma=0.0001$)

#### Adjustable Gradient Clipping

An issue when training deep CNNs is the slow speed of convergence. One tactic to speed up training is to increase the learning rate $\alpha$, however this can lead to exploding gradients.

One solution to this is referred to as Gradient Clipping where the gradients of the parameters with respect to the loss function are clipped between a certain range $[-\theta, \theta]$. The issue with this approach is that, at the start of training when the learning rate is very high, $\theta$ must be very small to prevent exploding gradients, however as the network is trained, learning rate is annealed and as such $\alpha \frac{\partial{\mathcal{L}}}{\partial{W}}$ gets increasingly smaller.

The suggested method is to set gradients between $[-\frac{\theta}{\alpha}, \frac{\theta}{\alpha}]$, so the clipping is adjusted based on the current learning rate.

#### Multi-Scale

The model can be adapted to handle mutliple scales by simply training it on data of varying scales.
Images are divided into sub-images without overlap where sub-images from different scales are present.

The paper tests the performance of a model trained with $s_{train}=\\{2\\}$ (scale factor of 2 in the training set) on different input scales and sees that for $s_{train} \ne s_{test}$, performance is bad. However when $s_{train}=\\{2,3,4\\}$ the performance at each scale factor is comparable with a corresponding single-scale network, even outperforming single-scale models at large scales (3,4).

### Results

VDSR outperforms Bicubic, A+, RFL, SelfEx, and SRCNN (all methods listed) in every regard (PSNR/SSIM/time).

Benchmarks were made on Set5, Set14, B100 and Urban100 datasets.