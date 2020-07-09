---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "A Neural Algorithm of Artistic Style"
subtitle: ""
summary: "Creating artistic images using Deep Neural Networks"
authors: [Leon A. Gatys, Alexander S. Ecker, Matthias Bethge]
tags: []
categories: []
date: 2020-07-02T22:51:27-07:00
lastmod: 2020-07-02T22:51:27-07:00
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

Find the paper [here](https://arxiv.org/abs/1508.06576)

Convolutional Neural Networks (CNN) are powerful in image processing related tasks. They consist of layers of small computational units that process visual information in a heirachical fashion. Each layer is essentially a collection of image filters that extract a certain feature from the input image.

When trained on tasks such as image detection, CNNs develop image representations that are increasingly explicit along the processing path. In other words, the input image is successively transformed into representations that increasingly care about content of the image instead of local pixel values.

The paper refers to feature respnses in higher layers of the network as the content representation.

To obtain a meaningful representation of the style of an input image, the authors utilize a texture feature space. Essentially, by taking the filter responses in each layer of the networks and taking correlations between them over the channels in the feature maps, an objective method of texture evaluation is constructed.

Reconstructions from style features produce versions of the input that are texturized and capture its general appearance in color and local structure. The size and complexity increases along the hierarchy, due to increasing feature complexity and receptive field size. This representation is denoted as style representation.

The key finding in the paper is that content and style representations in CNNs are separable.

Novel images can be synthesized by finding an image that can simultaneously match the content representation of an input photo as well as the style representation of a respective piece of art.

The most visually appealing images are created by matching the style representation including up to the highest layers in the network, whereas style can be defined more locally by only including a lower number of layers.

When synthesizing style and content, both must be included. As such, the loss function that is minimized includes components for both content and style with hyperparameters that allow to tweak emphasis on either.

The authors derived these content and style representations from feature responses of DNNs trained on object recognition.

Rendering a photo in the style of a certain artwork is approached by the computer vision field of non-photorealistic rendering. These methods relied on non-parametric techniques that manipulate the pixel representations directly. DNNs, on the other hand, manipulate feature spaces that represent style and content.

## Methods

The paper's results were based on the VGG-Network, VGG19 I believe.

The authors disregard the fully connected layers, and only focus on the 16 CONV and 5 POOL layers to build the feature space.

Instead of Max Pooling, the authors found that Average Pooling worked better - it improves gradient flow and the results become more visually appealing.

Let the input image be denoted as $\vec{x}$.

Each layer of the VGG Network beign considered has $N_l$ filters of size $M_l = W_l \times H_l$.

Responses of layer $l$ are stored in $F^{l}$, where $F^{l} \in \mathcal{R}^{N_l \times M_l}$. We index $F$ as $F^{l}\_{ij}$ which is the activation of filter $i$ at position $j$ in $M^l$ of layer $l$.

Gradient Descent is performed on a white noise image to find one that matches the feature responses of the original image.

Given original image, $\vec{p}$, and generated image, $\vec{x}$,  with feature representations at layer $l$ given by $P^{l}$ and $F^{l}$ respectively, the content loss is the squared-error loss between the feature representations:

$$\mathcal{L}\_{content}(\vec{p},\vec{x},l) = \frac{1}{2}\sum_{ij}(F^{l}\_{ij} - P^{l}\_{ij})^2$$

So, the derivative of the loss with respect to the activations is:

$$\frac{\partial \mathcal{L}\_{content}}{\partial  F^{l}\_{ij}} = \left\\{
                \begin{array}{ll}
                  (F^l-P^l)\_{ij} & \text{if} F^{l}\_{ij} > 0\\\\
                  0 & \text{if} F^{l}\_{ij} < 0
                \end{array}
              \right.$$ 

Then, we change $\vec{x}$ untill it generates the same response in a certain layer as $\vec{p}$.

The style representation is built by taking the correlations between different filter responses. These correlations are given by Gram matrix $G^{l} \in \mathcal{R}^{N_l \times N_l}$. Here, $G^{l}\_{ij}$ is the inner product between feature maps $i,j$ in layer $l$:

$$G^{l}\_{ij} = \sum_{k} F^{l}\_{ik}F^{l}\_{jk}$$

Once more, we use Gradient Descent from a white noise image to find another image that matches the style representation. Like the content representation, we minimize the mean-squared distance between entries of the Gram matrix of the original image and the Gram matrix of the generated image.

Given original image, $\vec{a}$, and generated image, $\vec{x}$, with style representations at layer $l$ given by $A^{l}$ and $G^{l}$ respectively, the loss from a single layer $l$ is:

$$E\_l = \frac{1}{4N^{2}\_{l}M^{2}\_{l}} \sum_{ij}(G^{l}\_{ij}-A^{l}\_{ij})^2$$

With total loss:

$$\mathcal{L}\_{style}(\vec{a},\vec{x}) = \sum^{L}\_{l=0} w_lE_l$$

with $w_l$ weighting factors of the contribution of each layer. The derivatice of $E_l$ with respect to activations in layer $l$ becomes:

$$\frac{\partial E_l}{\partial  F^{l}\_{ij}} = \left\\{
                \begin{array}{ll}
                  \frac{1}{N_l^2 M_l^2}((F^l)^T (G^l-A^l))\_{ji} & \text{if} F^{l}\_{ij} > 0\\\\
                  0 & \text{if} F^{l}\_{ij} < 0
                \end{array}
              \right.$$ 
To generate images that mix the content of a photo with the style of a painting, we need to minimize the distance of a white noise image to the content representation of the photo in a layer as well as the style representtion of the paining over a number of layers. Given photograph $\vec{p}$ and artwork $\vec{a}$, the loss function becomes:

$$\mathcal{L}\_{total}(\vec{p}, \vec{a}, \vec{x}) =  \alpha \mathcal{L}\_{content}(\vec{p}, \vec{x}) + \beta \mathcal{L}\_{style}(\vec{a},\vec{x})$$

With $\alpha, \beta$ weighting factors to control content and style reconstruction respectively.

In the paper, content representation was matched for layer `conv4_2` and the style representation used layers `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, and `conv5_1`, with $w_l = \frac{1}{5}$ in those layers and $w_l=0$ in all others. In general, $w_l = \frac{1}{N_a}$ with $N_a$ being the number of active layers.

Additionally, the paper uses $\frac{\alpha}{\beta} \in \\{1\times 10^{-3},1 \times 10^{-4}\\}$