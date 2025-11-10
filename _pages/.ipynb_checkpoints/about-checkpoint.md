---
permalink: /
title: ""
author_profile: true
redirect_from:
  - /about/
  - /about.html
---
 
I’m a Research Fellow at the Flatiron Institute, Simons Foundation, working jointly at the Center for Computational Neuroscience and the Center for Computational Mathematics. I did a Ph.D. in Data Science at the Center for Data Scienceat New York University, advised by [Eero Simoncelli](https://www.cns.nyu.edu/~eero/). Here is my [thesis](https://www.cns.nyu.edu/pub/lcv/kadkhodaie-phd.pdf). I studied Solid State Physics for my bachelor’s and Psychology for my master’s.

I'm broadly interested in vision and more specifically in **probability densities of natural images**. 
I have studied these densities from various angles: ***learning*** them from data, ***understanding and evaluating*** the learned models, and ***utilizing*** them for real-world problems. These areas are closely intertwined: understanding a learned model can inspire the design of better and more efficient ones. Conversely, better performance can hint at something meaningful the model has captured about the underlying data structures.
I enjoy studying these complementary perspectives and seeing how they inform one another through careful and controlled **scientific** experimentation. 


<!-- when engineeing creativity leads to improved performance, it often hints at something meaningful the model has captured about the "true" natural image density.  -->
<!-- reveal new insights into the structure of natural images. -->

 
<!-- ------------------------------------------------- -->
<!-- ------------------------------------------------- -->
#  <span style="color:#A52A2A"> Learning Image Density Models from Data </span>
<!-- ------------------------------------------------- -->
<!-- ------------------------------------------------- -->

## <span style="color:#008000"> Learning and sampling from a density implicit in a denoiser </span>
<!-- ## Learning and sampling from the density implicit in a denoiser -->

Before deep learning, one of the major approches to solve Gaussian denoising problem (as well as other inverse problems) was to assume a prior over the space of images (e.g. Gaussian, Union of subspaces, Markov random fields) and then estimate a solution in a Bayesian framework. The denoiser performance depended on how well this prior approximated the "true" images density. Designing image priors, however, is not trivial and progress relied on empirical findings about image structures -- like spectral, sparsity, locality -- which led to a steady but slow improvments. 


Deep learning revolution upended this trend. We gained access to computrational tools to learn, with unprecedented success, complex high-dimensional mappings for tasks such as denoising, segmentation, classification, etc. without assuming a prior. Yet this phenomenal performance raises a question: *what is the **prior** that the learned mapping impliciltly relies on?* 


Remarkably, in the case of Gaussian denoising, the relationship between the denoising mapping and the prior is exact and explicit, thanks to a classical statistics result [Robin 1956, Miyasawa 1961]: 

$$ \hat{x}(y) = y + \sigma^2 \nabla_y \log p (y)$$

See [Raphan 2011](https://www.cns.nyu.edu/pub/eero/raphan10.pdf) for proof.

A Deep Neural Network (DNN) denoiser, $$\hat{x}_{\theta}(y)$$, hence, computes the score (gradient of the log probablity) of noisy images, $$y$$. When the DNN denoiser learns to solve the problem at all nosie levels, it could be used in an iterative **coarse-to-fine gradient ascent algorithm**  to sample from the density embedded in the denoiser. We introduced this algorithm in the paper below. Its core idea is similar and concurrent to what became known as **diffusion models**.  


<p align="center" markdown="1">
<img src="https://zahra-kadkhodaie.github.io/images/manifold_diffusion2.png" alt="Project schematic" width="90%"><br>
  <span style="font-size: 0.80em; color: #555;">
    A two-dimensional simulation of the sampler. Right panel shows trajectory of our iterative coarse-to-fine sampling algorithm, starting from the same initial values y (red points) of the first panel. The trajectories are curved, and always arrive at solutions on the signal manifold.
  </span>
</p>

<p align="center" markdown="1">
<img src="https://zahra-kadkhodaie.github.io/images/trajectory_mnist.png" alt="Project schematic" width="70%"><br>
<!-- <img src="https://zahra-kadkhodaie.github.io/images/trajectory_patches.png" alt="Project schematic" width="60%"><br> -->
  <span style="font-size: 0.80em; color: #555;">
    Example sampling trajectory for a model trained on MNIST images.
  </span>
</p>
  
<!-- <p align="center">
  <iframe width="20" height="10"
          src="https://www.youtube.com/embed/wfOq7kAc3Z8"
          title="YouTube video player"
          frameborder="100"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowfullscreen>
  </iframe>
</p> -->


A key property of our algorithm is that the denoiser is noise-level-blind -- it does not take as input $$\sigma$$. This allows an **adaptive** noise schedule during sampling, where the step size depends on the noise amplitute estimated by the model. 
Additionally, the injected noise at each iteration can be tuned to steer the sampling trajectory toward lower- or higher-probability regions of the distribution, with guaranteed convergence.


<p align="center" markdown="1">
<img src="https://zahra-kadkhodaie.github.io/images/convergence.png" alt="Project schematic" width="70%"><br>
<span style="font-size: 0.80em; color: #555;">
Left: Noise level of the sample as a function of iteration in synthesis, shown for 3 levels of injected noise. More noise (smaller beta) results in more steps. Right: Adaptive algorithm results in varying number of steps for different samples. 
</span>
</p>

<p align="center" markdown="1">
<img src="https://zahra-kadkhodaie.github.io/images/beta_effect.png" alt="Project schematic" width="50%"><br>
      <span style="font-size: 0.80em; color: #555;">
More noise during synthesis results in higher probability images (right panel) by escaping smaller maxima (model trained on patches of grayscale images). 
  </span>
</p>
**Reference:** <br>
ZK & Simoncelli, Solving linear inverse problems using the prior implicit in a denoiser. arXiv, 2020.  [PDF](https://arxiv.org/pdf/2007.13640) | [Project page](https://github.com/LabForComputationalVision/universal_inverse_problem)<br>
Later published as: ZK & Simoncelli, Stochastic Solutions for Linear Inverse Problems using the Prior Implicit in a Denoiser. NeurIPS, 2021. [PDF](https://proceedings.neurips.cc/paper/2021/hash/6e28943943dbed3c7f82fc05f269947a-Abstract.html)  <br>  



<!-- ------------------------------------------------- -->

## <span style="color:#008000"> Learning normalized image density rather than the score </span>
<!-- ## Learning normalized image density rather than the score -->

Can the embeded prior in a denoiser be made more explict by predicting the energy ($$-\log p$$) rather than the score ($$ \nabla \log p$$)? 

There are two main problems to tackle to make this happen: 1) finding the right architecture and 2) normalizing the density. Neither of these problems exit for score models. Architecures have been refined, through a collective effort, to have the right inductive biases. This evolution has not happened for energy models, putting them at a considerable disadvange. Additionally, in score models, the normalizing factor is eliminated thanks to the gradient. In the paper below, we introduced two simple tricks to overcome these two issues and learn $$\log p$$ directly. 

First, we showed that score model architetures can be re-purposed for energy models, by setting the energy to be 

$$U_{\theta}(y, t) =  \frac{1}{2} \langle y , s_{\theta}(y,t) \rangle$$

<!-- for this to be true, the score  model is required to be conservative and homogeneous. -->

Second, to get the normalization factor right (up to a global constant), we add a regularization term to the loss function that gaurantees the diffusion equation hold across time (noise levels). 

$$
\ell_{\rm TSM}(\theta,t) = \mathbb{E}_{x,y} \left[{ \left( {\partial_t U_{\theta}(y,t) - \frac{d}{2t} + \frac{\Vert{y-x}\Vert^2} {2t^2}} \right)^2}\right]
$$

In effect, minimizing this term ties together the normalization factors of individual $$p(y,t)$$. Since the diffused density models are tied together, after training, we can set the normalization factor of $$p(y,t=0)$$ by analytcically computing it for $$p(y,t \to \infty)$$ (Standard Gaussian) and then transferring it to $$p(x)$$. 

<p align="center" markdown="1">
<img src="https://zahra-kadkhodaie.github.io/images/diffusing_barrier_decorated.png" alt="Project schematic" width="30%"><br>
      <span style="font-size: 0.80em; color: #555;">
  </span>
</p>

These two changes do not deteriorate denoisnig performance. This implies the minimizers of the two terms in the dual loss do not fight but reinfornce one another. A model trained using these two tricks returns $$\log p(x)$$ in only one forward pass: **1000 times** faster than cumbersome computation using a score model. 

A good energy model assigns low energy to in-distribution images. We test this on a model trained on ImageNet and show that $$-\log p(x)$$ are within the state-of-the-art range. 

<p align="center" markdown="1">
<img src="https://zahra-kadkhodaie.github.io/images/table2.png" alt="Project schematic" width="90%"><br>
      <span style="font-size: 0.80em; color: #555;">
  </span>
</p>

A nice consequence of having direct access to $\log p(x)$ is that we can now explore image probabilies and study how they relate to image structrues. The first surprising (even shocking!) observation is **the unbelievably vast range of natural image probabilities**. Unlike the common assumption about image distributions, images vary in their probability by a factor of $$10^{14,000}$$ (no concentration!). This implies that rare events in the space of images are not so rare when you think about probabilty mass: level sets of low probabilty images has to compensate for a low probability with the inverse of its volume. There are many more low probability images that high probability ones.  

Additionally, there is a **perceptual component** strongly tied to the probablity of an image: high probablity images contain more flat regions while low probability images contain lots of details and smaller features which makes them less denoisable. 


<p align="center" markdown="1">
<img src="https://zahra-kadkhodaie.github.io/images/gumble.png" alt="Project schematic" width="90%"><br>
      <span style="font-size: 0.80em; color: #555;">
          Histogram of log probabilities of images in the ImageNet dataset. Color-coded arrows
indicate values for the example images on the right. 
  </span>
</p>   


**Reference:**  <br>
Guth, ZK & Simoncelli, Learning normalized image densities via dual score matching. NeurIPS, 2025  [PDF](https://arxiv.org/pdf/2506.05310) <br>


<!-- ------------------------------------------------- -->
<!-- ------------------------------------------------- -->
# <span style="color:#A52A2A"> Understanding and Evaluating Learned Density Models </span>
<!-- ------------------------------------------------- -->
<!-- ------------------------------------------------- -->
<!--  -->
Deep neural networks have grown increasingly complex and deep, while our understanding of them remains comparatively shallow.
Why should we try to understand them? Aside from the intrinsic satisfaction of figuring things out, a deeper understanding is essential for **evaluating** these learned models. In the context of density learning, assessing how “good” a model really begs two questions: 1) *How well does it generalize?* 2) *How accurately does it approximate the true density?* Answering these requires knowing where and how such models fail—insight that, in turn, comes from studying why they succeed where they do. I approach these questions through **scientific experimentation**: explore the data, form hypotheses, and test them under controlled conditions. I believe this mindset suits modern models well. After all, they have evolved through an accelerated process of “natural selection”—only the most effective architectures have survived—making today’s networks far too complex to be fully understood through a purely reductionist, bottom-up theoretical approach. 

<!-- ------------------------------------------------- -->

## Generalization in diffusion models
generalization paper:
  strong generalization
  
<!-- ------------------------------------------------- -->

## <span style="color:#008000">  Denoising is a soft projection on an adaptive basis  </span>
Classical denoising heavily relied on designing transformations in which the image representation was **sparse**.
Many of these denoisers worked in three stages: 1) transform the noisy image where noise and image are separable, 2) apply a shrinkage function (**soft projection**) to suppress the noise, and 3) transform back to pixel space. To maximally preserve the image and remove noise, the image represention in the transformed space shoud be as sparse and compact as possible. But, due to computataional limitations, these transformations were often linear (e.g. Fourier, Wavelet), so failed to fully harvest the intrinsic low-dimensionality of images. Deep neural network denoisers are many times more capable than their classical predecessors. But how do they work? *What is the transformation they learn from data?*  

To analyze and understand how deep net denoisers work we drew on the insight from the classical literature. In the paper below, we showed that locally-linear DNN denosiers can be described as soft projection (shrinkage) in a sparse basis. What makes them so powerful is that the basis is adaptive to the underlying image, thanks to the nonlinearities of the mapping. The adaptive basis can be exposed by Singular Value Decompotion (SVD) of the Jacobian ($$A_y$$) of the denoising mapping w.r.t. the noisy input. The top singular vectors span the **signal subpace** which can be interpreted as the **tangent plane to the (blurred) image manifold at clean image point**.

$$
\hat{x}(y) = A_y y = USV^T y = \Sigma_{i =1} ^N s_i (V_i^T y) U_i . 
$$

<p align="center" markdown="1">
<img src="https://zahra-kadkhodaie.github.io/images/svd_1.png" alt="Project schematic" width="45%">
<img src="https://zahra-kadkhodaie.github.io/images/top_sing_vect.png" alt="Project schematic" width="45%"><br>    
      <span style="font-size: 0.80em; color: #555;">
          Left: Fast decay of singular values shows that the adaptive basis is very sparse for the input image. Middle: The histogram shows that the Jacobian is almost symmetric. So, the network implements a soft projection onto a basis adaptive to the input. Right: Top singular vectores capture image features which will be preserved and bottom singular vectors are noise which will be supressed. 
  </span>
</p>

Dimensionality of the subspace depends on the noise level on the input image. At higher noise levels, fewer signal dimensions can survive the noise. Empirically, dimensionality drops differently for different images, but on average it drops proportional to the inverse of noise level. (See paper for results that shows the subspaces at higher noise levels are nested within subsapces with lower noise levels). 

<p align="center" markdown="1">
<img src="https://zahra-kadkhodaie.github.io/images/effective_dim.png" alt="Project schematic" width="45%"><br>
      <span style="font-size: 0.80em; color: #555;"> 
  </span>
</p>

In addition to analysizing the column space of the Jacobian, we also analyzed its row space. Interestingly, we could interpet the DNN denoising mapping as an **adaptive filtering procedure in pixel domain**, which ties it to another way of formulating denoising in classical signal processing literture (see [this review paper](https://users.soe.ucsc.edu/~milanfar/publications/journal/ModernTour.pdf)). Here, a pixel is estimated by a weighted average of neighboring pixel. The **neighborhood weights** are daptive to both the noise level and the underlying image structure. 

<p align="center" markdown="1">
<img src="https://zahra-kadkhodaie.github.io/images/filtering.png" alt="Project schematic" width="80%"><br>
      <span style="font-size: 0.80em; color: #555;">
          
  </span>
</p>


**Reference:**  <br>
Mohan\*, ZK\*, Simoncelli & Fernandez-Granda, Robust And Interpretable Blind Image Denoising Via Bias-Free Convolutional Neural Networks. ICLR, 2020. <br>
  [PDF](https://openreview.net/pdf?id=HJlSmC4FPS) | [Project page](https://labforcomputationalvision.github.io/bias_free_denoising/) | [Code](https://github.com/LabForComputationalVision/bias_free_denoising) <br>
<sub>\* denotes equal contribution</sub>

<!-- ------------------------------------------------- -->

## GAHBs

## Conditional locality of image densities
Learning multi-scale local conditional probability models of images: conditional locality

   How do denoisers embed densities despite the curse of dimensionality?
   Factorizing to lower dimensional densities. (old trick, markov random fields)
   first learn global coarse stuff (big features). Don't need all the details -> down sample. zero padding -> breaking translation invariance.
   then you can model details with local density models, conditioned on gloval stuff.
   give an example: if you have a blurred outline of the face and locations of the details, then you can refine it
   How small can you make the RF or neighborhood in the MRF model? We showed you can go pretty small.
   Does this hold in more complex cases? Linear encoding can only takes us so far, 


<!-- ------------------------------------------------- -->

## unsupervised representation learning via denoising
representation
   open the black box. What representation arises from learning the score.
   spatial average of channels in the deepest layer: sparse and selective (union of subspaces)




<!-- ------------------------------------------------- -->
<!-- ------------------------------------------------- -->
# <span style="color:#A52A2A"> Utilizing Learned Density Models to Solve Inverse Problems </span>
<!-- ------------------------------------------------- -->
<!-- ------------------------------------------------- -->

Ultimately, we want to learn the density to use it! Inverse problems in signal processing (a particular approach: it is stochastic) 

## Linear inverse problems 
<!-- ------------------------------------------------- -->

## Solving inverse problem(click to expand)
<!-- ------------------------------------------------- -->

## optimal measurement (click to expand)
<!-- ------------------------------------------------- -->


##  cone excitation (click to expand)
<!-- ------------------------------------------------- -->


## non-linear inverse problems
- feature- guided 
- texture model 





