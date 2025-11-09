---
permalink: /
title: ""
author_profile: true
redirect_from:
  - /about/
  - /about.html
---
 
I’m a Research Fellow at the [Flatiron Institute](https://www.simonsfoundation.org/flatiron/), Simons Foundation, working jointly at the Center for Computational Neuroscience and the Center for Computational Mathematics. I did a Ph.D. in Data Science at the [Center for Data Science](https://cds.nyu.edu/) at New York University, advised by [Eero Simoncelli](https://www.cns.nyu.edu/~eero/). See my [thesis](https://www.cns.nyu.edu/pub/lcv/kadkhodaie-phd.pdf) here. I studied Solid State Physics for my bachelor’s and Psychology for my master’s.

I'm broadly interested in vision and more specifically in **probability densities of natural images**. 
I have studied these densities from various angles: ***learning*** them from data, ***understanding and evaluating*** the learned models, and ***utilizing*** them for real-world problems. These areas are closely intertwined: understanding a learned model can inspire the design of better and more efficient ones. Conversely, practical advances can hint at something meaningful the model has captured about the underlying data structures.
I enjoy studying these complementary perspectives and seeing how they inform one another through careful and controlled **scientific** experimentation. 


<!-- when engineeing creativity leads to improved performance, it often hints at something meaningful the model has captured about the "true" natural image density.  -->
<!-- reveal new insights into the structure of natural images. -->

# Research 
<!-- ------------------------------------------------- -->
<!-- ------------------------------------------------- -->
## Learning image density models from data: 
<!-- ------------------------------------------------- -->
<!-- ------------------------------------------------- -->

### <span style="color:blue"> Learning and sampling from the density implicit in a denoiser </span>

Before deep learning, one of the major approches to solve Gaussian denoising problem (as well as other inverse problems) was to assume a prior over the space of images (e.g. Gaussian, Union of suspaces, Markov random fields) and then to estimate a solution in a Bayesian framework. The denoiser performance depended on how well this prior approximated the "true" images density. Designing image priors, however, is not trivial and progress relied on empirical findings about image structures -- like spectral, sparsity, locality -- which led to a steady but slow improvments. 


Deep learning revolution upended this trend. We gained access to computrational tools to learn, with unprecedented success, complex high-dimensional mappings for tasks such as denoising, segmentation, classification, etc. without assuming a prior. Yet this phenomenal performance raises a question: *what is the **prior** that the learned mapping impliciltly relies on?* 


Remarkably, in the case of Gaussian denoising, the relationship between the denoising mapping and the prior is exact and explicit, thanks to a classical statistics result [Robin 1956, Miyasawa 1961]: 


$$ \hat{x}(y) = y + \sigma^2 \nabla_y \log p (y)$$

See Raphan for proof. 

A Deep Neural Network (DNN) denoiser, $$\hat{x}_{\theta}(y)$$, hence, computes the score (gradient of the log probablity) of noisy images, $$y$$. When the DNN denoiser learns to solve the problem at all nosie levels, it could be used in an iterative **coarse-to-fine gradient ascent algorithm**  to sample from the density embedded in the denoiser. We introduced this algorithm in the paper below. Its core idea parallels to what later became known as **diffusion models**.  


<p align="center">
  <img src="https://Zahra-Kadkhodaie.github.io/images/manifold_diffusion.png" alt="Project schematic" width="70%">
    Here is some text describing the project that wraps around the image.
</p>
<!-- <img src="/images/project_photo.jpg" alt="project image" width="300" align="left" style="margin-right:15px;"> -->

A key property of our algorithm is that the denoiser is noise-level-blind -- it does not take as input $$\sigma$$. This allows an **adaptive** noise schedule during sampling, where the step size depends on the noise amplitute estimated by the model. Additionally, the injected noise at each iteration can be tuned to steer the sampling trajectory toward lower- or higher-probability regions of the distribution, with guaranteed convergence.


**Reference:** <br>
ZK & Simoncelli, Solving linear inverse problems using the prior implicit in a denoiser. arXiv, 2020.  [PDF](https://arxiv.org/pdf/2007.13640) | [Project page](https://github.com/LabForComputationalVision/universal_inverse_problem)<br>
Later published as: ZK & Simoncelli, Stochastic Solutions for Linear Inverse Problems using the Prior Implicit in a Denoiser. NeurIPS, 2021. [PDF](https://proceedings.neurips.cc/paper/2021/hash/6e28943943dbed3c7f82fc05f269947a-Abstract.html)  <br>  



<!-- ------------------------------------------------- -->

### <span style="color:blue"> Learning normalized image density rather than the score </span>
Can we make the embeded prior in a denoiser even more explict by predicting the energy ($$- \log p$$) rather than the score ($$ \nabla \log p$$)? There are two main problems to tackle to make this transition: 1) finding the right architecture and second and 2) normalizing the density. Neither of these problems exit for score models: through a massive collective effort the architecures have been imporved and refined to have the right inductive biases. This evolution has not happened for energy models, putting them at a considerable disadvange. Additionally, in score models, thanks to the gradient, the normalizing factor (partition function) goes away resolving the problem automatically. In the paper below, we introduced two simple tricks to overcome these issues. 

First, we showed that we can re-purpose score model architetures for energy models, by setting the energy to be 

$$U_{\theta}(y, t) =  \frac{1}{2} \langle y , s_{\theta}(y,t) \rangle$$

for this to be true, the score  model is required to be conservative and homogeneous.

Second, to get the normalization right (up to a global constant), we add a regularization term to the loss function that gaurantees the diffusion equation holds across time (noise levels). 

$$
    \ell_{\rm TSM}(\theta,t) = \expect[x,y]{\paren{\partial_t \energy(y,t) - \frac{d}{2t} + \frac{\norm{y-x}^2}{2t^2}}^2} .
$$

In effect, it ties together the normalization constants of indivisual $$p(y,t)$$ such that the normalization factor is not a function of time anymore. Since the diffused density models are tied together, after training, we can compute the normalization factor of $$p(y,t=0)$$ by analytcically computing it for $$p(y,t=\infty)$$ (Standard Gaussian) and transferring that to $$t=0$$. 


(Add fig, connected path )

These two changes do not deteriorate denoisnig performance, meaning that the optimum for the dual loss happens at the same place as the single loss. 


A model trained using these two tricks compute $$\log p(x)$$  in one forward pass (**1000 times** faster than cumbersome computation using a score model). 

A good energy model assigns low energy to in distribution images. We test this on a model trained on ImageNet and show that $$-\log p(x)$$ are within the state-of-the-art range. 

(add table NLL)
 
**Reference:**  <br>
Guth, ZK & Simoncelli, Learning normalized image densities via dual score matching. NeurIPS, 2025  [PDF](https://arxiv.org/pdf/2506.05310) <br>


<!-- ------------------------------------------------- -->
<!-- ------------------------------------------------- -->
## Understanding and Evaluating learned density models: 
<!-- ------------------------------------------------- -->
<!-- ------------------------------------------------- -->

scientifc method shines: deep nets have evolves through a natural selection, we can examine them by hypothesizing about what they are and how they work, design controlled experiments to test them. 
shallow understanding of deep models.
curse of dimensionality
what is a good model? 
why do we care about understanding? predict when generalization and when fails


<!-- ------------------------------------------------- -->

### <span style="color:blue">  Denoising is a soft projection on an adaptive basis (tangent plane of a blurred manifold) </span>

Classical denoisers: find a space where image is compact, shrink, go back. Examples: Fourier (would be perfect if the world was gaussian but it is not). Wavelet (prior is union of subspaces), markov random fields (GSM). 
Of course there were models that did not rely on priors: tour of modern denoising - filtering, BM2D, non local means 
with deep net denoiser: they solve the denoisng problem by non-linear regression (like every other problem). can we deepen our shallow understading of deep nets? How do they work? What is the transformed space they operate in? 
We made one change to make the network more analyzable: removing addetive constant (call bais in pytorch implementation) from the model to make it locally linear. Now we can use linear algebra! 
Jacobian: symmetric 
intrepret rows: filtering: old lit. point is the increasing size of weighted averging. Model figures out the noise size (size of neighborhood) and is adaptive to the content 
interpret columns: basis in which a noisy image is being denoised 

Both filtering and basis are noise level dependent. This can be formulated in a noise-dependent effective dimesionality 


<!-- ------------------------------------------------- -->

### <span style="color:blue"> Conditional locality of image densities (click to expand)</span>
Learning multi-scale local conditional probability models of images: conditional locality

   How do denoisers embed densities despite the curse of dimensionality?
   Factorizing to lower dimensional densities. (old trick, markov random fields)
   first learn global coarse stuff (big features). Don't need all the details -> down sample. zero padding -> breaking translation invariance.
   then you can model details with local density models, conditioned on gloval stuff.
   give an example: if you have a blurred outline of the face and locations of the details, then you can refine it
   How small can you make the RF or neighborhood in the MRF model? We showed you can go pretty small.
   Does this hold in more complex cases? Linear encoding can only takes us so far, 


<!-- ------------------------------------------------- -->

### <span style="color:blue"> Generalization in diffusion models (click to expand)</span>
generalization paper:
  strong generalization
   GAHBs


<!-- ------------------------------------------------- -->

### <span style="color:blue"> unsupervised representation learning via denoising (click to expand)</span>
representation
   open the black box. What representation arises from learning the score.
   spatial average of channels in the deepest layer: sparse and selective (union of subspaces)


###    <span style="color:blue"> Unbelievably vast range of natural image probabilities </span>

energy model: energy distribution of images    


<!-- ------------------------------------------------- -->
<!-- ------------------------------------------------- -->
## Utilizing learned density models to solve inverse problems: 
<!-- ------------------------------------------------- -->
<!-- ------------------------------------------------- -->

Ultimately, we want to learn the density to use it! Inverse problems in signal processing (a particular approach: it is stochastic) 

### Linear inverse problems 
<!-- ------------------------------------------------- -->

### <span style="color:blue"> Solving inverse problem(click to expand)</span>
<!-- ------------------------------------------------- -->

### <span style="color:blue"> optimal measurement (click to expand)</span>
<!-- ------------------------------------------------- -->


### <span style="color:blue"> cone excitation (click to expand)</span>
<!-- ------------------------------------------------- -->


### non-linear inverse problems
- feature- guided 
- texture model 





