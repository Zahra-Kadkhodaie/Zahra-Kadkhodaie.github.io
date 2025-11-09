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
#  <span style="color:#A52A2A"> Learning image density models from data </span>
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
 
**Reference:**  <br>
Guth, ZK & Simoncelli, Learning normalized image densities via dual score matching. NeurIPS, 2025  [PDF](https://arxiv.org/pdf/2506.05310) <br>


<!-- ------------------------------------------------- -->
<!-- ------------------------------------------------- -->
# <span style="color:#A52A2A"> Understanding and Evaluating learned density models </span>
<!-- ------------------------------------------------- -->
<!-- ------------------------------------------------- -->
Deep neural networks are many times more capable than their classical predecessors. They have grown increasingly complex and deep, while our understanding of them has remained comparatively shallow.
Why should we try to understand them? Beyond the intrinsic satisfaction of figuring things out, a deeper understanding is essential for evaluating these learned models.

In the context of density learning, assessing how “good” a model really is depends on two questions:

How well does it generalize?

How accurately does it approximate the true density?

Answering these requires knowing where and how such models fail—insight that, in turn, comes from studying why they succeed where they do.

I approach these questions through scientific experimentation: explore the data, form hypotheses, and test them under controlled conditions. I believe this mindset suits modern models well. After all, they have evolved through an accelerated process of “natural selection”—only the most effective architectures have survived—making today’s networks far too complex to be fully understood through a purely reductionist, bottom-up theoretical approach. 

<!-- ------------------------------------------------- -->

## Denoising is a soft projection on an adaptive basis  
(tangent plane of a blurred manifold)
Classical denoisers: find a space where image is compact, shrink, go back. Examples: Fourier (would be perfect if the world was gaussian but it is not). Wavelet (prior is union of subspaces), markov random fields (GSM). 
Of course there were models that did not rely on priors: tour of modern denoising - filtering, BM2D, non local means 
with deep net denoiser: they solve the denoisng problem by non-linear regression (like every other problem). can we deepen our shallow understading of deep nets? How do they work? What is the transformed space they operate in? 
We made one change to make the network more analyzable: removing addetive constant (call bais in pytorch implementation) from the model to make it locally linear. Now we can use linear algebra! 
Jacobian: symmetric 
intrepret rows: filtering: old lit. point is the increasing size of weighted averging. Model figures out the noise size (size of neighborhood) and is adaptive to the content 
interpret columns: basis in which a noisy image is being denoised 

Both filtering and basis are noise level dependent. This can be formulated in a noise-dependent effective dimesionality 


<!-- ------------------------------------------------- -->

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

## Generalization in diffusion models
generalization paper:
  strong generalization
   GAHBs


<!-- ------------------------------------------------- -->

## unsupervised representation learning via denoising
representation
   open the black box. What representation arises from learning the score.
   spatial average of channels in the deepest layer: sparse and selective (union of subspaces)


## Unbelievably vast range of natural image probabilities 

energy model: energy distribution of images    


<!-- ------------------------------------------------- -->
<!-- ------------------------------------------------- -->
# <span style="color:#A52A2A"> Utilizing learned density models to solve inverse problems </span>
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





