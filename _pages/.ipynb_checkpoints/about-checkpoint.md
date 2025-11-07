---
permalink: /
title: ""
author_profile: true
redirect_from:
  - /about/
  - /about.html
---
 
I’m a Research Fellow at the [Flatiron Institute](https://www.simonsfoundation.org/flatiron/), Simons Foundation, working jointly at the Center for Computational Neuroscience and the Center for Computational Mathematics. I did a Ph.D. in Data Science at the [Center for Data Science](https://cds.nyu.edu/) at New York University, advised by [Eero Simoncelli](https://www.cns.nyu.edu/~eero/). I studied Solid State Physics for my bachelor’s and Psychology for my master’s.

I'm broadly interested in vision and more specifically in **probability densities of natural images**. 
I have studied these densities from various angles: ***learning*** them from data, ***understanding and evaluating*** the learned models, and ***utilizing*** them for real-world problems. These areas are closely intertwined: understanding a learned model can inspire the design of better and more efficient ones. Conversely, practical advances can hint at something meaningful the model has captured about the underlying data structures.
I enjoy studying these complementary perspectives and seeing how they inform one another through careful and controlled **scientific** experimentation. 


<!-- when engineeing creativity leads to improved performance, it often hints at something meaningful the model has captured about the "true" natural image density.  -->
<!-- reveal new insights into the structure of natural images. -->

<!-- I’m broadly interested in vision, and in particular in probability densities of natural images.
My work explores different aspects of these densities — learning them from data, analyzing and evaluating the resulting models, and applying them to real-world problems. These directions are closely connected: gaining insight into a learned model can suggest ways to design better ones, while practical improvements often reveal something about what the model has captured from the underlying data. I enjoy studying these ideas from multiple perspectives and seeing how they inform one another.
 -->


# Research 
## Learning image density models from data: 




<details>
  <summary><strong> <span style="color:blue"> Deep neural network denoisers have implicit image densities in them (aka Score-based reverse diffusion models).  (click to expand)</span> </strong></summary>

in classical lit: design denoiser based on density. assume a density. design a basis in which signal and noise are separable (signal is compact, sparse and noise is dense). shrinkage
- here: learning! can't a density directly, but can learn  denoiser (a non-linear regression problem). learn the good denoiser and then extract the density it is relying on (embedding).
- How: tweedie: relationship between grad of log p and the denoising function.
- all noise levels, coarse to fine sampling - say diffusion - concurrent 
- blind denosiers (don't need to feed time as an argument)
- adjustable injected noise: high p vs low p images
- adaptive time schedule, h*sigma: goes faster 
- cite papers

- Z Kadkhodaie, E. P. Simoncelli, **Stochastic Solutions for Linear Inverse Problems using the Prior Implicit in a Denoiser**. NeurIPS, 2021. <br>
  [PDF](https://proceedings.neurips.cc/paper/2021/hash/6e28943943dbed3c7f82fc05f269947a-Abstract.html) | [Project page](https://github.com/LabForComputationalVision/universal_inverse_problem)

- Z. Kadkhodaie & E. P. Simoncelli, **Solving linear inverse problems using the prior implicit in a denoiser**. arXiv, 2020.<br>
  [PDF](https://arxiv.org/pdf/2007.13640) | [Project page](https://github.com/LabForComputationalVision/universal_inverse_problem)


 Gaussian denoising is arguably the most simple and fundamental problem in image processing. In classical (i.e. pre-deep learning era) probabilistic signal processing, engineering denoisers relied on assuming a density model over the signal (images) and a transformation that takes the image 
to design a denoiser, one assumes a probability distribution over the signal (images), for example spectral Gaussian, and then finds a represenation in which the signal is 
 

</details>

<!-- ------------------------------------------------- -->

<details>
  <summary><strong> <span style="color:blue"> How to learn a model to predict energy directly? (Rather than the score) (click to expand)</span> </strong></summary>
  
2) Learning normalized image densities via dual score matching: learn the log p directly.
   two tricks: keep the architecture (inductive biases already tested and evolved) and add the second loss (ties together the normalization factor (partition function) across the noise levels/times/ trajectory). Gets us within the state of the art NLL 
 
</details>

## Understanding and Evaluating learned density models: 
scientifc method shines: deep nets have evolves through a natural selection, we can examine them by hypothesizing about what they are and how they work, design controlled experiments to test them. 
shallow understanding of deep models.
curse of dimensionality
what is a good model? 
why do we care about understanding? predict when generalization and when fails





<details>
  <summary><strong> <span style="color:blue"> Denoising is a soft projection on an adaptive basis (tangent plane of a blurred manifold)  (click to expand)</span> </strong></summary>

1) Classical denoisers: find a space where image is compact, shrink, go back. Examples: Fourier (would be perfect if the world was gaussian but it is not). Wavelet (prior is union of subspaces), markov random fields (GSM). 
Of course there were models that did not rely on priors: tour of modern denoising - filtering, BM2D, non local means 
with deep net denoiser: they solve the denoisng problem by non-linear regression (like every other problem). can we deepen our shallow understading of deep nets? How do they work? What is the transformed space they operate in? 
We made one change to make the network more analyzable: removing addetive constant (call bais in pytorch implementation) from the model to make it locally linear. Now we can use linear algebra! 
Jacobian: symmetric 
intrepret rows: filtering: old lit. point is the increasing size of weighted averging. Model figures out the noise size (size of neighborhood) and is adaptive to the content 
interpret columns: basis in which a noisy image is being denoised 

Both filtering and basis are noise level dependent. This can be formulated in a noise-dependent effective dimesionality 
</details>


<details>
  <summary><strong> <span style="color:blue"> Conditional locality of image densities (click to expand)</span> </strong></summary>
Learning multi-scale local conditional probability models of images: conditional locality

   How do denoisers embed densities despite the curse of dimensionality?
   Factorizing to lower dimensional densities. (old trick, markov random fields)
   first learn global coarse stuff (big features). Don't need all the details -> down sample. zero padding -> breaking translation invariance.
   then you can model details with local density models, conditioned on gloval stuff.
   give an example: if you have a blurred outline of the face and locations of the details, then you can refine it
   How small can you make the RF or neighborhood in the MRF model? We showed you can go pretty small.
   Does this hold in more complex cases? Linear encoding can only takes us so far, 
</details>


<details>
  <summary><strong> <span style="color:blue"> Generalization in diffusion models (click to expand)</span> </strong></summary>

3) generalization paper:
  strong generalization
   GAHBs
</details>


<details>
  <summary><strong> <span style="color:blue"> unsupervised representation learning via denoising (click to expand)</span> </strong></summary>

4) representation
   open the black box. What representation arises from learning the score.
   spatial average of channels in the deepest layer: sparse and selective (union of subspaces)
</details>
   
   
   
## Utilizing learned density models to solve inverse problems: 
Ultimately, we want to learn the density to use it! Inverse problems in signal processing (a particular approach: it is stochastic) 
### Linear inverse problems 
Solving inverse problem (updated version neurips)
optimal measurement 
cone excitation
### non-linear inverse problems
- feature- guided 
- texture model 




<!-- My PhD thesis link  -->

