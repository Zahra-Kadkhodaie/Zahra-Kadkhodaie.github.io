---
title: "Stochastic Solutions for Linear Inverse Problems using the Prior Implicit in a Denoiser"
collection: publications
category: conferences
permalink: /publication/2021-12-06-paper-title-stochastic
excerpt: 'In this paper, we introduce an algorithm to draw samples from a prior embedded in a deep net denoiser. We then upgrade the algorithm to draw samples conditioned on a linear constraint.'
date: 2021-12-06
venue: 'NeurIPS'
paperurl: 'https://proceedings.neurips.cc/paper/2021/file/6e28943943dbed3c7f82fc05f269947a-Paper.pdf'
citation: 'Kadkhodaie, Zahra. (2021). &quot;Stochastic solutions for linear inverse problems using the prior implicit in a denoiser.&quot; <i>Advances in Neural Information Processing Systems</i> 34 (13242-13254).'
---

Deep neural networks have provided state-of-the-art solutions for problems such as image denoising, which implicitly rely on a prior probability model of natural images. Two recent lines of work–Denoising Score Matching and Plug-and-Play–propose methodologies for drawing samples from this implicit prior and using it to solve inverse problems, respectively. Here, we develop a parsimonious and robust generalization of these ideas. We rely on a classic statistical result that shows the least-squares solution for removing additive Gaussian noise can be written directly in terms of the gradient of the log of the noisy signal density. We use this to derive a stochastic coarse-to-fine gradient ascent procedure for drawing high-probability samples from the implicit prior embedded within a CNN trained to perform blind denoising. A generalization of this algorithm to constrained sampling provides a method for using the implicit prior to solve any deterministic linear inverse problem, with no additional training, thus extending the power of supervised learning for denoising to a much broader set of problems. The algorithm relies on minimal assumptions and exhibits robust convergence over a wide range of parameter choices. To demonstrate the generality of our method, we use it to obtain state-of-the-art levels of unsupervised performance for deblurring, super-resolution, and compressive sensing.
