

<div style="text-align: center;">
  <a href="https://github.com/ml-jku/neugk" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=FFFFFF" alt="code">
  </a>
  &nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2502.07469" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-B31B1B?style=for-the-badge&logo=arXiv&logoColor=FFFFFF" alt="paper">
  </a>
</div>

<!-- # <img src="imgs/neugk_icon.png" alt="neugk Icon" height="24px"> Efficiently Modeling 5D Plasma Turbulence Simulations -->

---

<figure style="text-align: center;">
    <img src="imgs/velocity_surfaces.gif" alt="5d distribution function" width="70%">
    <figcaption style="color: white; font-size: 14px; margin-top: 8px;">
    Figure 1: Visualizing the 5D distribution funciton, as 3D toruses within the 2D velocity space. 5D plotting has never been easier ;)
    </figcaption>
</figure>

## TL;DR
<div style="border-left: 4px solid #633d11; background-color: #c27721; padding: 12px 16px; margin: 1em 0; border-radius: 4px;">
<strong>Nuclear fusion is hard</strong>, as it requires understanding physical phenomena like plasma turbulence. One way to do this is with very expensive <strong>numerical simulations, called gyrokinetics</strong>. We propose <img src="imgs/neugk_icon.png" alt="neugk Icon" height="12px"> <strong>NeuGK</strong>, a neural surrogate model based on <strong>swin transformers</strong> for nonlinear gyrokinetic equations, which models Plasma turbulence in a <strong>5D phase space</strong>, unlike existing methods which take reduced approaches, and offers a <strong>>1000x speedup</strong> compared to numerical gyrokinetics solvers.
</div>

## Introduction

Nuclear Fusion is a promising contender for sustainable energy production. For humans, we can use the energy released by the reaction of fusing two hydrogen atoms (to form helium and a neutron, where most energy ends up) to power, for example, a turbine. In nature, fusion naturally occurs in stars, where the strong gravitational force produces massive pressures which overcome the repelling force between two hydrogen atoms (_Coulomb barrier_). The figure below shows the fusion reaction for hydrogen isotopes Deutereum and Tritium.

<figure style="text-align: center;">
    <img src="imgs/dt_fusion.gif" alt="D-T nuclear fusion reaction" width="70%">
    <figcaption style="color: white; font-size: 14px; margin-top: 8px;">
    Figure 2: Animated deuterium-tritium nuclear fusion. Source: <a href="https://commons.wikimedia.org/wiki/File:Animated_D-T_fusion.gif">wikimedia.org</a>
    </figcaption>
</figure>

Since the gravitational force on planet Earth is insufficient to overcome this barrier, we require massive amounts of heat to increase the probability of atoms fusing. In reality, this means heating up a gas to hundreds of millions of degrees, which is also called a **Plasma**. At these conditions, hydrogen atoms split into ions and electrons. Due the amount of heat, the Plasma has to be confined within magnetic fields, as there is no material that could withstand such extreme temperatures.

In order to turn nuclar fusion into a viable energy source, Plasma needs to be confined over long periods of time. This is difficult, as it's inherently unstable, and often tries to escape confinement. One contributor to this behavior is __turbulence__, which arises due to temperature gradients within the Plasma. Therefore it is essential to understand and model Plasma turbulence in modern reactors such as Tokamaks, in order to downstream design new reactors and confinement control systems.
However, understanding and modelling Plasma turbulence is an incredibly hard problem, and practictioners must rely on expensive numerical simulations.

> _"Nuclear fusion is not rocket science, because it's way harder."_  
> — *William Hornsby*

Because it promises __sustainable, safe, and relatively affordable energy__, __nuclear fusion__ remains a key focus in the pursuit of securing our __future energy supply.__

## The Problem

Understanding **plasma turbulence** is crucial for modelling plasma scenarios for confinement control and reactor design. Numerically, Plasma turbulence is governed by the **nonlinear gyrokinetic equation**, which evolves a **5D distribution function** over time **in phase space**.

Let $f = f(x, y, s, v_{\parallel}, \mu)$ where:

- $x$, $y$ are spatial coordinates along a toroidal C-section of a torus in real space.
- $s$ is the toroidal coordinate along the field line, going around the torus.
- $v_{\parallel}$ the parallel velocity component along the field lines.
- $\mu$ is the magnetic angular moment, related to the gyral motion of particles.

The time-evolution of the perturbed distribution $f$, usually called $\delta f$, is governed by the gyrokinetic equation [[5](#ref-gyrokinetics), [6](#ref-gyrokinetics2), [7](#ref-gyrokinetics3)], a reduced form of the Vlasov-Maxwell PDE system

$$\frac{\partial f}{\partial t} + (v_\parallel \mathbf{b} + \mathbf{v}_D) \cdot \nabla f -\frac{\mu B}{m} \frac{\mathbf{B} \cdot \nabla B}{B^2} \frac{\partial f}{\partial v_\parallel} + \mathbf{v}_\chi \cdot \nabla f = S$$

Where:

- $v_{\parallel} \mathbf{b}$ is the motion along magnetic field lines.  
- $\mathbf{b} = \mathbf{B} / B$ is the unit vector along the magnetic field $\mathbf{B}$ with magnitude $B$.
- $\mathbf{v}_D$ is the magnetic drift due to gradients and curvature in $\mathbf{B}$.
- $\mathbf{v}_\chi$ describes nonlinear drifts arising from crossing electric and magnetic fields.
- $S$ is a forcing term [TODO].

The **nonlinear term** $\mathbf{v}_\chi \cdot \nabla f$ describes turbulent advection, and the resulting nonlinear coupling constitutes the computationally most expensive term. For more details on gyrokinetics and the derivation of the equation, check _"The non-linear gyro-kinetic flux tube code GKW_" from _Arthur Peeters et al._ [[5](#ref-gyrokinetics)] and _"Gyrokinetics_" by _Xavier Gerbet and Maxime Lesur_ [[6](#ref-gyrokinetics3)].

<div style="border-left: 4px solid #633d11; background-color: #c27721; padding: 12px 16px; margin: 1em 0; border-radius: 4px;">
    The phase-space distribution function (<strong>Eulerian</strong>) is not the only way gyrokinetics can be parametrized. Instead, we can describe it with an ensemble of trajectories governed by an SDE, where gyrocenters are tracked directly as particles (<strong>Lagrangian</strong>). With the Lagrangian approach, the distribution function can then be recovered from by sampling many gyro-paths (particle-in-cell methods, [<a href="#ref-pic">8</a>]). 
    <strong>Both frames are physically equivalent, but offer different insights and numerical advantages</strong>. <br><br>
    This duality is related to the <strong><a href="https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck_equation">Fokker-Planck equation</a></strong> [<a href="#ref-fokker">9</a>], which describes the evolution of probability densities under drift and diffusion. The Fokker-Planck equation is derived from an SDE, such as the <a href="https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process">Ornstein–Uhlenbeck process</a> (as the forward Kolmogorov), and so it links the distribution-based and SDE-based descriptions.
    <!-- It is also connected to how collisions are modeled in gyrokinetics, with the Vlasov-Fokker-Planck equation.<br><br> -->
</div>

Fully resolved gyrokinetics simulations are prohibitively expensive and often times practitioners need to rely on **reduced order models**, such as quasilinear models (for example QuaLiKiz [[10](#ref-qualikiz)]). QL models are fast but severely limited in generalization and accuracy, as they entirely neglect the nonlinear term $\mathbf{v}_\chi \cdot \nabla f$. However, machine learning (neural) surrogate models have the potential to overcome this limitation if we develop methodologies that can cope with the complex nature of 5D data.

## Our Approach

Our intuition is that **modeling the entire 5D distribution function is vital** to accurately model and comprehend turbulence in Plasmas.
Therefore we require techniques that can process 5D data. The classic repertoire of an ML engineer comprises a variety of different techniques. Let's break it down whether there are suitable to proccess 5D data.

- **Convolutions?** There is no out-of-the-box convolution kernel for &gt;3D, so they need to be implemented either directly, recursively, or in a factorized manner. Regardless, convolutions become expensive and memory-intensive. [[11](#ref-cnn)]
- **Transformers?** ViTs can be applied to any number of dimension (provided proper patching / unpatching layers), but their quadratic scaling makes them unfeasible in our 5D setting due to quickly growing sequence lengths. [[12](#ref-attention), [13](#ref-vit)]
- **Linear Attention?** Vision Transformers with linear attention such as the Shifted Window Transformer (swin) [[14](#ref-swin)] overcome quadratic scaling by performing attention locally in a simple way, making them an handy candidate for our case. However, to date, no implementation of swin Transformer exists that can process 5D data.

<strong> <span>&#8618;</span> </strong> There is no architecture out there that can natively handle 5D inputs. <strong><em>So, now what?</em></strong>

We generalize the local window attention mechanism used in Swin Transformers, together with patch merging and unpatching layers, to n-dimensional inputs. To this end, we generalize the (patch and window) partitioning strategy used in these layers to be able to process inputs of arbitrary dimensionalities. A standalone implementation of our n-dimensional swin layers can be found [on github](https://github.com/gerkone/ndswin). The figure below provides an illustration on how we extend (shifted) window attention to n dimensions.

<figure style="text-align: center;">
    <img src="imgs/archv2.png" alt="nD swin attention in the 5D case" width="100%">
    <figcaption style="color: white; font-size: 14px; margin-top: 8px;">
    Figure 3: Shifted window attention in the 2D (image), 3D (video), and 5D (our) case. In a layer, attention is performed locally only within components with the same color.
    </figcaption>
</figure>

We propose Neural Gyrokinetics, <img src="imgs/neugk_icon.png" alt="neugk Icon" height="12px"> **NeuGK**, the first ever neural surrogate for nonlinear Gyrokinetic simulations. We start from the popular UNet architecture [[15](#ref-unet)] and model the temporal evolution of the distribution function in an autoregressive manner. Furthermore, NeuGK is a multitask model and not only predicts the evolution of the 5D distribution function, but also 3D potential fields and heat flux, which are usually obtained by performing complex integrals on the 5D fields. NeuGK achieves this through three output branches at different dimensions that share latents with cross attention.


<figure style="text-align: center;">
    <img src="imgs/figure1.png" alt="NeuGK architecture compared to quasilinear" width="80%">
    <figcaption style="color: white; font-size: 14px; margin-top: 8px;">
    Figure 4: NeuGK multitask trainin pipeline. We directly model the 5D distribution function of nonlinear gyrokinetics and incorporates 3D electrostatic potential fields and turbulent transport quantities, such as heat flux.
    </figcaption>
</figure>

## Results

### Does NeuGK accurately predict heat flux?

> [TODO: Visualization of heat flux trajectory]

### Does NeuGK capture underlying physics?

> [TODO: Diagnostics plots — zonal flow and spectra]

# Wrapping up

NeuGK outperforms reduced numerical approaches and machine learning baselines in modelling plasma turbulence. It accurately captures nonlinear phenomena and spectral quantities self-consistently, while offering a three order of magnitude speedup compared to the numerical solver GKW [[7](#ref-gyrokinetics3)]. As a result, NeuGK offers a fruitful alternative to efficient approximation of turbulent transport and opens up a variety of research directions leveraging the potential of neural surrogates for Plasma turbulence modelling. Finally, Plasma turbulence modelling is an incredibly hard problem, but we believe that Machine Learning will disrupt the landscape of Plasma turbulence modelling in the future. 

<figure style="text-align: center;">
    <img src="imgs/here_to_help.jpg" alt="xkcd 1831: here to help (edited)" width="100%">
    <figcaption style="color: white; font-size: 14px; margin-top: 8px;">
    xkcd: 1831
    </figcaption>
</figure>

## Cite
```
@misc{galletti20255dneuralsurrogatesnonlinear,
      title={5D Neural Surrogates for Nonlinear Gyrokinetic Simulations of Plasma Turbulence}, 
      author={Gianluca Galletti and Fabian Paischer and Paul Setinek and William Hornsby and Lorenzo Zanisi and Naomi Carey and Stanislas Pamela and Johannes Brandstetter},
      year={2025},
      eprint={2502.07469},
      archivePrefix={arXiv},
      primaryClass={physics.plasm-ph},
      url={https://arxiv.org/abs/2502.07469}, 
}
```

## References
<a name="ref-chen"></a>
[1] Francis F. Chen, *Introduction to Plasma Physics and Controlled Fusion*, 3rd ed., Springer, 2016.

<a name="ref-freudenrich"></a>
[2] Craig Freudenrich, "Physics of Nuclear Fusion Reactions," *HowStuffWorks*, Aug. 4, 2015. [Online]. Available: [http://science.howstuffworks.com/fusion-reactor1.html](http://science.howstuffworks.com/fusion-reactor1.html)

<a name="ref-tokamaks"></a>
[3] J. Wesson, *Tokamaks*, 4th ed., Oxford University Press, 2011. 

<a name="ref-inertial"></a>
[4] S. Atzeni and J. Meyer-ter-Vehn, *The Physics of Inertial Fusion: Beam Plasma Interaction, Hydrodynamics, Hot Dense Matter*, Oxford University Press, 2004.

<a name="ref-gyrokinetics"></a>
[5] A. G. Peeters et al., *The non-linear gyro-kinetic flux tube code GKW*, Computer Physics Communications, 180(12), 2650–2672, 2009.

<a name="ref-gyrokinetics2"></a>
[6] E. A. Frieman and L. Chen, “Nonlinear gyrokinetic equations for low-frequency electromagnetic waves in general plasma equilibria,” *Phys. Fluids*, vol. 25, no. 3, pp. 502–508, Mar. 1982.

<a name="ref-gyrokinetics3"></a>
[7] X. Garbet and M. Lesur, *Gyrokinetics*, Lecture notes, France, Feb. 2023. [Online]. Available: https://hal.science/hal-03974985

<a name="ref-pic"></a>
[8] C. K. Birdsall and A. B. Langdon, *Plasma Physics via Computer Simulation*, Taylor & Francis, 2004.

<a name="ref-fokker-planck"></a>
[9] H. Risken, *The Fokker-Planck Equation: Methods of Solution and Applications*, 2nd ed., Springer, 1989.

<a name="ref-qualikiz"></a>
[10] A. Casati, J. Citrin, C. Bourdelle, et al., "QuaLiKiz: A fast quasilinear gyrokinetic transport model," *Computer Physics Communications*, vol. 254, p. 107295, 2020.

<a name="ref-cnn"></a>
[11] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, pp. 436–444, 2015.

<a name="ref-attention"></a>
[12] A. Vaswani, N. Shazeer, N. Parmar, et al., "Attention Is All You Need," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

<a name="ref-vit"></a>
[13] A. Dosovitskiy, L. Beyer, A. Kolesnikov, et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *International Conference on Learning Representations (ICLR)*, 2021.

<a name="ref-swin"></a>
[14] Z. Liu, Y. Lin, Y. Cao, et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 10012–10022.

<a name="ref-unet"></a>
[15] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2015, pp. 234–241.

---
2025, Gianluca Galletti