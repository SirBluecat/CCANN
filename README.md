# CCANN
Convolutional Cell Averaged-based Neural Network
## Introduction
The Cell Averaged-based Neural Network (CANN) method provides a local solution to the next time step based on the local solution at the current time step. This means that for each time step, we must repeatedly apply the network to each point in the domain. This limits speed of approximation, especially if one wishes to lift the method to higher dimensions.
The goal of extending CANN to a convolutional structure is to enable efficient lifting of solutions to higher dimensions and longer time-scales. If this proves successful, we will also work on the advection equation $u_t+u_x=0$, which is more challenging due to the hyperbolic nature of the equation.
## Details
In traditional numerical methods for approximation of partial differential equation solutions (finite difference, volume, element, etc.), time step sizes are restricted by CFL conditions—necessary conditions for convergence while solving partial differential equations. For example consider the 1D heat equation
$$u_t-u_{xx} = 0.$$
The explicit finite difference approximation is then
$$\dfrac{u^{n+1}_j-u^n_j}{\Delta t}-\dfrac{u_{j-1}^n-2u_j^n+u_{j-1}^n}{\Delta x^2} = 0.$$
Small time steps are needed to maintain stability, i.e. $\Delta t \approx \Delta x^2$. Implicit schemes can also be used, e.g. $$\dfrac{u^{n+1}_j-u^n_j}{\Delta t}-\dfrac{u_{j-1}^{n+1}-2u_j^{n+1}+u_{j-1}^{n+1}}{\Delta x^2} = 0$$
which can take larger time steps, but require solving a large linear system of equation for each time step. 

The model learns a map from $u(x,0)$ to $u(x,\Delta t)$, and is able to extrapolate solutions to larger times $u(x,n\Delta t)$. Thus, data will be generated from the exact solution to this smaller time domain. For example, one solution of the heat equation is $u(x,t)=\sin(\omega x)e^{-\omega^2t}$ so we can take training data from several different $\omega$ values, and test on a new set of frequencies. 
In this scheme, our features are the solution values at each cell. One cell represents a portion of our computational domain $$I_1 = [0,\Delta x],~I_2 =[\Delta x, 2\Delta x],~\ldots$$
The labels are not labels in the traditional sense. Instead, they are the cell averages at the \textit{next} time step, since this is what the model is designed to predict. 
## Related Work
[Qiu and Yan](https://arxiv.org/abs/2107.00813) introduced the original cell average based neural network method. This method takes a finite volume approach to solving PDEs of the form $u_t = \mathcal{L}(u)$ and split the spatial domain into cells $I_j$ and define the cell average $\bar{u}_j(t)=\frac{1}{\Delta x}\int_{I_j} u(x,t) ~dx$. Now we integrate the PDE over one cell $I_j$ from $t_n$ to $t_{n+1}$ to get 
$$\frac{1}{\Delta x}\int^{t_{n+1}}_{t_n}\int_{I_j} ~ u_t ~dxdt=\frac{1}{\Delta x}\int^{t_{n+1}}_{t_n}\int_{I_j} ~ -\mathcal{L}(u)~ dxdt \Rightarrow \bar{u}_j(t_{n+1}) -\bar{u}_j(t_{n})=\frac{1}{\Delta x}\int^{t_{n+1}}_{t_n}\int_{I_j} ~ -\mathcal{L}(u)~ dxdt $$
A neural network is used to approximate the right hand side 
$$\mathcal{N}(V^{in}; \Theta)\approx \frac{1}{\Delta x}\int^{t_{n+1}}_{t_n}\int_{I_j} ~ -\mathcal{L}(u)~ dxdt$$\\
Leading to the update scheme
$$\bar{v}_j(t_{n+1})\approx \bar{v}_j(t_{n})+\mathcal{N}(\vv^{in}; \Theta)$$
$\vv^{in}$ represents the numerical stencil of the method around the $I_j$ cell. For example a symmetric stencil may have the form $\vv^{in} = [\bar{v}_{j-1}, \bar{v}_j, \bar{v}_{j+1}]$. The main advantage of the method is the ability to take larger time steps than a traditional numerical method. 
