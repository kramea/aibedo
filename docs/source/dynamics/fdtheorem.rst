.. aibedo_fd:

Fluctuation Dissipation Theorem
===============================

Fluctuation Dissipation Theorem (FDT) for the climate system posits that the climate system response to relatively small perturbations can be estimated using a climate response operator derived from the statistics of unforced climate variability. That is, the response of a variable :math:`X` to a forcing :math:`\delta f` can be estimated as 

.. math::
    L\delta X = \delta f

where :math:`L` is typically a linear response function (LRF) representing all relevant processes connecting :math:`f` to :math:`X` (Majda et al., 2010). The LRF is estimated by integrating

.. math::
    L_{FDT} = - \left[\int_0^{\infty } \mathbf{C}(\tau)\mathbf{C}(0)^{-1} d\tau \right]^{-1}

for covariance matrices :math:`C` and time lag :math:`\tau`. To effectively estimate the LRF, it is critical to have a large sample of internal variability with which to estimate the covariance matrices. 

We are therefore aiming to use the internal variability relationships between cloud variables and target climate variables such as surface temperature and pressure to estimate the response to MCB. However, there are significant shortcomings of the linear FDT method (Liu et al., 2018). Thus, we instead use the philosophy of the FDT and replace the LRF with an ML model, allowing a more comprehensive determination of the relationships between variables and possibly capturing non-linear interactions. This nevertheless makes the assumption that the relationships between modes and variables does not change as the climate changes, which we assume hold for the levels of warmings in our study (historical and near future warming).In order to construct a response operator with AiBEDO, we thus train separate versions of the model at different time lags and integrate them together, out to a time lag wherein the AiBEDO response converges to zero. The theoretical approach to this problem is under development.

Majda, Andrew J., Boris Gershgorin, and Yuan Yuan. “Low-Frequency Climate Response and Fluctuation–Dissipation Theorems: Theory and Practice.” Journal of the Atmospheric Sciences 67, no. 4 (April 1, 2010): 1186–1201. https://doi.org/10.1175/2009JAS3264.1.

Liu, Fukai, Jian Lu, Oluwayemi Garuba, L. Ruby Leung, Yiyong Luo, and Xiuquan Wan. “Sensitivity of Surface Temperature to Oceanic Forcing via Q-Flux Green’s Function Experiments. Part I: Linear Response Function.” Journal of Climate 31, no. 9 (May 1, 2018): 3625–41. https://doi.org/10.1175/JCLI-D-17-0462.1.