.. g2 documentation master file, created by
   sphinx-quickstart on Fri Sep 12 15:51:09 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:math:`g^{(2)}` Documentation
=============================

Package for the calculation of Hanbury-Brown Twiss and intensity interferometry obserables
and signal-to-noise in astronomcial observations.

The astronomical source is specified by its intensity distribution on the sky :math:`I_\nu`,
and its second-order coherence function :math:`g^{(2)}`.

The observation is specified by the telescope aperture, baseline between telesocopes, the
bandpass, detector jitter, throughput, and exposure time.

Important functionalities include:

- Calculation of the partial derivatives of the signal with respect to the source parameters.
- Calculation of the expected signal-to-noise ratio for a given source and observation setup.

The current implementation is restricted to several conditions:

- The source is chaotic.
- The source is unpolarized.
- The detector timing jitter is Gaussian with standard deviation :math:`\sigma_t` .
- The bandwidth :math:`\Delta\omega` is sufficiently narrow such that :math:`I_\nu` does not vary across the bandpass.
- The bandwidth :math:`\Delta\omega` is sufficiently narrow such that :math:`\sigma_t \Delta\omega \gg 1`.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   api
