Usage
=====

Installation
------------

To use g2, first install it from GitHub.  As we are completely starting there
is no release.  Work with the main repository.

.. code-block:: console

   (.venv) $ git clone git@github.com:AlexGKim/g2.git

Install the Package

.. code-block:: console

   (.venv) $ cd g2
   (.venv) $ pip install .

quickstart
----------

The Basics
^^^^^^^^^^

The objective is to determine quantities related to the second-order coherence function :math:`g^{(2)}`
and measurements thereof.  The current implementation is constrained to restrive conditions 
listed in :ref:`index-label`, which simplify the required inputs and calculations.

One condition is that the source be chaotic.  The source is then represented as an object of the 
abstract base class :class:`g2.models.base.source.ChaoticSource`.  The methods that need to
be specified are :meth:`g2.models.base.source.AbstractSource.intensity`, :meth:`g2.models.base.source.AbstractSource.total_flux`,
and  :meth:`g2.models.base.source.AbstractSource.get_params`. 


There are several concrete classes
available in :mod:`g2.models.sources`.

.. literalinclude:: ../examples/uniform_disk.py
   :language: python