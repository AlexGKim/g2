Usage
=====

Installation
------------

You may or may not want to install in a virtual environment.  If you do, create and activate
a virtual environment first.  For example, with conda

.. code-block:: console

   $ conda create -n g2 python=3.10

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

Examples
^^^^^^^^

A simple example is a uniform disk source, which is implemented in :class:`g2.models.sources.simple.UniformDisk`.

.. literalinclude:: ../examples/examples_source.py
   :start-after: # uniform_disk-begin
   :end-before: # uniform_disk-end
   :language: python

Another example is a more complex source model based on a spatial grid of intensity profiles
from a supernova simulation.  This is implemented in :class:`g2.models.sources.grid_source.GridSource`,
which has a class method to instantiate a source based on the Type Ia supernova SN2011fe.

.. literalinclude:: ../examples/examples_source.py
   :start-after: # sn2011fe-begin
   :end-before: # sn2011fe-end
   :language: python

Some of the information that can be accessed from a source is demonstrated with the call

.. literalinclude:: ../examples/examples_source.py
   :start-after: # summary-call-begin
   :end-before: # summary-call-end
   :language: python

where the function :func:`g2.utils.summary` is

.. literalinclude:: ../examples/examples_source.py
   :start-after: # summary-begin
   :end-before: # summary-end
   :language: python

The full example code is in :file:`docs/examples/examples_source.py`.