from setuptools import setup, find_packages
setup(
    name='g2',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[sphinx==8.3.0, numpy>=1.20.0, scipy>=1.7.0, jax>=0.3.0, jaxlib>=0.3.0,
                      sncosmo>=2.0.0, matplotlib>=3.5.0, astropy>=5.0.0, sphinx_copybutton]
)