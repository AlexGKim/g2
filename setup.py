from setuptools import setup, find_packages

setup(
    name="g2",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "jax>=0.3.0",
        "jaxlib>=0.3.0",
        "sncosmo>=2.0.0",
        "matplotlib>=3.5.0",
        "astropy>=5.0.0",
    ],
    author="Alex Kim",
    description="Inverse Noise Calculation for Chaotic Sources",
    python_requires=">=3.9",
)
