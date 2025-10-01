from setuptools import setup, find_packages

setup(
    name="graphspec",
    version="0.1.0",
    description="Spectral Graph Feature Learning for MLPs",
    author="mdindoost",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.8",
)
