from setuptools import setup, find_packages

setup(
    name="MVPy",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.10.1"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "flake8>=3.8.0"
        ]
    },
)