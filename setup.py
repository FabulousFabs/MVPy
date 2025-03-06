from setuptools import setup, find_packages

setup(
    name="mvpy",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.10.1",
        "pandas>=1.5.3",
        "torch>=2.5.1",
        "scikit-learn>=1.2.1",
        "mne>=1.8.0",
        "tqdm>=4.64.1"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "flake8>=3.8.0"
        ]
    },
)