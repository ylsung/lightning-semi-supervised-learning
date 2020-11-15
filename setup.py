#!/usr/bin/env python

import platform
from setuptools import setup, find_packages


setup(
    name="lightning-semi-supervised-learning",
    description="Ready-to-use semi-supervised learning under one common API",
    version="master",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["torch==1.7.0", "torchvision", "pytorch-lightning==1.0.2"],
    extras_require={
        "test": [
            "coverage",
            "pytest",
            "flake8",
            "pre-commit",
            "codecov",
            "pytest-cov",
            "pytest-flake8",
            "flake8-black",
            "black",
        ]
    },
)
