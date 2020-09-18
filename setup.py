#!/usr/bin/env python

import platform
from setuptools import setup, find_packages

def torch_urls(version):
    platform_system = platform.system()
    if platform_system == 'Windows':
        return f"torch@https://download.pytorch.org/whl/cu90/torch-{version}-cp36-cp36m-win_amd64.whl#"
    return f"torch=={version}"

setup(
    name="lightning-semi-supervised-learning",
    description="Ready-to-use semi-supervised learning under one common API",
    version="master",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        torch_urls("1.6"),
        "torchvision",
        "pytorch-lightning"
    ],
    extras_require={
        "test": ["coverage"]
    },
)