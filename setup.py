#!/usr/bin/env python3

import os
from setuptools import setup

def _read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as stream:
        return stream.read()


setup(
    name='DFOL-VQA',
    author="Saeed Amizadeh",
    author_email="saamizad@microsoft.com",
    description="Differentiable First Order Logic Reasoning for Visual Question Answering",
    long_description=_read('./README.md'),
    long_description_content_type='text/markdown',
    keywords="neurosymbolic reasoning,visual reasoning,differentiable first order logic,visual question answering,vqa",
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    url="https://github/Microsoft/DFOL-VQA",
    version='0.0',
    python_requires=">=3.6",
    install_requires=[
        "numpy >= 1.16.4",
        "torch >= 1.3.0",
        "h5py >= 2.9.0",
        "pyyaml >= 5.1.2",
        "opencv-python >= 4.1.0",
        "pattern >= 3.6"
    ],
    package_dir={"": "src"},
    packages=[
        "nsvqa.data",
        "nsvqa.nn",
        "nsvqa.nn.interpreter",
        "nsvqa.nn.parser",
        "nsvqa.nn.vision",
        "nsvqa.train"
    ],
    scripts=[
        "src/gqa_interpreter_experiments.py",
        "src/gqa_preprocess.py"
    ]
)
