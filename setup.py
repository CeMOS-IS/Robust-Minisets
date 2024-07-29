# To build and upload to PyPI:
#
#     python3 setup.py sdist bdist_wheel
#     python3 -m twine upload dist/*
#

from setuptools import setup, find_packages

import robust_minisets


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


README = readme()


setup(
    name='robust-minisets',
    version=robust_minisets.__version__,
    url=robust_minisets.HOMEPAGE,
    license='Apache-2.0 License',
    author='CeMOS-IS',
    author_email='oehri.sven@gmail.com',
    python_requires=">=3.8.0",
    description='[Robust-Minisets] A collection of low resolution robustness and generalization benchmarks for image classification',
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "torch",
        "torchvision",
        "scikit-image",
        "tqdm",
        "fire"
    ],
    zip_safe=True,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only"
    ]
)