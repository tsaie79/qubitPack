# add a setup file to the package directory

from setuptools import setup, find_packages

setup(
    name="qubitPack",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "atomate@git+https://github.com/tsaie79/atomate.git@c2b16d05a68999b4adc88afa2802804132c691b3#egg=atomate"
    ],
)

