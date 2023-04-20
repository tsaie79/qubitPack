# add a setup file to the package directory

from setuptools import setup, find_packages

setup(
    name="qubitPack",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pymatgen-db==2019.3.28",
        "pymatgen_diffusion@git+https://github.com/materialsvirtuallab/pymatgen-analysis-diffusion.git"
        "@51d84ea1fd034941baa22d4b1b8610a4cc6fb801#egg=pymatgen_diffusion",
        "ase@git+ssh://git@github.com/rosswhitfield/ase.git@07de35654601ddbb2b23a4e7df7091696b0af108#egg=ase",
        "atomate@git+https://github.com/tsaie79/atomate.git@c2b16d05a68999b4adc88afa2802804132c691b3#egg=atomate",
        "MPInterfaces@git+https://github.com/henniggroup/MPInterfaces.git@f46fe69f224511c3537beecf8152659ec10b20cf#egg=MPInterfaces",
        "pycdt@git+https://github.com/tsaie79/pycdt.git@aaa8249ed4dace846ca49299b83aa831230b70f8#egg=pycdt",
        "pymatgen@git+ssh://git@github.com/tsaie79/pymatgen.git@7b24db75bab5fc30735e92d293db3f0c05b7aefe#egg=pymatgen",
    ],
)

