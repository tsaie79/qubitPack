# add a setup file to the package directory

from setuptools import setup, find_packages

setup(
    name="qubitPack",
    author="Jeng-Yu Tsai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "smoqe@git+https://github.com/dangunter/smoqe.git@master#egg=smoqe"
        "pymatgen_db@git+https://github.com/materialsproject/pymatgen-db.git@c3271276c2ef26dc98ccc86634405a04cd677395"
        "#egg=pymatgen_db",
        "pymatgen_diffusion@git+https://github.com/materialsvirtuallab/pymatgen-analysis-diffusion.git"
        "@51d84ea1fd034941baa22d4b1b8610a4cc6fb801#egg=pymatgen_diffusion",
        "ase@git+ssh://git@github.com/rosswhitfield/ase.git@07de35654601ddbb2b23a4e7df7091696b0af108#egg=ase",
        "pycdt@git+https://github.com/tsaie79/pycdt.git@aaa8249ed4dace846ca49299b83aa831230b70f8#egg=pycdt",
        "pymatgen@git+https://github.com/tsaie79/pymatgen.git@master#egg=pymatgen",
        "atomate@git+https://github.com/tsaie79/atomate.git@c2b16d05a68999b4adc88afa2802804132c691b3#egg=atomate",
        "FireWorks==1.9.6",
        "custodian==2020.4.27",
        "phonopy==2.12.0",
        "pymongo==3.11.0",
    ],
)

