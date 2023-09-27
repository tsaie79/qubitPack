# add a setup file to the package directory

from setuptools import setup, find_packages

setup(
    name="qubitPack",
    author="Jeng-Yuan Tsai",
    version="0.1.0",
    email="tsaie79@gmail.com",
    packages=find_packages(),
    install_requires=[
        "smoqe@git+https://github.com/dangunter/smoqe.git@master#egg=smoqe"
        "pymatgen_db@git+https://github.com/materialsproject/pymatgen-db.git@c3271276c2ef26dc98ccc86634405a04cd677395"
        "#egg=pymatgen_db",
        "pymatgen_diffusion@git+https://github.com/materialsvirtuallab/pymatgen-analysis-diffusion.git"
        "@51d84ea1fd034941baa22d4b1b8610a4cc6fb801#egg=pymatgen_diffusion",
        "ase@git+ssh://git@github.com/rosswhitfield/ase.git@07de35654601ddbb2b23a4e7df7091696b0af108#egg=ase",
        # add ase from github with the version 3.18.1
        # "ase==3.18.1",
        "pycdt@git+https://github.com/tsaie79/pycdt.git@master#egg=pycdt",
        "pymatgen@git+https://github.com/tsaie79/pymatgen.git@master#egg=pymatgen",
        "atomate@git+https://github.com/tsaie79/atomate.git@master#egg=atomate",
        "FireWorks==1.9.6",
        "custodian==2020.4.27",
        "phonopy==2.12.0",
        "pymongo==3.11.0",
    ],
)

