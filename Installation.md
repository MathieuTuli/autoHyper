There are two versions of the autoHyper code contained in this repository.
1. a python-package version of the autoHyper code, which can be `pip`-installed.
2. a static python module (unpackaged), runable as a script.

#### Python Package ####

---

##### Repository Cloning #####
After cloning the repository (`git clone https://github.com/MathieuTuli/autoHyper.git`), simply run
```console
python setup.py build
python setup.py install
```
or
```console
pip install .
```
If you will be making changes and wish to not have to reinstall the package each time, run
```console
pip install -e .
```

Note that `pip install -e .` is a command included in the [requirements.txt](requirements.txt) file.
