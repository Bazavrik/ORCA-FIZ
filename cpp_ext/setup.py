from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "fuzzy_sd_cpp",
        ["fuzzy_sd_pybind.cpp"],
        include_dirs=["."],
        cxx_std=17,
    ),
]

setup(
    name="fuzzy_sd_cpp",
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)