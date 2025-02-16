from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "graphics_utils",
        [
            "src/bindings.cpp",
            "src/films.cpp",
        ],
        include_dirs=["include"],
    ),
]

setup(
    name="graphics_utils",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)