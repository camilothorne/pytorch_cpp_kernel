from torch.utils.cpp_extension import BuildExtension, CppExtension
from setuptools import setup
from distutils.tests.test_archive_util import test_suite

__version__ = "0.0.1"

my_ext_modules=[
        CppExtension(name='torch_kernel', 
                     sources=['src/torch_kernel.cpp'], 
                     extra_compile_args=['-std=c++14'],
                     define_macros=[("VERSION_INFO", __version__)],
                     ),
    ]

setup(
    name="torch_kernel",
    version=__version__,
    author="Camilo Thorne",
    description="A test PyTorch project using pybind11",
    ext_modules=my_ext_modules,
    extras_require={"test": "pytest"},
    test_suite = 'tests',
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    python_requires=">=3.6",
)