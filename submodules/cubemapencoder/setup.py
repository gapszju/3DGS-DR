from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

nvcc_flags = [
    '-O3', '-std=c++17',
    #'-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
	'-DCUDA_HAS_FP16=1', '-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__',
    '-use_fast_math'
]


c_flags = ['-O3', '-std=c++17']


setup(
    name="cubemapencoder",
    packages=["cubemapencoder"],
    ext_modules=[
        CUDAExtension(
            name="_cubemapencoder",
            sources=[
            "src/cubemapencoder.cu",
            "src/bindings.cpp",
	],
            extra_compile_args={"nvcc": nvcc_flags, "cxx": c_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
