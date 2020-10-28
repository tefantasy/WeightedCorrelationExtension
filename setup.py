from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='weighted_corr',
    ext_modules=[
        CUDAExtension('weighted_corr_cuda', [
            'weighted_corr.cpp',
            'weighted_corr_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
