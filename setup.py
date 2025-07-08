from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    include_package_data=True,
    ext_modules=[
        CUDAExtension(
            name='fssim.bindings',
            sources=['fssim/bindings.cpp', 'fssim/kernel.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
