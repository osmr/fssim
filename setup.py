from setuptools import setup, find_packages
from os import path
from io import open
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fssim',
    description='Fused SSIM',
    url='https://github.com/osmr/fssim',
    author='Oleg Sémery',
    author_email='osemery@gmail.com',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='image processing',
    packages=find_packages(exclude=['others', '*.others', 'others.*', '*.others.*']),
    install_requires=['numpy'],
    python_requires='>=3.10',
    include_package_data=True,
    ext_modules=[
        CUDAExtension(
            name='fssim.bindings',
            sources=['fssim/bindings.cpp', 'fssim/kernel.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
