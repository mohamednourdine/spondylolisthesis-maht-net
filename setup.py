from setuptools import setup, find_packages

setup(
    name='spondylolisthesis-maht-net',
    version='0.1.0',
    author='Your Name',
    author_email='mohamednjikam25@hotmail.com',
    description='Automated Spondylolisthesis Grading using MAHT-Net',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/spondylolisthesis-maht-net',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'opencv-python',
        'PyYAML',
        'tqdm',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)