from setuptools import setup


setup(
    name='patchwork',
    version='1.0',
    description='Hierarchical CNNs for segmentations',
    license='MIT',
    keywords='CNN medical imaging',
    author='Marco Reisert',
    url='https://bitbucket.org/reisert/patchwork',
    download_url='https://bitbucket.org/reisert/patchwork/src/master',
    install_requires=['tensorflow', 'nibabel', 'numpy', 'matplotlib'],
    packages=['patchwork'],
    python_requires='>=3'
)
