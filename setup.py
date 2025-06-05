from setuptools import setup, find_packages

setup(
    name='specml',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},  # Tell setuptools where to look for the modules
    install_requires=[         # List dependencies here
        'numpy',
        'scikit-learn',
        'matplotlib',
        'torch',
        # Add more 
    ],
    author='Krzysztof Dziuba',
    description='Spectroscopy Machine Learning Library',
    license='MIT',
    python_requires='>=3.7',
)
