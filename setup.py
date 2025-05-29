from setuptools import setup

setup(
    name='cosmoemu_jax',
    version='0.1.0',
    description='Code for emulating 2x3pt statistics',
    url='',
    author='Pierre Burger',
    author_email='pierre.burger@uwaterloo.ca',
    packages=['cosmoemu_jax'],
    install_requires=[
        
        'gdown',
        'matplotlib',
        'blackjax',
        'tqdm',
        'getdist',
        
    ],
    extras_require={
        'cpu': [
            'numpy',
            'jax',
            'scipy',
            'optax',
        ],
        'gpu': [
            'numpy<2.0',
            'jax==0.4.20',
            'jax-metal==0.0.5',
            'scipy==1.11.4',
            'optax==0.1.7',
        ],
    },
    python_requires=">=3.10",
)