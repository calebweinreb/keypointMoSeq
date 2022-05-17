import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='keypointMoSeq',
    version='0.0.0',
    author='Caleb Weinreb',
    author_email='calebsw@gmail.com',
    description='Motion Sequencing (MoSeq) for keypoint data',
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.7',
    install_requires=[
        'jax>=0.3',
        'tfp-nightly[jax]',
        'numba',
    ], 
    url='https://github.com/calebweinreb/keypointMoSeq'
)
