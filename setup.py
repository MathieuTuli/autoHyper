from setuptools import setup
from glob import glob

setup(
    name='autohyper',
    use_scm_version=True,
    packages=[''],
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    python_requires='~=3.7',
    extras_require={
        'docs': [
            'sphinx',
            'sphinx_rtd_theme',
            'recommonmark',
        ],
    },
    dependency_links=[
    ],
    setup_requires=[
    ],
    license='License :: Other/Proprietary License',
    author='',
    author_email='',
    description='Python package for autoHyper',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
    ],
    scripts=glob('bin/*'),
)
