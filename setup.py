from setuptools import setup, find_packages

setup(
    name='fwl-zalmoxis',          # Project name
    version='0.1.0',                   # Version
    packages=find_packages(),          # Automatically find all packages
    install_requires=[                 # List core dependencies
        'numpy',
        'scipy',
    ],
    extras_require={                   # Define optional dependencies
        'docs': [
            'mkdocs',
            'mkdocs-material',
            'mkdocstrings',
            'markdown_include'
        ],
    },
    include_package_data=True,         # Optional: include other files like README.md, etc.
)
