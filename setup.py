# setup.py
from setuptools import setup, find_packages

setup(
    name="NichGPDRL",
    version="0.1.0",
    packages=find_packages(),  # Automatically finds directories with __init__.py
    # packages=['your_package_name'], # Or explicitly list your packages
    install_requires=[
        # ... other dependencies ...
    ],
    # other metadata: author, description, url, classifiers etc.
)
