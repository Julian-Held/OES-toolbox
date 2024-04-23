import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="OES-toolbox",
    version="0.2.2",
    author="Julian Held",
    author_email="julian.held@umn.edu",
    license='MIT',
    platforms=['any'],
    description="Tool for low-temperature plasma optical emission spectroscopy.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mimurrayy/OES-toolbox/",
    packages=setuptools.find_packages(),
    install_requires=[
          'owlspec>=0.2.2', 'moose-spectra>=0.2.0', 'pyqtgraph>=0.13.1', 'PyQt6>=6.5.0',
          'sif-parser>=0.3.0', 'file-read-backwards>=3.0.0', 'matplotlib>=3.7.0'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10.9',
    entry_points = {'gui_scripts': ['OES-toolbox = cost_power_monitor:main']}
)
