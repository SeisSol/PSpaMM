import setuptools

with open("pypspamm/VERSION", "r") as fh:
    current_version = fh.read().strip()

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    install_requires = [s.strip() for s in fh.readlines() if s.strip() != ""]

setuptools.setup(
    name="PspaMM",
    version=current_version,
    license="BSD-3-Clause",
    author="Peter Wauligmann, Nathan Brei, Alex Puscas, David Schneller",
    author_email="david.schneller@tum.de",
    description="An inline assembly generator for sparse matrix multiplications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/pspamm/pspamm",
    python_requires=">=3.7",
    install_requires=install_requires,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "pspamm-generator = pypspamm.cli:main",
        ]
    },
)
