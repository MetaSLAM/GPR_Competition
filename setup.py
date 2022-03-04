
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gpr",
    version="0.0.1",
    author="MateSLAM",
    author_email="hitmaxtom@gmail.com",
    description="A tool box for general place recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MetaSLAM/GPR_Competition",
    project_urls={
        "Bug Tracker": "https://github.com/MetaSLAM/GPR_Competition/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3 License",
        "Operating System :: LINUX",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)