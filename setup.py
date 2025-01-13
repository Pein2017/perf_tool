from setuptools import find_packages, setup

from perf_checker import __version__

setup(
    name="perf_checker",
    version=__version__,
    packages=find_packages(include=["perf_checker", "perf_checker.*"]),
    install_requires=[
        "mindstudio_probe>=1.1.1",
        "torch>=2.0.1",
        "torch_npu>=2.0.1",
    ],
    author="Pein",
    description="Performance monitoring tools for Deep Learning inference pipelines",
    python_requires=">=3.8",
    package_data={
        "perf_checker": [
            "configs/*.json",
            "requirements.txt",
        ],
    },
    include_package_data=True,
)
