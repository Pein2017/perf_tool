from setuptools import find_packages, setup

setup(
    name="perf_tool",
    version="1.0.0",
    packages=find_packages(include=["perf_checker", "perf_checker.*"]),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pyyaml>=5.4.0",
        "torch>=2.0.1",
        "msprobe>=0.1.0",  # For precision monitoring
    ],
    entry_points={
        "console_scripts": [
            "perf_tool=perf_checker.monitor_script:main",
        ],
    },
    author="Pein",
    description="Performance monitoring tools for ML inference pipelines",
    python_requires=">=3.8",
    package_data={
        "perf_checker": ["configs/*.yaml", "configs/*.json"],
    },
    include_package_data=True,
)
