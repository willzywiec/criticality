from setuptools import setup, find_packages

setup(
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    package_data={
        "criticality_torch": ["data/extdata/*.csv", "data/extdata/*.pt"],
    },
)
