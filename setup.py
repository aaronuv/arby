from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="arby",
    version="0.1",
    # description="Short description",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Aaron Villanueva",
    # author_email="your email",
    # url="https://gitlab.com/aaronuv/rbpy",
    py_modules=["arby",],
    install_requires=["numpy>=1.6", "scipy>=0.16"],
    test_suite="tests",
)
