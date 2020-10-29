from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
arby_init_path = os.path.join(BASE_DIR, "arby", "__init__.py")

with open(arby_init_path, "r") as f:
    for line in f:
        if line.startswith("__version__"):
            _, _, ARBY_VERSION = line.replace('"', "").split()
            break


setup(
    name="arby",
    version=ARBY_VERSION,
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
