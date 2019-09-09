import pkg_resources
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sponge-bob-magic",
    version = pkg_resources.get_distribution("sponge-bob-magic").version,
    author="Шминке Борис Андреевич",
    author_email="Shminke.B.A@sberbank.ru",
    description="библиотека рекомендательных систем",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://sbtatlas.sigma.sbrf.ru/stash/projects/AILAB/repos/sponge-bob-magic",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.7',
)
