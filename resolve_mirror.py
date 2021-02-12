# pylint: disable-all
import urllib.request
import urllib
import os

packages = "pypandoc cython optuna poetry"


def fix():
    net = "free"

    try:
        urllib.request.urlopen("http://mirror.ca.sbrf.ru")
        net = "ca"
    except urllib.error.URLError:
        pass

    try:
        urllib.request.urlopen("http://mirror.sigma.sbrf.ru")
        net = "sigma"
    except urllib.error.URLError:
        pass

    pip_install = "pip install -U "
    if net != "free":
        url = f"http://mirror.{net}.sbrf.ru/pypi/simple"
        host = f"mirror.{net}.sbrf.ru"
        pip_install += f"--index-url {url} --trusted-host {host} "

        with open("pyproject.toml") as file:
            txt = file.read()
        if txt.find("tool.poetry.source") == -1:
            with open("pyproject.toml", "a") as file:
                file.write(
                    f"""
[[tool.poetry.source]]
name = "private-pypi"
url = "{url}"
default = true
"""
                )
    os.system(pip_install + "pip")
    os.system(pip_install + packages)


if __name__ == "__main__":
    fix()
