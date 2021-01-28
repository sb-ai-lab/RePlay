# pylint: disable-all
import urllib.request
import urllib
import os

packages = "poetry pip pypandoc cython optuna"


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

    if net == "free":
        os.system(f"pip install {packages}")
    else:
        url = f"http://mirror.{net}.sbrf.ru/pypi/simple"
        host = f"mirror.{net}.sbrf.ru"
        command = f"pip install --index-url {url} --trusted-host {host}  -U {packages}"
        os.system(command)

        with open("pyproject.toml", "a") as file:
            file.write(
                f"""
[[tool.poetry.source]]
name = "private-pypi"
url = "http://mirror.{net}.sbrf.ru/pypi/simple/"
default = true
"""
            )


if __name__ == "__main__":
    fix()
