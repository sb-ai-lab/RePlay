# RePlay

Эта библиотека должна, как губка, впитать в себя магию из других
библиотек, посвящённых рекомендательным системам.

## Как присоединиться к разработке

```bash
git clone ssh://git@10.21.25.60:8878/ailab/replay.git
cd replay
python3 -m venv .
source ./bin/activate
pip install poetry
poetry install
```

## Как запустить сборку локально

Из виртуального окружения

```bash
./test_package.sh
```

Для регулярной проверки кода рекомендуется сделать

```bash
pre-commit install
```
