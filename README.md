# RePlay

### Установка
Для корректной работы необходимы python 3.6+ и java 8+. \

Клонируйте репозиторий RePlay: \
 в _sigma_:
```bash
git clone https://sbtatlas.sigma.sbrf.ru/stash/scm/ailab/replay.git
```
в _alpha_:
```bash
git clone ssh://git@stash.delta.sbrf.ru:7999/ailabrecsys/replay.git
```
Рекомендуется устанавливать библиотеку в виртуальное окружение. 
Если оно не создано, можно воспользоваться `install.sh`, который создаст окружение и установит туда библиотеку.
Если необходимо установить в уже имеющееся окружение, это можно сделать с помощью следующих команд из папки `replay`.
```
python3 resolve_mirror.py
poetry install
```

### Проверка работы библиотеки
Запустите тесты для проверки корректности установки. \
Из директории `replay`:
```bash
pytest ./tests
```

### Документация

Запустите формирование документации из директории `replay`:
```bash
cd ./docs
mkdir -p _static
make clean html
```
Документация будет доступна в `replay/docs/_build/html/index.html`

## Как присоединиться к разработке
[Инструкция для разработчика](README_dev.md)
