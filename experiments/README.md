# Эксперимент: Замена Classification Head с `nn.Linear` на `SwitchBackLinear`

## Описание
В данном эксперименте производится замена слоя `nn.Linear` на слой `SwitchBackLinear`, который выполняет форвард-проходы в формате `int8`. Это позволяет ускорить обучение модели.

## Шаги для запуска

1. Скачайте Docker-образ:
    ```bash
    docker pull dmitryredkosk/bitsandbytes_recsys_clear
    ```

2. Перенесите содержимое файла `config_ml20_swichback.yaml` в основной файл конфигурации `config.yaml`

3. Запустите основной скрипт:
    ```bash
    python RePlay-Accelerated/main.py
    ```

## Проверка обновления слоя

Чтобы убедиться, что слой `SwitchBackLinear` успешно заменил `nn.Linear`, вы можете добавить вывод инициализированной модели в метод `run` файла `RePlay-Accelerated/replay_benchmarks/train_runner.py`:
```python
    def run(self):
        """Execute the training pipeline."""
        train_dataloader, val_dataloader, prediction_dataloader = (
            self._load_dataloaders()
        )

        logging.info("Initializing model...")
        model = self._initialize_model()

        print(model)
```
