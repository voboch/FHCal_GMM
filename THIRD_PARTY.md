# Происхождение шаблона и материалов FHCal

Структура основана на [Blinorot/pytorch_project_template](https://github.com/Blinorot/pytorch_project_template),
ветка `main`, ревизия `600143298293fc2c17497e3becbf4f221d5506bb`.
Именованные конфигурации находятся в `src/configs`, точки
входа вынесены в `train.py` и `inference.py`. Архивное ядро и его численные
зависимости перенесены из FHCal. Для конфигураций используется PyYAML с
переопределениями скалярных полей `key=value`.

[Версия шаблона](TEMPLATE_SOURCE.json) ·
[MIT notice шаблона](third_party/template/LICENSE) ·
[Метаданные для цитирования шаблона](third_party/template/CITATION.cff)

Архивные эксперименты FHCal описаны в [протоколе](docs/experiment_protocol.md).
Экспорты сохраняют хеши входных данных, карту и настройки расчёта.
MIT notice относится к шаблону. Условия распространения исходных
данных, кода и материалов BM@N FHCal определяются их правообладателями.
