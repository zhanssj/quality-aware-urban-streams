# Архитектура исследовательского ядра

Целевая структура, которую Codex создаёт по задачам:

```text
src/aqstream/
  cli.py
  config.py
  data/             # HDF5 read-only, сетка, IDs, source validity
  features/         # причинная история, общие и quality features
  replay/           # virtual clock, drops, delivery, scheduled forecasts
  models/           # persistence, standard, quality, model manifest
  evaluation/       # truth lookup, metrics, paired analysis
  artifacts/        # run manifests, hashes, report exports
 tests/
  unit/
  integration/
  fixtures/         # только крошечные синтетические данные
 configs/           # версионированные configs; без абсолютных путей/секретов
 reports/           # небольшие итоговые CSV/JSON/Markdown
 artifacts/         # большие prediction/model файлы вне Git
```

## Границы

Reader знает значения и source mask. Replay публикует только доступную входную часть.
Feature builder получает только observer state/cutoff, а не truth store и будущий календарь потерь.
Fit pipeline получает train labels отдельно. Predict принимает только features.
Evaluator присоединяет truth после прогноза и считает метрики.
Один feature builder используется offline и будущим worker; второй реализации на Java не делать.

## Память

Читать HDF5 блоками по выбранным станциям/времени. Не загружать все годы и все каналы в RAM.
Для первого сравнения использовать PM2.5 и bounded cohort. Сохранение промежуточных Parquet по конфигурации; не создавать миллион мелких файлов.
Список 669 - кандидат на расширение после измерения ресурсов, не требование первого smoke-test.

## Реализованные команды B01/B02

```text
python -m aqstream prepare --config configs/smoke.yaml
python -m aqstream train --config configs/baseline.yaml
python -m aqstream evaluate --predictions artifacts/<run>/predictions_validation.csv --out artifacts/<run>/recomputed.json
python -m aqstream export-review --out exports/B01-review.zip
python -m aqstream replay --config configs/b02_pilot.yaml
```

Не создавать интерфейс-команды, которые печатают успех без реализации. Не реализованная команда должна отсутствовать либо завершаться явной ошибкой.
Benchmark пока не реализован. Оценщик проверяет expected plan (явный --plan или файл рядом с CSV).
Точные имена можно уточнить в T01, обновив документы и handoff.

## Запись прогноза

`run_id, model_id, station_id, origin_index, target_index, cutoff, prediction, has_history, scenario_id, wait, processing_ms`.
Truth и source validity присоединяются оценщиком; в published prediction использовать отдельные поля, не давать их predict API.
До подтверждения физической семантики сохранять `time_semantics=provisional_interval_end` и `units_verified=false`.

## Run manifest

`run_id, code_commit, dirty_tree, task_id, config_sha256, dataset_release, input_hashes, cohort_sha256, split_definition, feature_spec, model_params, seeds, scenario_hash, python_version, dependency_lock_sha256, hardware, commands, exit_codes, started_at, finished_at, status, artifacts`.
Отсутствующие значения отмечаются null/NOT_RUN, а не заполняются выдумками.

## Будущее расширение T08

Replay producer -> Kafka -> Python worker -> predictions/results.
PostgreSQL: метаданные запусков; object storage: артефакты. Spring: опциональный control API, не дублирование feature logic.
Никакой инфраструктуры T08 до приёмки ядра. Не делать HA и deployment ради конференционного MVP.
