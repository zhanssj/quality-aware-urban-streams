# Quality-Aware Urban Streams

Экспериментальная основа статьи о прогнозировании PM2.5 при неполных и запаздывающих входах.
Реализованы T01 и блок B01 (T02 + T03): read-only loader, причинные признаки, три baseline,
train-only preprocessing и общий оценщик validation. Статус и evidence: [STATUS.md](STATUS.md),
[handoffs/B01.md](handoffs/B01.md). План: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
В B02 добавлены expected-plan проверка полноты CSV, общие loss calendars и virtual delivery/wait replay
с frozen моделями B01. Метод и команды: [docs/B02_METHOD.md](docs/B02_METHOD.md), итог: [handoffs/B02.md](handoffs/B02.md).

## Установка

Проверенное локальное окружение: CPython 3.14.7, Windows. Точные версии — `requirements.lock.txt`.
Нужен Python 3.14 или новее; CI описан для Python 3.14 на Windows/Linux.
Для нового окружения Windows PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
$env:AQ_DATA_ROOT = (Resolve-Path .\airqualitybench_tools\data).Path
.\.venv\Scripts\python.exe -m aqstream prepare --config configs/smoke.yaml
```

Linux:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
export AQ_DATA_ROOT="$PWD/airqualitybench_tools/data"
.venv/bin/python -m aqstream prepare --config configs/smoke.yaml
```

На текущей машине можно использовать уже существующий `airqualitybench_tools/.venv` вместо нового
окружения, установив те же зависимости и editable package. Само окружение в Git не входит.
`.env.example` документирует переменные; CLI читает окружение процесса и не загружает `.env` автоматически.

`prepare` сейчас проверяет только загрузчик: 2–4 станции, не более 168 строк одного тренировочного года.
Результат — небольшой manifest в `artifacts/T01-*/`, без копирования исходных значений.
`AQ_ARTIFACT_ROOT` меняет папку результатов (по умолчанию `artifacts`).
Доступны `python -m aqstream ...` и установленная команда `aqstream ...`.
`configs/baseline.yaml` — фиксированный bounded B01 pilot; запуск через `prepare` завершается ошибкой.
Работают `train --config`, `evaluate --predictions --out`, `export-review --out`.
Работает `replay --config configs/b02_smoke.yaml` (затем `b02_pilot.yaml`). `benchmark` пока отсутствует.
Новый evaluator требует независимый expected plan: `--plan` либо evaluation_plan.json рядом с CSV.

T06: `validate --config configs/t06.yaml` обучает отдельные ablation/augmented модели только на
train и выполняет pilot/final validation, bootstrap, benchmark и diagnostic freeze.
Полная матрица сохраняет все predictions локально в NPZ; пересчёт:
`evaluate-tensor --predictions <case>/predictions.npz --truth <run>/truth.npz --config configs/t06.yaml --out <out>.json`.
Отдельный benchmark:
`benchmark --run <T06-run> --manifest-sha256 <SHA256 manifest.json> --out <new-output-directory>`.
Протокол и границы измерений: [docs/T06_METHOD.md](docs/T06_METHOD.md).
T07/2025 остаётся закрытым до отдельного разрешения владельца после внешнего ревью freeze.

## Локальные проверки без сети и исходного датасета

После установки зависимостей:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m ruff format --check src tests
```

На Linux заменить путь интерпретатора на `.venv/bin/python`. Fixtures создаются во временных каталогах.
CI запускает только эти проверки и CLI help; HDF5, credentials и доступ к dataset ему не нужны.
Установка зависимостей в новом окружении требует package index или заранее подготовленных wheels.

## Данные и исследовательские ограничения

- Локальные HDF5 уже находятся в `airqualitybench_tools/data`. В Git остаётся только их существующий
  `download_manifest.json`: revision и ранее рассчитанные SHA256. Полные файлы повторно не скачивались.
- В `airqualitybench_tools` сохранены исходные инструменты и ранее полученные CSV/JSON/Markdown аудитов.
  Архивы ZIP, окружение, raw data, metadata/scaler и крупные результаты исключены из Git.
- `verification_status.json`, `README_RU.md` внутри legacy tools и `PACKAGE_CHECKS.json` описывают
  прошлые сеансы/исходный пакет. Они не являются статусом текущего кода или подтверждением новых запусков.
- Вход доступен только при `(mask == 1) & isfinite(value)`. Валидные нули и конечные отрицательные
  значения сохраняются. Скрытые числа заменяются NaN до передачи клиенту. Сетка строк не сжимается.
- Train 2021–2023, validation 2024, test 2025. Reader отвергает 2025 до открытия файла;
  T01 CLI дополнительно ограничен train. Разрешение T07 и freeze пока отсутствуют.
- Физические единицы и смысл временного интервала не подтверждены. Loader smoke не является
  результатом прогноза для статьи. См. [протокол](docs/EXPERIMENT_PROTOCOL.md).

Работа ведётся в отдельных `codex/Txx-*` ветках. `main` обновляется после независимого ревью владельцем.
Материалы текущего блока для ChatGPT Pro: [handoffs/B02.md](handoffs/B02.md).
