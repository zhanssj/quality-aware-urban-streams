# B02 evidence

Передача 2026-11-03: **READY_FOR_REVIEW**, не внешнее APPROVED.
Code commit synthetic checks и real runs: `f500c01d6f64311bba743c61fdb86a6715758c23`.
Полный результат и условия: [handoffs/B02.md](../../handoffs/B02.md).
Реальные прогоны выполнены 2026-11-02 с чистой рабочей копией; итоговая byte/hash сверка — 2026-11-03.

| Файл | Происхождение / назначение |
|---|---|
| verification.json | Реальные argv/exit codes, synthetic checks, real runs, пересчёт CSV и итоговая проверка всех artifact hashes |
| N1_gate.txt / T04_gate.txt / T05_gate.txt | 15 / 8 / 9 tests, exit 0; эти tests также входят в full suite |
| synthetic_e2e.txt | 7 tests полного replay pipeline на synthetic fixtures, exit 0 |
| pytest.txt | Все tests: 158 passed, 1 Windows longdouble skipped; exit 0 |
| ruff_check.txt / ruff_format.txt | Check и format --check, exit 0; 35 files formatted |
| smoke.txt / smoke_summary.json | 8 policy cases, первые 168 validation steps; summary — точная копия local manifest |
| pilot.txt / pilot_summary.json | 51 pilot policy case + 1 full zero-control; summary — точная копия local manifest |
| pilot_metrics.json | Точная копия local summary_metrics.json; все 51 per-case results и full control |
| zero_control.json | Извлечённый full zero-control result; все 281088 predictions на метод совпали с B01 |
| metrics_table.csv | 153 строки case × model: per-seed MAE/RMSE/counts, calendar hash, фактические drop/delay rates |
| seeds_summary.csv | 57 строк scenario/wait/model: арифметическое среднее метрики по 3 perturbation seeds; clean n=1 |
| self_review.md | Внутренняя самопроверка, исправления и границы проверки |

Не складывать counts разных сценариев как независимые observations. В каждом pilot case 21504
scheduled/predicted, 20703 scored, 801 naturally missing truth; один и тот же target grid.
Среднее RMSE по seeds в seeds_summary.csv не является RMSE объединённого CSV.
Full zero-control охватывает 8784 validation steps; pilot clean — первые 672. Их ошибки относятся
к разным временным объёмам, хотя модели, cohort и правила признаков совпадают.

Raw HDF5, values, models, predictions CSV и calendar NPZ не включены в Git/review ZIP.
Локальные run folders: `artifacts/B02-20261102T114608Z-7d38a7c8` и
`artifacts/B02-20261102T114700Z-a0e807a3`. Пилотный manifest перечисляет 365 outputs
с размерами/SHA256 (288168776 bytes); итоговая независимая сверка записана в verification.json.
Оригинальные 10 artifacts B01 проверены и неизменны, включая модели, cohort и predictions.
Hashes HDF5 — исторические из B01 manifest; полный аудит/повторное полное хеширование HDF5 не проводились.

Ошибки diagnostic/file_units, units_verified=false. Реальная временная семантика не подтверждена.
Virtual delay/wait/remaining lead отделены от wall-clock computation; timing scopes — в B02_METHOD.
Нет статистической значимости, T06 ablation/benchmark, real augmented fit или модельной оценки 2025.
Новый GitHub CI/Linux не наблюдался. Review bundle: `exports/B02-review.zip`, внутренняя
таблица `bundle_sha256.json`; outer hash хранится рядом в `.zip.sha256`, вне собственного payload.
