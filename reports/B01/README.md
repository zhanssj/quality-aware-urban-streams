# Evidence B01

Code commit: `0a70134e32d86c5216e6f42e9647adb780756c64`.
Real run: `B01-20261102T095148Z-df565b77`. Полный контекст: `handoffs/B01.md`.

- `verification.json` — реальные команды, exit codes и проверка пересчёта CSV metrics.
- TXT — фактический вывод соответствующих проверок; один комплект логов для всего блока.
- `validation_metrics.json` — небольшая точная копия metrics JSON пилота.
- `pilot_summary.json` — точная копия manifest с версиями, ресурсами, input provenance и hashes артефактов.
- `cohort.json` — выбор 32 train-selected станций, зафиксированный до обучения/сравнения.
- `self_review.md` — внутренний reviewer-проход и исправления.

Полные predictions (51355796 bytes), training sample IDs, train matrices и models
остаются в `artifacts/B01-20261102T095148Z-df565b77/`, вне Git и review bundle.
Исходные HDF5 не включены. Полные dataset hashes переиспользованы как historical provenance;
они не были заново проверены этим пилотом. Ошибки диагностические в file units.
Новый CI в этих локальных evidence не наблюдался; Windows skip longdouble не считается PASS.
Архив использует явный whitelist обязательных материалов; дополнительные CLI/pip логи доступны здесь.
