# T06 evidence

State: **READY_FOR_REVIEW**, 2026-11-03. Code commit: `fca85909241ac7efd3546db00cf1a79f96b5dcd2`.
Общая передача: [handoffs/T06.md](../../handoffs/T06.md); условия: [T06_METHOD](../../docs/T06_METHOD.md).
Real run: `T06-20261103T064826Z-e9a79599`. Final = весь2024, 32 frozen stations, 5 validation seeds;
pilot = первые672 positions и3 seeds. В каждой phase общий календарь/target для12 regime-method outputs.

| Файл | Происхождение |
|---|---|
| verification.json | Команды, реальные exit codes, внешний CLI wall, equality recheck, sizes/SHA256 всех output inventories |
| summary.json | Побайтная копия manifest.json реального run; hardware/deps/resources/selection/control/artifact hashes |
| fit_plan.json | Побайтная копия train-only plan:10 views,623380 rows, total weight≈62338, hashes matrix/labels/weights/calendars |
| selection.json | Побайтная копия результата заранее объявленного выбора regime; выбран clean |
| matrix_metrics.csv | 996 rows:83 final cases ×12 regime/methods; все MAE/RMSE/macro/counts/rates/hash |
| pilot_metrics.csv | 612 rows:51 pilot case ×12 regime/methods, отдельная phase |
| slice_metrics.csv | 2988 rows: final disruption/recovery24/ordinary, явные scheduled/scored counts |
| paired_comparisons.csv | 1328 pointwise paired bootstrap results;1000 replicas,53 time blocks по168 positions, станции вместе |
| benchmark.json | Первый отдельный benchmark CLI после полного run:2 warmups,7 measured repeats,raw traces,p50/p95/throughput/RSS |
| N2_gate.txt / synthetic_e2e.txt | 37 /22 tests; входят также в full suite, не складывать counts |
| pytest.txt | 209 passed,1 Windows longdouble skipped; exit0 |
| ruff_check.txt / ruff_format.txt | PASS,44 files already formatted |
| real_validation.txt | Реальный validate CLI на чистом code commit; exit0 |
| self_review.md | Внутренняя самопроверка и границы результата |

Полные per-station metrics, NPZ predictions/truth, exact calendars и модели остаются локально
в ignored artifacts/. Manifest перечисляет839 files /1589558481 bytes. Оригинальные B01/B02
manifests и все их outputs проверены и не изменены. Полные HDF5 повторно не хешировались/не аудировались.
Набор scored IDs в final общий,262904 targets; в каждом method/case281088 predictions,
18184 missing truth. Разные policy/scenario rows не независимые новые observations.

Индексы/порядок/cohort/model names/context NPZ проверены против отдельного expected plan.
Естественно отсутствующий target не восстанавливается. При missing prediction все methods
score на общем intersection, coverage остаётся явной. CSV evaluator тоже сохранён.

Режимы и отрицательные результаты не отбирались по выгодности: опубликованы все12 outputs.
Average MAE/RMSE по seeds/scenarios требует явных весов; mean RMSE не равен pooled RMSE.
Balanced summary в handoff — equal mean по scenario names после mean по их seeds/waits.
CI pointwise, без multiplicity adjustment; нельзя объявлять общий family-wise эффект по1328 intervals.
Benchmark p50/p95 относится к одному batch/job из5376 positions с12 outputs;7 repeats не SLA.
Встроенный benchmark сохранён отдельно в local run/summary manifest; в benchmark.json используется
первый standalone CLI, без выбора более быстрых repeats. cli_wall_seconds — timer функции,
внешний полный subprocess wall приведён отдельно в verification.json.

[protocol_freeze.json](../../protocol_freeze.json) — diagnostic freeze, selected clean, test2025 CLOSED,
owner authorization null. Physical units и точная временная семантика UNVERIFIED; hourly metadata
не подменяет проверку единиц каждого sensor. T07/T08 и новый remote CI/Linux не запускались/не наблюдались.

Review export: `exports/T06-review.zip`,143 whitelist payloads с внутренним bundle_sha256.json;
внешний SHA256 — рядом в `.zip.sha256`. Payloads проверены против текущих repository bytes,
пути с данными/моделями/NPZ/секретами отсутствуют. Большие артефакты остаются у владельца локально.
