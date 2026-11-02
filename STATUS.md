# Статус

Состояние проекта: T01 READY_FOR_REVIEW. Код, проверки и handoff готовы; независимое ревью ожидается.
Новый полный аудит: не назначен и не нужен.
Текущая задача: T01. Приоритет: первый работающий исследовательский срез.
Remote repository: https://github.com/zhanssj/quality-aware-urban-streams.git.
Рабочая ветка: codex/T01-foundation-loader. Main не изменяется.
Test gate 2025: CLOSED.
Physical units: UNVERIFIED.
Time semantics of published HDF5: UNVERIFIED; development convention documented.

| Задача | Статус | Code commit | Handoff | Review |
|---|---|---|---|---|
| T01 | READY_FOR_REVIEW | 252fbb2f670eb9131e264d51adcb08e634ccba24 | handoffs/T01.md | PENDING |
| T02 | TODO | - | - | - |
| T03 | TODO | - | - | - |
| T04 | TODO | - | - | - |
| T05 | TODO | - | - | - |
| T06 | TODO | - | - | - |
| T07 | TODO | - | - | - |
| T08 | DEFERRED_OPTIONAL | - | - | - |

Допустимые переходы: TODO -> IN_PROGRESS -> READY_FOR_REVIEW -> APPROVED или CHANGES_REQUESTED.
BLOCKED допустим с конкретной причиной и выполненной независимой частью задачи.
Codex не ставит APPROVED сам. Следующая задача запускается по отдельному указанию владельца.

## Проверки T01, 2026-11-02

PASS: 50 synthetic tests; ruff check/format; CLI help; pip check; SHA256 контекста и committed blobs;
соответствие установленных версий lock; Git exclusions; main unchanged; diff whitespace.
PASS: реальный bounded loader smoke — 168 строк 2023 × 4 станции, 7 блоков, 656 доступных из 672.
Логи с exit codes и manifest: `reports/T01/`. Полная передача: `handoffs/T01.md`.
GitHub CI/Linux: NOT_RUN в локальном evidence. Обучение, causal features, replay и метрики моделей: NOT_RUN.
T02 не начат по выбору владельца: сначала ревью T01 в ChatGPT Pro.
