# Статус

Состояние проекта: B01 READY_FOR_REVIEW (T02 + T03 выполнены; ожидается внешнее ревью).
Новый полный аудит: не назначен и не нужен.
Текущий блок: B01 завершён. Gate T02 пройден до T03, общий handoff подготовлен; работа остановлена.
Remote repository: https://github.com/zhanssj/quality-aware-urban-streams.git.
Рабочая ветка: codex/B01-features-baselines от main 444f816. Main подтянут по указанию владельца.
Test gate 2025: CLOSED.
Physical units: UNVERIFIED.
Time semantics of published HDF5: UNVERIFIED; development convention documented.

| Задача | Статус | Code commit | Handoff | Review |
|---|---|---|---|---|
| B01 | READY_FOR_REVIEW | 0a70134e32d86c5216e6f42e9647adb780756c64 | handoffs/B01.md | PENDING (внешнее) |
| T01 | APPROVED (внешнее) | 42e3a4457b6beed6adb6a273d0353fb5f05e9ab4 | handoffs/T01.md | reviews/T01-R1.md; подтверждено владельцем |
| T02 | READY_FOR_REVIEW в B01 | 0a70134e32d86c5216e6f42e9647adb780756c64 | handoffs/B01.md | локальный gate PASS; внешнее PENDING |
| T03 | READY_FOR_REVIEW в B01 | 0a70134e32d86c5216e6f42e9647adb780756c64 | handoffs/B01.md | внешнее PENDING |
| T04 | TODO | - | - | - |
| T05 | TODO | - | - | - |
| T06 | TODO | - | - | - |
| T07 | TODO | - | - | - |
| T08 | DEFERRED_OPTIONAL | - | - | - |

Допустимые переходы: TODO -> IN_PROGRESS -> READY_FOR_REVIEW -> APPROVED или CHANGES_REQUESTED.
BLOCKED допустим с конкретной причиной и выполненной независимой частью задачи.
Codex не ставит APPROVED сам. Следующая задача запускается по отдельному указанию владельца.
Владелец явно назначил B01 = T02 + T03; между ними достаточно внутренних тестов T02.
После общего handoff остановиться; T04 и test-2025 не назначены.

## B01, 2026-11-02

PASS: 11 обязательных tests T02; 10 synthetic end-to-end tests; полный pytest 105 passed,
1 platform skip; ruff check/format; pip check; CLI; пересчёт CSV metrics совпал точно.
Внутренний reviewer-проход завершён, замечания исправлены. Это не внешнее APPROVED.
Real run B01-20261102T095148Z-df565b77: 32 станции, seed 2026, train 2021–2023,
62338 source-valid fit labels из 105120 заранее выбранных supervised positions.
Validation 2024: 281088 scheduled/predicted, 262904 общих scored targets; coverage 100%.
Диагностическая MAE в file units: Persistence 1.549977, Standard 1.908577, Quality-aware 1.902250.
Persistence лучше обеих HGB в данном фиксированном пилоте. Параметры по результатам не менялись.
Evidence: reports/B01/. Общий handoff: handoffs/B01.md. T04 и HDF5 2025 не запускались.
Новый GitHub CI/Linux: не наблюдался в локальном evidence.

## Исправление R1, 2026-11-02

PASS: оригинальные synthetic tests — 50 passed; приложенные review tests — 17 passed, 1 skipped;
дополнительные dtype contract tests — 17 passed; ruff check и format. Итого 84 passed, 1 skipped.
Windows longdouble не шире float64; соответствующий skip не считается PASS.
Новый code commit проверен на чистой рабочей копии. Evidence: `reports/T01/R1/`;
подробности и ограничения: `handoffs/T01.md`. Только безопасные dtype values принимаются в loader;
остальные дают DataContractError до чтения numeric blocks. Контракт обновлён.
Повтор реального smoke и новый Linux CI: NOT_RUN/не наблюдался в локальном evidence.
Полный аудит и чтение HDF5 2025 не выполнялись, T02 не начат.
Исходный `reviews/T01.md` сохраняет решение CHANGES_REQUESTED для старого reviewed HEAD.

## Проверки исходной реализации T01, 2026-11-02 (история до R1)

PASS: 50 synthetic tests; ruff check/format; CLI help; pip check; SHA256 контекста и committed blobs;
соответствие установленных версий lock; Git exclusions; main unchanged; diff whitespace.
PASS: реальный bounded loader smoke — 168 строк 2023 × 4 станции, 7 блоков, 656 доступных из 672.
Логи с exit codes и manifest: `reports/T01/`. Полная передача: `handoffs/T01.md`.
GitHub CI/Linux: NOT_RUN в локальном evidence. Обучение, causal features, replay и метрики моделей: NOT_RUN.
T02 не начат по выбору владельца: сначала ревью T01 в ChatGPT Pro.
