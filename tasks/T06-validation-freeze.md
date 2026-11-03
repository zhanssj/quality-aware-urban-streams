# T06: Матрица validation, ablation, benchmark и freeze

Статус: READY_FOR_REVIEW, 2026-11-03. Назначено владельцем после приёмки и merge B02.
Code commit: fca85909241ac7efd3546db00cf1a79f96b5dcd2.
Evidence, результаты и ограничения: [handoffs/T06.md](../handoffs/T06.md).

Предусловия: T05 принят.

## Реализовать и выполнить

Pilot 3 seeds, итоговая validation matrix 5 фиксированных seeds на согласованной группе.
Сравнить фиксированные варианты, определить training regime (clean/общая augmented mixture) без использования2025; результаты режимов разнести.
Ablation masks-only, age-only, full quality на фиксированном наборе сценариев.
Отдельный benchmark feature/inference/job latency, warmup, p50/p95, throughput, RSS, CPU/RAM/threads/batch. Не измерять только время fit вместо prediction latency.
Рассмотреть native-missing контроль без расширения scope на глубокие нейросети.
Документально уточнить доступные сведения единиц/часовых агрегатов; не делать ещё один полный аудит. Неразрешённое честно оставить unknown и ограничить claims.
Зафиксировать final cohort, модель, признаки, гиперпараметры, seeds, интервалы, метрики, шаблоны таблиц, protocol/config hashes в protocol_freeze.json.
Сохранить tests и complete evidence; решение открыть test принимает владелец отдельно.

## Тесты/критерии

Парные comparisons имеют одинаковые target/scenario. Bootstrap использует временные блоки, станции внутри блока не разрываются независимо.
Ни один tuned артефакт не использует2025. Все trial/regime отражены, плохие результаты не скрыты.
Benchmark не смешивает wait и compute. Расширение числа станций для нагрузки отмечено synthetic.
Test gate остаётся CLOSED. Открытый физический вопрос не маскируется статусом verified.

## Передача результата

Заполни `handoffs/T06.md` по шаблону, приложи реальные команды, exit codes и evidence.
Обнови STATUS до READY_FOR_REVIEW или BLOCKED. Не ставь APPROVED, не начинай следующую задачу автоматически.
Критерии научной корректности из AGENTS.md и протокола обязательны, даже когда не повторены ниже.
