# Передача T01 в ChatGPT Pro

Дайте проверяющему доступ к repository и ветке `codex/T01-foundation-loader` после её публикации
либо приложите файлы/снимок diff. Локальную ветку ChatGPT через GitHub не увидит.
Точный commit кода и результаты проверок находятся в `handoffs/T01.md`.

Текст для чата:

> Проверь T01 проекта zhanssj/quality-aware-urban-streams на ветке codex/T01-foundation-loader.
> Сначала прочитай AGENTS.md, PROJECT.md, docs/EXPERIMENT_PROTOCOL.md, tasks/T01-foundation-loader.md
> и handoffs/T01.md. Сопоставь code_commit из handoff с реальным кодом и diff от 6687db6.
> Проверь read-only loader, source_valid, выбор station IDs и PM2.5 по имени, chunking,
> сохранение пустых строк, закрытый доступ к 2025, synthetic tests, lock-файл и Git exclusions.
> Используй реальные evidence в reports/T01. Различай чтение логов и независимый повтор.
> Дай APPROVED / CHANGES_REQUESTED / BLOCKED с конкретными файлами и причинами.
> Не начинай T02 и не пиши Results: обучение и модельная оценка пока NOT_RUN.

Автоматического канала в Pro-чат нет. Результат ревью перенесите в `reviews/T01.md`
или сообщите Codex вместе с commit. После принятия отдельно назначается T02.
