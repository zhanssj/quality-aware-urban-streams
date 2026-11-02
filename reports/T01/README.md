# Evidence T01

Проверенный code commit: `252fbb2f670eb9131e264d51adcb08e634ccba24`.
Проверки выполнены на чистой рабочей копии. Полная передача: `handoffs/T01.md`.

`checks.json` содержит реальные argv и exit codes; TXT — вывод соответствующих команд.
`loader_smoke.json` — точная копия небольшого manifest реального чтения 2023, без исходных значений.
`integrity.txt` — выполненные проверки SHA256 контекста (включая committed blobs), lock и Git exclusions.
Нормализация evidence: в checks JSON имя временного поля исправлено на `recorded_at_utc`,
поскольку timestamp был снят после окончания команд. Числа и результаты не изменялись.

Loader доступность и время чтения не являются модельными метриками или latency benchmark.
GitHub CI, Linux, обучение и test-2025 не считаются пройденными по этим логам.
