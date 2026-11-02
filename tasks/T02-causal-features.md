# T02: Причинные признаки и временное разделение

Статус: TODO

Предусловия: T01 принят.

## Реализовать

Sample identity: station_id + origin_index + target_index. Split назначается по target.
Feature builder из доступной истории: лаги 1/2/3/6/12/24 относительно target; rolling 6/24; качественные маски, age, missing ratios, has_history.
Один common feature pipeline для Standard и Quality-aware; quality block добавляется отдельно.
Train-only fallback и явное состояние fitted preprocessors. Истина не входит в predict API.
Прогнозируемые origins формируются по расписанию даже без новых сообщений. Сетка сохраняется.
Неполная история допустима; target без source_valid не восстанавливается для scoring.
Надёжно обрабатывать переходы годов, первые наблюдения и большую пустоту.

## Тесты приёмки

Future-invariance при фиксированном fitted state; изменение target не меняет features; скрытые значения не меняют rolling; train fit не читает val/test.
Валидационный target не попадает в train даже при предыдущем тренировочном origin.
Пустота 100 шагов не превращается в один шаг; age/ratios корректны на ручном fixture.
Нет истории -> согласованный fallback и has_history=false.
Standard и Quality-aware имеют одинаковый common block и sample IDs.

## Вне задачи

Новые исследования качества исходного набора и подбор модели по 2025.

## Передача результата

Заполни `handoffs/T02.md` по шаблону, приложи реальные команды, exit codes и evidence.
Обнови STATUS до READY_FOR_REVIEW или BLOCKED. Не ставь APPROVED, не начинай следующую задачу автоматически.
Критерии научной корректности из AGENTS.md и протокола обязательны, даже когда не повторены ниже.
