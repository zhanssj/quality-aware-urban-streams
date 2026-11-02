# Handoff Txx

## Идентификация

- Task: Txx
- State: READY_FOR_REVIEW / BLOCKED
- Repository / branch / PR: <actual or NOT_SET>
- Base commit: <actual SHA>
- Code commit tested: <actual SHA; before a separate report-only commit if necessary>
- Dirty working tree during run: <true/false and details>
- Report commit: <optional; can be supplied outside this file>

## Что выполнено

<Конкретные изменения и пути файлов. Без самооценки «всё идеально».>

## Проверки

| Команда | Входы | Результат PASS/FAIL/NOT_RUN | Exit code | Лог/артефакт |
|---|---|---|---|---|
| <actual command> | <synthetic/local> | NOT_RUN | - | - |

Различать синтетические тесты, реальный smoke-test и полный эксперимент.
NOT_RUN с причиной не считать успешной проверкой.

## Evidence

- Config/hash:
- Dataset release/hash (не выдумывать, можно переиспользовать проверенный manifest):
- Cohort/hash:
- Run ID:
- Seeds и scenario hash:
- Число scheduled / predicted / scored примеров:
- Метрики: <actual only; NOT_RUN if not measured>
- Оборудование, лимиты, версии:
- Время подготовки признаков / inference / ожидание данных: <раздельно>

## Научные инварианты

<Какие защищены тестами; какие ещё не проверены.>

## Отклонения и ограничения

<В том числе неизвестные единицы/временная семантика, использованные допущения.>

## Известные проблемы

<Конкретные проблемы, а не скрытие под «готово».>

## Предложение следующего шага

<Одна задача или исправления. Не начинать без назначения.>
