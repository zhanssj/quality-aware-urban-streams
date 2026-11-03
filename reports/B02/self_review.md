# Внутренняя проверка B02

Дата завершения: 2026-11-03. State: READY_FOR_REVIEW. Code commit: f500c01d6f64311bba743c61fdb86a6715758c23.
Выполнена основным Codex в этом блоке; это самопроверка, не независимое внешнее ревью и не APPROVED.

## Требования и результат

- N1: PredictionPlan задаётся независимо до CSV; exact expected grid и контекст run/scenario/wait
  проверяются до PASS. Пропавшие, лишние, заменённые или duplicate IDs и смешанный context дают ошибку.
  Missing prediction в существующей строке понижает coverage и общий scored intersection.
- T04: seeded calendars на общей временной сетке; natural missing не считается повторным drop;
  random 5/10/20%, single/group blocks 6/24, явные rates/denominators. Модели получают один календарь;
  truth хранится отдельно. Weighted train sample generator реализован с суммой весов 1 и train-only gate.
- T05: origin/arrival/cutoff/target end, tie<=cutoff, запрет незавершённых/целевых событий,
  late updates по event index, идемпотентные дубликаты, immutable snapshots, прогнозы без сообщений,
  waits0/5/15 с remaining lead60/55/45. Timers ожидания и вычислений раздельны.
- Все real модели/fallback/cohort frozen; нет автоматического refit или выбора параметров по ошибкам.
  Baseline manifest pinned; локальные artifacts проверяются до загрузки доверенной local модели.

## Исправления до окончательного code commit

1. Обнаружена возможность использовать primed prefix для backcast API до границы прогрева.
   Введён minimum_target: такой forecast отклоняется. Regression test
   `test_priming_cannot_be_used_for_a_forecast_before_its_history_boundary` прошёл.
2. Конфликтующий duplicate value теперь явно DataContractError; оригинальное событие сохраняется.
   Regression test `test_conflicting_duplicate_is_error_and_original_message_is_unchanged` прошёл.
3. Подготовка training mixture завершена до weighted FeatureFrame/TruthFrame generator,
   с train-only проверкой до чтения значений и explicit weights. Это подготовка API;
   обучение real augmented моделей не выполнялось.
4. 14 внешних reviewer checks B01 перенесены в автоматические tests с expected plan/context.
   Их архивные оригиналы в reviews/evidence/B01 не изменялись. В fixture метрик старые 5 строк
   дополнены шестой с missing truth для полного Cartesian grid; прежние scored errors сохранены,
   coverage стала 5/6 вместо .8. Новая fixture соответствует обязательному plan contract.

## Подтверждение

На чистом f500c01: N1 15 passed; T04 8 passed; T05 9 passed; replay e2e 7 passed;
full suite 158 passed, 1 skipped; ruff check/format exit0. Windows skip — longdouble не шире float64.
Synthetic tests без сети/исходных HDF5; отдельные fixtures обучают только synthetic модели.
Эти результаты не складываются: локальные gates входят в full suite.

Реальные read-only smoke и pilot завершились PASS. Все 281088 predictions на метод в full zero-control
равны B01 с max absolute difference0 при заданном tolerance1e-10. В 51 pilot case общий scored ID hash,
21504 predictions/20703 scored на метод, coverage100%, zero missing predictions.
Отдельный evaluate CLI для delay30_seed3101_wait15 дал точно тот же JSON метрик.
Итоговая stdlib-only проверка пересчитала SHA256/размеры 10 baseline, 60 smoke, 365 pilot outputs;
копии JSON совпали с local outputs. Проверка не загружала модели и не читала HDF5.
Source/config/tests/lock после tested code commit не менялись; отчётные файлы идут следующим commit.

## Оставшиеся ограничения

Незавершённых требований N1/T04/T05 не обнаружено. Это ограниченное наблюдение по коду,
synthetic tests и frozen pilot, а не доказательство отсутствия всех возможных ошибок.
Новый Linux/remote CI не проверялся; реальный pilot независимо другим участником не повторялся.
Units/time semantics unverified; до подтверждения нельзя публиковать физические выводы.
Perturbation seeds не независимые города; нет CI/bootstrap, ablation или выводов о значимости.
Warmup/performance study/p50/p95 и end-to-end latency относятся к следующим задачам.
Получение событий разделяется между waits; receive CPU — incremental work одного replay,
не три независимо измеренных сервиса. Batch inference не гарантирует публикацию в real-time deadline.
HDF5 2025, T06–T08, новый аудит и real augmented fit не выполнялись.
