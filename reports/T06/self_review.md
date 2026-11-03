# Самопроверка T06

2026-11-03; code `fca85909241ac7efd3546db00cf1a79f96b5dcd2`; READY_FOR_REVIEW.
Выполнил основной Codex. Это внутренняя самопроверка, не независимое внешнее APPROVED.

## Проверенные требования

- N2 исправлен до augmented fit: fingerprint вынесен из record/цикла кадров; count regression
  проверяет один вызов на calendar, прежние IDs/weights/features/truth/hashes сохраняются.
  28 внешних B02 array/CSV reviewer tests перенесены и отформатированы; archived adapters не попали в src.
- До результатов commit a9d6bac объявил32 stations, полный validation2024,5 seeds,regimes/weights,
  ablations/native control, scopes, bootstrap, selection и freeze. Параметры по результатам не менялись.
- Clean Standard/fullQA сохраняют исходные B01 estimators. Три дополнительных clean и5 augmented
  estimators имеют одинаковые HGB params/seed; тот же source-valid label order/target_stride8.
  10 augmented views дают вес≈1 на исходный sample; total62338.00000000001 — float rounding,
  не новый независимый размер выборки. Label equality проверена при сборке каждого view.
- Все fit завершены до validation numeric reads. Synthetic spy отказывает при раннем2024 или2025.
  Ни preprocessing, ни cohort, ни fitting threshold не обучаются на validation/test.
- Модели принимают только arrays common/quality/persistence. Event calendars, slice labels,
  target truth и будущая доставка не передаются predict. Оценка/CI выполняется после prediction.
- Каждый NPZ сохраняет все scheduled positions; отдельный plan и fixed model order проверяются
  до metrics PASS. Regression tests отвергают missing row/replaced/duplicate IDs/mixed context.
  Missing prediction понижает coverage/общий intersection; naturally missing target не заполняется.
- Bootstrap парный по времени, все станции блока вместе; partial final block взвешен по counts.
  Disruption/recovery24/ordinary используют только настоящий/pрошлый artificial input state.
  Разбиение полное, с явными знаменателями, не model features.
- Native NaN control определён явно: observed lags; rolling только complete windows, иначе NaN.
  Это отдельный контроль, не незаметная смена заполнения в Standard/QA.
- Benchmark отделяет warmup, measured repeats, feature/receive/transform/inference/serialization/job.
  Полный subprocess timer дополнительно покрывает Python startup/loader; virtual wait отдельно.
- Freeze сохраняет выбранный clean и все hashes/params/feature names/seeds/slices/table templates.
  Gate CLOSED, owner authorization null; код2025 не открывает.

## Исправления и проверки до реального запуска

N2 и формат приложенных reviewer tests исправлены до code commit. При самопроверке добавлены
ограничение числа train seeds, запрет выбора regime без конечной scored MAE и отказ при изменённой
модели benchmark до unpickle. Synthetic tests подтверждают эти условия/invalid truth/план NPZ.
Первый T06 synthetic pass:20 tests; после дополнительных проверок22. Итог на чистом code commit:
N2/reviewer37, T06 e2e22, full209 passed+1 Windows longdouble skip, Ruff check/format44files PASS.
Предварительные Ruff замечания перенесённых tests исправлены; окончательные логи — в reports/T06.

## Реальные подтверждения

Один real validation run, exit0:51 pilot и83 final cases для обоих regimes/всех variants.
Final:281088 scheduled/predicted и262904 scored на метод/case, coverage100%,18184 missing truth.
Zero-control сравнил все281088 predictions Persistence/Standard/QA с B01, max difference0.
Clean waits0/5/15 также равны друг другу по точным predictions.npy bytes у всех12 outputs,
в pilot и final. Отдельный evaluate-tensor CLI дал точный metrics JSON, включая CI.
Первый standalone benchmark CLI exit0; его7 measured repeats опубликованы полностью.
Итоговая сверка: все10 B01,365 B02 и839 T06 artifact sizes/SHA256 совпали; старые файлы неизменны.
Source/config/tests/lock после tested code commit не менялись. Итоговый commit — документация/evidence.

## Границы

Незавершённых обязательных требований T06 по этой самопроверке не обнаружено.
Это не доказательство отсутствия всех ошибок. Новый remote CI/Linux и независимый real repeat
другим участником не выполнялись/не наблюдались. Исторический внешний CI B02 сохранён как история.
Units/time UNVERIFIED; прежняя сеть32 stations и stride8 ограничивают выводы и не доказывают
репрезентативность/оптимальность. Пять perturbation seeds не независимые города.
95% CI pointwise/validation conditional, без family-wise correction и без post-selection test claims.
Не измерены CI wait15-minus0; wait tradeoff описывается наблюдёнными MAE и оставшимся lead.
Benchmark7 repeats/одна машина/один workload, не real-time SLA; job timer исключает source/model
loading/fit/scoring. Функциональный cli_wall_seconds исключает startup/final JSON write, полный
внешний subprocess wall записан отдельно. Нет broker end-to-end latency, synthetic station copies,
нового аудита, открытия HDF5 2025 или инфраструктуры T08. После handoff остановка для внешнего ревью.
