# B02: controlled losses, delivery and waiting

Назначен владельцем: N1 + T04 + T05. Один общий handoffs/B02.md, затем остановка.
T06–T08, новые full audits и HDF5 2025 не назначены. B01 принят и остаётся frozen baseline.

## N1: оценка относительно независимого плана

`PredictionPlan` хранит station IDs, global target range, target local offset, год 2024 и контекст
run_id/scenario_id/wait. Схема проверяет expected Cartesian grid, identity, local/year consistency,
дубликаты, контекст и (при наличии) clock metadata. Число строк должно равняться размеру ожидаемого
множества после проверки каждого ID. Missing/extra/replaced rows вызывают DataContractError до PASS.
`prediction_coverage` вычисляется по запланированным позициям; missing prediction внутри присутствующей
строки явно понижает coverage и общий scored intersection. Отсутствующий CSV record не скрывается.

Новые runs сохраняют evaluation_plan.json до predictions. Оценщик использует явный план или файл
рядом с CSV; без него оценка запрещена. Для исходного B01 (созданного до N1) replay сохраняет
baseline_evaluation_plan.json в новой B02 папке и повторно проверяет старый CSV. Сам B01 не изменяется.
Новые поля context/expected_plan_sha256 зависят от уникального run ID; reproducibility сравнивает
все numerical metrics/counts, исключая только runtime identity. Ядро MAE/RMSE не изменено.

## Состояния и модель времени

Source validity, artificial drop, доставленность и evaluator truth разделены.
Искусственные потери не меняют source-valid targets. Скрытые raw числа не публикуются observer.
Model predict получает только common/quality/persistence arrays: без truth, типа сценария или будущего календаря.
Feature kernel `frame_from_history` общий для B01 и delivery snapshots. Он реализует те же лаги,
forward-fill, rolling ddof=0, padding, masks, age и missing ratios, что принятый B01.

Строка i — условный интервал [i*60,(i+1)*60), interval_end=(i+1)*60.
Для target=t: origin=t*60, cutoff=origin+wait, target_end=(t+1)*60.
Доступны только завершённые к origin интервалы i<t, прибывшие arrival<=cutoff.
Интервал target запрещён даже при ошибочном early arrival. Незавершённые интервалы до interval_end
не принимаются. Clock monotone; tie arrival==cutoff видим, cutoff+epsilon невидим.
Wait 0/5/15 даёт remaining lead 60/55/45 **симуляционных** минут до конца одного target.
Последние две выдачи находятся уже внутри целевого интервала. Это не постоянный 60-minute horizon.
Реальные времена доставки и физическая семантика ряда не восстановлены.

DeliveryHistory получает только уже пришедшие события. Late updates исправляют историю только
для будущих snapshots; last-known выбирается по максимальному event index, не arrival order.
Duplicate (station,event_index) с тем же значением идемпотентен, конфликт вызывает ошибку.
Выданные feature snapshots — immutable copies; старые predictions не переписываются.
Расписание существует даже без сообщений. Прогретая история не используется для более ранних origins.
Физический inference/CSV write батчируется по замороженным snapshots: это offline simulation,
не гарантия публикации реальным сервисом в deadline. Frozen HGB.predict rowwise и invariant к batch size.

## Календари нарушений

Один календарь на event grid для всех методов и waits. Никакого RNG на overlapping feature window.
Scope inputs=[target_start-1,target_stop-1); truth=[target_start,target_stop). Последняя строка train
является первым возможным нарушаемым input на validation boundary. Более ранняя history clean и уже доставлена.

SeedSequence([seed,stream]) + PCG64: drop stream101, delay202, block303.
Random 5/10/20% — первые round(p*source_valid_count) индексов seeded permutation source-valid cells.
Blocks 6/24 steps: один полный блок в каждом bucket 168 steps, случайный start; одна station
или фиксированная группа ceil(.25*N), выбранная по ID без использования значений/ошибок.
Группа не означает физически соседние станции или установленную причину outage.
Уже естественно отсутствующие клетки не считаются новой искусственной потерей.
Delays: 10/30% недропнутых source-valid сообщений, uniform integer 1–30 simulated minutes.
Combined — random10 + delay30; streams раздельны. Rates имеют явные знаменатели:
source-valid inputs, все input cells, eligible messages после drop. Zero denominators обозначаются null.

Calendar JSON содержит spec/seed/blocks/statistics/source-valid hash/calendar hash;
локальные NPZ и changed_events.csv хранят точные drop/delay/arrival. Истина не копируется в event DTO.
Train mixture API выдаёт weighted global calendar views и train-only FeatureFrame/TruthFrame samples;
вес каждого исходного sample суммарно равен 1, включая clean. Train seeds 2101–2103 отделены от
validation 3101–3103. Augmented model fit в B02 **NOT_RUN**, все real модели frozen clean-trained.

## Заранее фиксированный pilot

`configs/b02_smoke.yaml`: первые 168 validation steps, один seed, 8 policy cases.
`configs/b02_pilot.yaml`: первые 672 steps, три stochastic seeds, 51 policy case:
clean waits0/5/15 (один zero calendar), losses wait0, delay/combined waits0/5/15.
Дополнительно full zero-loss/zero-delay validation 2024 wait0: каждый prediction/truth сравнивается
с сохранённым B01 CSV при tolerance1e-10. Clean pilot controls всех waits также проверяются против B01.
Scored target IDs одинаковы внутри pilot у всех методов/seeds/policies; full control — иной временной объём.
Baseline cohort/models/hyperparameters и старый natural-missingness результат не перезаписываются.
Manifest B01 закреплён SHA256 в YAML; model/fallback/cohort/CSV hashes проверяются до доверенной local unpickle.
Нет automatic refit при недостающих/несовместимых артефактах.

В памяти только PM2.5 32 станций за 2021–2024 (values ~9 MiB + bool mask); чтение chunks168.
Replay snapshots/arrays ограничены тем же cohort; batch168, 2 потока, sampled RSS guard2048 MiB.
Все большие predictions/calendars/models остаются локальными, в Git — summaries и hashes.

## Метрики и измерения

Все ошибки diagnostic/file_units, units_verified=false. Статистическая значимость/физические единицы
не заявляются; perturbation seeds не являются независимыми городами. Нет tuning по результатам.
Case metrics отдельны для каждого run/scenario/seed/wait; смешанный context отвергается.
Пересчитываются из CSV с планом. Сохраняются scheduled/predicted/scored, missing truth/prediction,
availability лаг1 и фактические доли потерь/задержек.

Per-event simulated delay, policy wait и физические timers раздельны. processing_batches.csv
содержит реальные forecast construction, incremental delivery processing и estimator inference seconds.
Delivery CPU относится к общему последовательному replay waits, не независимой latency трёх сервисов.
Forecast timer исключает receive/predict/CSV; inference timer исключает input stacking и IO.
Controlled workload wall включает integrity checks/source/replay, исключает final hashing/manifest write.
Нет real-time sleeps, warmup/performance study/p50/p95/end-to-end broker latency: это будущий T06/T08.

## CLI

```text
python -m aqstream replay --config configs/b02_smoke.yaml
python -m aqstream replay --config configs/b02_pilot.yaml
python -m aqstream evaluate --predictions artifacts/<B02>/cases/<case>/predictions.csv --plan artifacts/<B02>/cases/<case>/evaluation_plan.json --out artifacts/<B02>/cases/<case>/recomputed.json
python -m aqstream export-review --out exports/B02-review.zip
```

AQ_DATA_ROOT — существующая локальная папка HDF5, AQ_ARTIFACT_ROOT — локальные outputs.
Безопасная загрузка frozen artifact предполагает доверенные локально созданные B01 files и pinned manifest;
модели из review attachments/произвольного внешнего источника не используются.
