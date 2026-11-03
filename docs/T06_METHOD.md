# T06: заранее фиксированный validation/freeze

Назначено владельцем 2026-11-03 после внешнего APPROVED B02 и merge main fb17f92.
Вначале N2: fingerprint один раз на календарь; регрессия проверяет число вычислений и равенство
признаков/labels/weights/hash. Независимые reviewer tests добавляются отдельно от архивных adapters.
Исторические B01/B02 results/models/cohort не перезаписываются.

## Матрица до результатов

Прежние 32 train-selected станции, fallback и параметры HGB из pinned B01 manifest.
Без расширения сети, поиска гиперпараметров, ранней остановки или fit на validation.
configs/t06.yaml закрепляется до training/validation. Pilot: первые 672 позиции2024,
первые3 seeds. Final matrix: все8784 позиции2024, 5 seeds4101–4105; train seeds2101–2103.
Сценарии/частоты/интервалы как B02: 11 сценариев; clean waits0/5/15, потери wait0,
delay/combined waits0/5/15. Всего51 pilot и83 final policy cases, каждый для обоих regimes.
Calendar один для всех методов/regimes/waits; history перед нарушаемым окном clean.
Truth общий; scheduled targets сохраняются, missing truth не восстанавливается.

## Модели и training regimes

Persistence плюс5 HGB: Standard(10 common), +masks(6 observed lag masks),
+age(один age), full quality(11 quality), native-missing control(10 common positions).
У native отсутствующие lag превращаются в NaN; rolling mean/std доступны только при полном
наблюдении соответствующих6/24 positions, иначе NaN. Это строгий complete-window NaN control,
не partial-window observed mean и не основной Standard/QA протокол заполнения.
Native не использует дополнительные quality predictors. Все5 HGB имеют один seed/params/labels/weights.

Clean-trained: Standard/full quality — исходные frozen B01 models; masks/age/native дополнительно
обучаются на точно тех же saved B01 train arrays/labels. Никаких изменений исторического baseline.
Augmented-trained: отдельные5 models; те же targets target_index%8==0 и source-valid labels.
Clean view weight.25 один раз; random10/group24/delay30 по.25, каждый разбит поровну между3
train seeds. 10 views всего; сумма весов для каждого исходного sample=1. Natural-invalid labels
исключаются одинаково из всех views; искусственная потеря входа label не удаляет.
Все fit завершаются до первого чтения numeric validation rows. Fallback не refit.
Сохраняются отдельные model hashes, fit plans, sample IDs/content hashes и training_mode.

## Хранение и оценка

Полная матрица хранит predictions в локальных compressed NPZ с явными model names, station IDs,
target indices и контекстом. Независимый PredictionPlan до каждого case определяет полный
Cartesian grid. Оценщик требует точного порядка/состава индексов, shape, context/plan hash;
пропавшая/лишняя/заменённая строка не может пройти проверку. NaN predictions понижают coverage;
общий source-valid scored intersection один для всех12 regime/model combinations.
Оригинальный CSV evaluator сохраняется; NPZ recompute CLI проверяется отдельно.
Scoring: MAE, RMSE, macro station MAE, scheduled/predicted/scored/coverage, per-station counts.
Каждый sample и prediction сохранён; NPZ не служит выборкой только лёгких/валидных targets.
Zero-control clean Standard/full/Persistence сравнивается с B01 для всех scheduled positions.

Evaluators отдельно выделяют artificial disruption последнего входа на cutoff, recovery и ordinary:
disruption=dropped или delay>wait для source-valid lag1; recovery=не disruption, но disruption
в предыдущих24 positions этой станции. Ordinary — остальные, включая natural missing.
Срезы — только metadata оценщика; модели их не получают. Они образуют полное непересекающееся
разбиение каждого case, без применения будущих событий.

Парный non-overlapping block bootstrap: blocks168 positions, станции блока вместе;
1000 replicas, PCG64 seed5101, 2.5/97.5 percentile, MAE(method)-MAE(reference).
Последний неполный блок сохраняется; суммы ошибок/числа scored взвешиваются по числу observations,
а не среднее средних блоков. Сравнения QA–Standard/QA–Persistence/masks–Standard/age–Standard,
native–Standard отдельно по regime/case; augmented–clean по каждому методу. Seeds нарушений
не новые независимые города. Не заявлять значимость без парного interval конкретного comparison.

## Выбор и freeze

До test выбирается один training regime для основной пары Standard/full quality. Правило:
mean MAE двух методов, затем mean по seeds и waits внутри каждого scenario, затем equal mean
по11 scenario names. Меньший score выбирается; равенство оставляет clean.
Все per-case результаты обоих regimes публикуются независимо от выбора. Ablations не используются
как скрытый перебор гиперпараметров. Итоговая test-постановка сохраняет Standard/QA/Persistence,
5 заранее заданных независимых test seeds6101–6105, тот же набор сценариев/ожиданий/32 stations,
block size168 и метрики. Test interval — все позиции2025 после отдельного разрешения T07.
protocol_freeze.json фиксирует selected regime, hashes source/config/model/cohort/fallback/lock,
feature names, conventions, seeds, intervals, таблицы и ограничения. Gate остаётся CLOSED,
owner_test_authorization=null. Freeze diagnostic/file_units; unknown units/time не объявляются verified.

## Benchmark

Отдельная команда benchmark; одна фиксированная нагрузка первых168 validation positions,
delay30 seed4101 wait0, обе training regimes/все6 методов, batch168 и2 threads.
2 warmups исключаются,7 measured repeats; одинаковые данные/модели/оборудование.
Feature construction, receive processing, model inference, transforms, serialization и compute-job
wall измеряются отдельно. Timed compute job: создание calendar/observer, delivery, features,
predict, запись локального NPZ; исключает исходное чтение HDF5/моделей, обучение и scoring.
Полное время CLI с integrity/source/model loading сохраняется отдельно от repeated compute job.
Метрики: p50/p95, samples/s, sampled peak RSS, CPU/RAM/versions/threads/batch и raw repeats.
Число output predictions не выдаётся за throughput независимых observations.
Simulated wait/delay не прибавляются к реально измеренным секундам. Нет real-time sleep,
копий станций, обещаний публикации в deadline или end-to-end broker latency.

## Единицы и время

Локальная dataset card pinned revision bbdcdd257bc7aed69598d423ca034d03f7ce015e описывает hourly
concentrations; формулировка единиц перечисляет gases/CO и не удостоверяет PM2.5 каждого sensor.
Официальный README https://github.com/Star-Learning/AirQualityBench (прочитан2026-11-03)
тоже описывает hourly data/original concentration units, но называет data/mask/coords вместо
values/masks/station_ids текущего pinned compact файла. Это документация benchmark, не доказательство
конкретного timezone/часового агрегата, границы интервала или единиц каждого выбранного sensor.
Pinned dataset card: https://huggingface.co/datasets/xuxing123/AirQualityBench/blob/bbdcdd257bc7aed69598d423ca034d03f7ce015e/README.md
(web чтение недоступно; использована уже скачанная read-only копия и её исторический hash).
Поэтому units_verified=false, interval_end convention остаётся симуляционной; claimed coverage
документации не повышается молча до verified physical result. Дополнительный полный аудит не нужен.
