# Решения и открытые вопросы

## 2026-11-02: B02 назначен владельцем

- Объём B02: N1 (план sample IDs/run/scenario/wait), T04 (потери), T05 (доставка/ожидание).
  Одна ветка, внутренние тесты между этапами, handoffs/B02.md и остановка. T06–T08/2025 закрыты.
- База main ea9205c; перенесён docs commit с приёмкой B01. Baseline код/артефакты/гиперпараметры
  и список 32 станций не изменяются. Только frozen clean-trained evaluation; augmented fit не запускается.
- N1: обязательный expected plan (станции × target grid, run/scenario/wait), проверка каждого ID
  и полного множества до PASS. CSV без плана не считается проверенным расписанием.
- Доставка: interval_end=(event_index+1)*60; origin=target_index*60; cutoff=origin+wait;
  target_end=(target_index+1)*60. Доступны только event_index<target и arrival<=cutoff,
  interval_end<=origin. Wait 0/5/15, remaining lead 60/55/45 симуляционных минут.
- Late события обновляют только будущие snapshots, last-known выбирается по event index.
  Дубликаты по (station,event_index) идемпотентны; конфликт значения вызывает ошибку.
- Нарушения применяются к source-valid inputs в [target_start-1,target_stop-1), truth независимо.
  Более ранняя история clean и уже доставлена. Календарь общий для методов и waits, не per-window RNG.
- Random drop: точное round(p*source_valid_count) по seeded permutation, p=5/10/20%.
  Blocks: длительность 6/24 steps, один выбранный station или фиксированная группа ceil(25%*N),
  один полный блок в каждом 168-step bucket с seeded start; natural missing не считаются new losses.
- Delay: 10/30% недропнутых source-valid сообщений, round count без возвращения,
  uniform integer 1–30 simulated minutes. Отдельные RNG streams для losses/blocks/delays.
- Pilot: первые 672 validation steps (не календарные дни), seeds 3101/3102/3103,
  модель/32-station cohort B01. Clean controls waits 0/5/15; losses без задержек wait=0;
  delay и random10+delay30 — waits 0/5/15. До этого smoke 168 steps/один seed.
- Дополнительно полный zero-loss/zero-delay контроль validation 2024, wait=0: сравнить
  с исходным B01 CSV и метриками, tolerance 1e-10. Исходный B01 не перезаписывается.
- Training mixture API: weighted calendar views с явными весами, независимые train seeds 2101/2102/2103;
  augmented model training — NOT_RUN. До validation результатов не меняем конфигурацию.
- Ограничения: 2 потока, HDF5 chunks 168, prediction batches 168 шагов, sampled RSS guard 2048 MiB.
  В RAM только PM2.5 выбранных 32 станций за 2021–2024 (~10 MiB), без пяти лет/всей сети.
  Seeded calendars и event traces локально; большие predictions/models не входят в Git/bundle.

## 2026-11-02: внешняя приёмка и merge B01

- Владелец подтвердил APPROVED для B01 (T02 + T03) и выполнил merge в main.
  Main ea9205c47832a27f0331b6fe0f0eda405eeef6be имеет тот же полный tree, что reviewed bab6d90.
- Решение и исходные свидетельства сохранены в reviews/B01.md и reviews/evidence/B01/.
  APPROVED перенесён от внешнего проверяющего, не присвоен исполнителем самому себе.
- N1 не блокирует B01. В начале отдельно назначенного B02 сверить фактические sample IDs
  и run/scenario/wait с ожидаемым планом до итогового PASS; неполный/подменённый CSV не должен
  выглядеть как полное расписание. Тесты: удаление/замена строки, дубликаты, смешение контекстов.
- Текущий пилот, cohort и hyperparameters не переписываются. Недостаток evaluator не доказывает
  потерю реальных predictions. Дополнительные scripts ревью архивированы, не исполнялись при переносе.
- B02 не назначен текущим сообщением; модельные прогоны, T04/T05 и test-2025 не запускаются.

## 2026-11-02: B01 назначен владельцем

- T01 принят внешним ChatGPT-ревью на daa0a04 (reviews/T01-R1.md), решение подтверждено владельцем.
- Владелец слил T01 в main. Main 444f816 имеет тот же полный tree, что принятый daa0a04
  (squash merge, не ancestry); это проверено git diff. Рабочая ветка codex/B01-features-baselines.
- B01 = T02 + T03 выполняется без промежуточного внешнего APPROVED. До T03 обязательные
  тесты причинности T02 должны пройти. Это внутренний gate, не внешнее одобрение.
- Итог: один handoffs/B01.md, READY_FOR_REVIEW и остановка. T04–T08 и test-2025 не назначены.
- Протокол, train-only fit, отдельная истина, закрытый 2025 и запрет самовольного merge сохранены.
- В пилоте фиксируем 32 станции по train-списку, seed 2026; PCG64 choice без возвращения из
  лексикографически сортированных ID, итоговые ID сортируются. Никакого отбора по validation.
- Train fitting моделей ограничивается расписанием target_index % 8 == 0 по всем станциям;
  это value-independent subsample (около 105120 scheduled rows при 32 станциях), общий для методов.
  Медианы fit на всех доступных train значениях выбранных станций. Validation сохраняется полностью.
  Причина subsampling — bounded CPU/RAM pilot, не результат ошибок; sample IDs сохраняются до fit.
- Лимиты пилота: 2 вычислительных потока, HDF5 chunk 168 строк, prediction batch 168 шагов,
  до 120000 scheduled train rows, 2 GiB RSS guard. HGB: max_iter=60, max_leaf_nodes=15,
  learning_rate=0.1, l2_regularization=1.0, min_samples_leaf=20, early_stopping=false.
- У первой позиции target=0 origin=-1; нет истории, используется train-only fallback.
  Rolling с ddof=0 по причинно заполненным позициям, до начала сетки padding fallback.
  Missing ratios считают недоступный padding пропусками. Age=-1 при has_history=false;
  иначе origin_index-last_observed_index. Индексы глобальные по непрерывной сетке 2021–2024.
  Календарные признаки отключены, единицы UNVERIFIED, ошибки диагностические в units файла.

| ID | Статус | Решение |
|---|---|---|
| D01 | Перенесено из обсуждения | PM2.5, AirQualityBench, качество входов и запаздывание |
| D02 | Перенесено из обсуждения | Полные аудиты закончены; следующее действие - реализация |
| D03 | Перенесено из обсуждения | Train 2021–2023 / validation 2024 / test 2025 |
| D04 | Предлагается в этом плане | Общий репозиторий как канал ChatGPT–Codex; один task на ветку |
| D05 | Предлагается в этом плане | Сначала Python core, инфраструктура отдельно |
| D06 | Предлагается в этом плане | 32 фиксированные train-selected станции для первого bounded baseline, затем расширение |
| D07 | Предлагается в этом плане | Временная конвенция interval_end + delay до подтверждения источника |
| D08 | Открыто | Подтверждение единиц и совместимой подгруппы для публикационных pooled metrics |
| D09 | Открыто | Подтверждение изготовления HDF5: часовой агрегат или иная выборка |
| D10 | Решено владельцем 2026-11-02 | Remote: https://github.com/zhanssj/quality-aware-urban-streams.git; локальные HDF5: airqualitybench_tools/data; путь через AQ_DATA_ROOT |

Новые изменения оформлять отдельной записью: дата, причина, основание, затрагиваемые run/config, кто разрешил.
Открытые D08/D09 не блокируют синтетические тесты и индексный прототип. Блокируют неподтверждённые физические утверждения статьи.

## 2026-11-02: реализация T01

- Владелец назначил T01 с последующим ревью в ChatGPT Pro; T02 не назначен.
- Рабочая ветка `codex/T01-foundation-loader`, без merge в main.
- Совместимый существующий интерпретатор: CPython 3.14.7. Зависимости фиксируются по реально
  установленным версиям в `requirements.lock.txt`. CI на Python 3.14; Linux пока не запускался локально.
- `prepare` в T01 означает bounded loader smoke. Он не создаёт обучающие признаки.
  Baseline config — явный план T03, а не работающая команда.
- Сохранены материалы starter и legacy audit; исторические verification JSON не считаются текущим evidence.
- 2025 закрыт непосредственно в reader; доступ будет расширен отдельным изменением T07 после freeze/разрешения.
- Для smoke используются первые 168 строк 2023 и четыре ID из train-selected списка 669.
  Список не изменяется по validation/test; это техническое чтение, без оценки точности.

## 2026-11-03: назначенный T06 после внешней приёмки B02

- Основание: владелец передал reviews/B02.md, подтвердил merge и назначил продолжение по задачам.
- Прежние32 станции/параметры/stride; B01/B02 сохраняются. Исправление N2 до augmented fit.
- До результатов: configs/t06.yaml и docs/T06_METHOD.md (commit a9d6bac): полный validation2024,
  5 независимых seeds, чистое/augmented обучение с явными весами, все ablations и NaN контроль.
- Новое хранение all-position predictions в NPZ требует независимого expected plan и проверки
  exact indices/cohort/model order/context; CSV evaluator не заменяется.
- Выбор training regime по заранее объявленному equal-scenario mean двух основных HGB;
  все результаты regimes/ablations сохраняются. Основное сравнение не требует победы QA.
- Block bootstrap168 steps/1000 replicas; срезы disruption/recovery24/ordinary; benchmark2 warmups/7 repeats.
- Freeze diagnostic/file_units, без превращения unknown units/time в verified. Gate2025 CLOSED;
  T07 и инфраструктурный T08 не назначены.
