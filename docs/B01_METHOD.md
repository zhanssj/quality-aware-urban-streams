# B01: причинные признаки и первый технический baseline

Назначение: T02 + T03, без промежуточного внешнего approval. T01 принят владельцем на основании
`reviews/T01-R1.md`. Gate T02 — локальные тесты, а не самостоятельное APPROVED.
После `handoffs/B01.md` остановка. T04/replay и test-2025 остаются закрыты.

## Признаки и расписание

Глобальная сетка — конкатенация полных строк выбранных последовательных годов до 2024 включительно.
Длины берутся из metadata. Пустые строки не удаляются. Sample ID: station_id + origin_index + target_index;
origin_index = target_index - 1. Для target=0 origin=-1, наблюдений ещё нет. Split определяется годом target.
Для row-interval конвенции cutoff_index=target_index — конец предшествующего интервала, wait=0.
Это индексная конвенция; календарная и физическая семантика не подтверждена.

`CausalFeatures.forecast()` вызывается **до** `observe(target)`. API forecast не принимает truth.
Загрузчик формирует отдельный TruthFrame; labels и оценщик используют его после построения признаков.
Feature/predict API не получают labels. В T03 доступность входов равна source_valid;
искусственных потерь и доставки пока нет. Батчирование объединяет уже причинно построенные snapshots.

Общий блок: лаги 1/2/3/6/12/24, mean/std 6 и 24 (ddof=0). Quality block: исходные availability
маски шести лагов, age, missing_ratio_6/24, current_gap, has_history.
Forward-fill использует последнее доступное прошлое; до него — train station median либо train global median
для станции без пригодных train измерений. Fit state явный и неизменяемый; fit читает только train годы.
Rolling считается по тому же причинно заполненному ряду в Standard и Quality-aware.
Padding до начала сетки — fallback; его availability=false и он входит в знаменатель missing ratios.
Age=origin_index-last_observed_index; без истории age=-1. Current_gap считает все подряд недоступные
позиции внутри сетки; до первой строки равен 0. Наблюдаемые нули и отрицательные числа сохраняются.
Первая validation позиция использует доступную историю train, затем поступившую прошлую validation историю.
Fitted state от validation не обновляется. Station ID и календарные признаки не являются predictors.

## Фиксированный пилот

`configs/baseline.yaml`: train 2021–2023 / validation 2024. Test reader отвергает 2025.
32 станции из 669: сортировка строковых train ID, PCG64(seed=2026), choice без возвращения,
затем сортировка выбранных ID. Список и hash сохраняются до чтения labels/сравнения.
Никакой оптимизации выбора по validation; исходный ретроспективный network selection бенчмарка сохраняется
как ограничение применимости результата.

Train subsample: target_index % 8 == 0, все станции; sample IDs записываются до fit.
Это 105120 scheduled train rows для полного train и 32 станций. Source-invalid labels исключаются,
естественная недоступность не заполняется. Медианы используют все доступные train строки выбранной группы.
`plan.json` до fit фиксирует cohort, config, лимиты и scheduled counts; `fit_plan.json` фиксирует фактические
source-valid labels и hashes входов. До первого validation numeric read оба HGB уже fitted.
Ограничение выборки связано с CPU/RAM pilot и не зависит от значений/ошибок моделей.

Persistence использует last-known либо тот же fallback. Standard и Quality-aware используют один
HistGradientBoostingRegressor, общие параметры и один train subset; отличаются только quality block.
Early stopping=false, validation_fraction=None; hyperparameter search отсутствует.
Параметры заранее заданы в YAML. Никакого обещания улучшения Quality-aware.

HDF5 читается блоками 168 строк × 32 станции × PM2.5. Predict batch — 168 шагов.
До 120000 scheduled train rows, 2 вычислительных потока, sampled RSS guard 2048 MiB.
RSS отслеживается каждые 50ms, проверяется между фазами/батчами; это не жёсткий OS memory limit.
Модели, train matrices, все predictions и sample IDs остаются в локальных artifacts.

## Метрики и воспроизведение

Wide CSV имеет один ряд на scheduled sample и три prediction columns. Естественно отсутствующая
истина сохраняется пустым target при source_valid=0; прогнозы сохраняются для всех scheduled rows.
Все методы score на пересечении source-valid target и конечных предсказаний **всех** методов.
Для каждого метода сохраняются scheduled/predicted/scored, prediction coverage, missing predictions,
skipped missing truth и skipped missing any prediction. Shared scored target IDs хешируются.
MAE/RMSE и macro station MAE — диагностические, `units_verified=false`, `file_units`.
Без подтверждения совместимых единиц pooled/macro числа не являются физическими выводами статьи.
Один seed, clean-trained evaluation на естественной неполноте; нет uncertainty intervals и обобщения на города.

Оценщик повторно читает тот же CSV и воспроизводит metrics JSON. Synthetic повтор с фиксированными
seed/config даёт те же метрики в установленном окружении; это не гарантия bitwise между версиями/OS.
Время forecast() construction (feature_compute_seconds), estimator fit/predict, полное job время и
simulated wait сохраняются раздельно. Observe/update и column stacking HGB не входят в таймер forecast().
job_wall_seconds снят до artifact hashing и записи финального manifest; полная latency CLI process не измерялась.
В B01 это один пилот, без warmup/повторов/performance claims T06.
Исторические dataset hashes не выдаются за новую полную проверку HDF5.

## Реализованные команды

```text
python -m pytest tests/test_features.py -q
python -m pytest tests/test_experiment.py -q
python -m aqstream train --config configs/baseline.yaml
python -m aqstream evaluate --predictions artifacts/<run>/predictions_validation.csv --out artifacts/<run>/recomputed.json
python -m aqstream export-review --out exports/B01-review.zip
```

AQ_DATA_ROOT указывает на существующие HDF5. AQ_ARTIFACT_ROOT по умолчанию `artifacts`.
Файлы конфигурации и пути внутри них задаются относительно working directory (корень проекта).
`prepare` остаётся ограниченным smoke T01. `replay`/`benchmark` не реализованы.
Review export использует whitelist, отбрасывает linked paths, модели/данные/полные predictions
и проверяет типовые token/key signatures. Архив содержит SHA256 каждого включённого файла;
это не гарантия обнаружения любых секретов и не независимое воспроизведение локальных артефактов.
