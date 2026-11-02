# AirQualityBench: локальный аудит

Проверено файлов: 5. Ошибок открытия/сканирования: 0.

Показатели ниже рассчитаны по локальным файлам. Годы взяты из имён файлов. Календарная ось, единицы и достоверность измерений независимо не подтверждены.

Доступное значение (usable): mask == 1 и конечное число. Отрицательные значения отмечаются отдельно, но автоматически не удаляются.

| Год | Строки | Станции | Доступность PM2.5 | Станции ≥80% | Без PM2.5 |
|---|---:|---:|---:|---:|---:|
| 2021 | 8760 | 3720 | 21.22% | 19 | 1807 |
| 2022 | 8760 | 3720 | 23.48% | 0 | 1623 |
| 2023 | 8760 | 3720 | 39.72% | 669 | 1520 |
| 2024 | 8784 | 3720 | 48.78% | 1457 | 1475 |
| 2025 | 8760 | 3720 | 49.07% | 1512 | 1426 |

## Проблемы и оговорки

### aq_compact_2021.h5
- WARNING: Physical units, source calibration and actual arrival times are not validated by this numerical audit

### aq_compact_2022.h5
- WARNING: Physical units, source calibration and actual arrival times are not validated by this numerical audit

### aq_compact_2023.h5
- WARNING: Physical units, source calibration and actual arrival times are not validated by this numerical audit

### aq_compact_2024.h5
- WARNING: Physical units, source calibration and actual arrival times are not validated by this numerical audit

### aq_compact_2025.h5
- WARNING: Physical units, source calibration and actual arrival times are not validated by this numerical audit

## Ограничения
- No confirmation of timestamp semantics, timezone, units or calibration
- A missing cell does not identify its cause: outage, absent sensor, filtering or other causes
- Long runs include station inactivity; internal gaps are reported separately
- No reconstruction of true arrival timestamps
- Do not choose stations, tune models, or fit scalers using 2025 test outcomes
- scaler.csv is not applied; fit any scaler on training data only

Проверки порядка станций между годами, перечень HDF5-полей и проверка координат находятся в summary.json.
Ни одна подвыборка станций не отбирается автоматически. Статистика тестового периода не должна управлять подбором модели.
