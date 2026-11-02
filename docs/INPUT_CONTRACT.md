# Контракт загрузчика T01

`HDF5Reader(data_root=None)` берёт путь из аргумента либо `AQ_DATA_ROOT`. Путь должен существовать.
Отсутствие файла или переменной — `DataContractError`, никаких сетевых fallback.
Имена файлов: `aq_compact_<year>.h5`. Годы 2021–2024 доступны API; 2025 запрещён до открытия.
CLI T01 допускает только 2021–2023, 2–4 станции, диапазон не более 168 строк.

Обязательные локальные datasets:

| Dataset | Форма | Проверка |
|---|---|---|
| values | (T, N, P), все оси непустые | real numeric |
| masks | та же форма | bool или real numeric; пригодно только значение 1 |
| station_ids | (N,) | строки/целые ID, без пустых и дубликатов |
| params | (P,) | уникальные нормализованные имена |

Внешние/soft links, virtual datasets и external storage отклоняются.
В исходных данных никаких записей нет: все HDF5 открываются только с режимом `r`.
Метаданные осей читаются целиком; `values`/`masks` — только выбранные time slices, station columns
и один channel. В памяти нужны массивы порядка `chunk_size × selected_stations`,
плюс небольшие оси метаданных и внутренний HDF5 chunk cache.

```python
from aqstream.data import HDF5Reader

reader = HDF5Reader()  # AQ_DATA_ROOT
for block in reader.iter_blocks(
    year=2023, station_ids=["1003", "1000"], start=0, stop=168,
    chunk_size=24, parameter="PM2.5",
):
    # values: float64, NaN для всего недоступного
    # source_valid: bool, независимо от будущих drops/delivery
    # start/stop: исходные локальные индексы, полуинтервал [start, stop)
    # station_ids: порядок запроса, а не порядок HDF5
    print(block.start, block.stop, block.values.shape)
```

Поддерживаются имена `PM2.5`, `pm25`, `PM₂.₅`, `pm2p5`. Канал ищется по имени, а не позиции.
Числа под невалидной маской и NaN/Inf под любой маской недоступны. Mask != 1 не исправляется.
Оси не переупорядочиваются по времени; пустые строки остаются. Нулей/отрицательных значений
под валидной маской loader не отбрасывает. Заполнение, timestamps, split по target и causal features — T02.

Manifest smoke сохраняет observed размер/mtime файла и ранее записанный SHA256 из download manifest
с `sha256_verified_in_this_run=false`. Новый полный hash исходного HDF5 не вычисляется.
`sanitized_blocks_sha256` относится только к выбранным очищенным блокам и зависит от chunk_size;
конфигурация и алгоритм кодирования записаны рядом. Это не hash всего датасета.
Предсказания, метрики модели, ожидание и inference обозначены NOT_RUN/null, когда их не было.
