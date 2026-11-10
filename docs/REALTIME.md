# Real-time контур: интенсивность, EWMA, кластеры, causal graph и runbook

Новый слой `src/ricochet/reallife/` — изолированная надстройка над пакетным
`leadlag.py` и XGBoost-ранжировщиком. Ядро не меняется: batch-детекция
по-прежнему ищет устойчивые пары на месячных окнах, а realtime-слой работает с
коротким окном инцидента и отвечает на вопросы «что происходит сейчас» и «что
проверить инженеру».

## Зачем интенсивность и EWMA

Netcool выдаёт редкие точки. Для причинного анализа нужен непрерывный ряд,
поэтому используется двухступенчатое преобразование:

1. **Intensity Function** — количество аварий в скользящем окне с шагом
   `step_seconds` (по умолчанию 10 секунд). Нет аварии — значение `0`. Это
   корректно: отсутствие события означает нулевую интенсивность, а не пропуск.
2. **EWMA** — экспоненциальное сглаживание:

   `s_t = α·x_t + (1−α)·s_{t−1}`

   где `α` вычисляется из `ewma_half_life_seconds` и шага. Сглаживание склеивает
   короткие разрывы доставки и даёт сигнал, пригодный для lagged-анализа.

Если окно слишком длинное, шаг автоматически увеличивается, чтобы матрица
`бины × ряды` не превысила `max_bins`. Это фиксируется в `warnings`.

## Единый вход

Все три эндпоинта принимают один и тот же пакет:

```json
{
  "incident_id": "INC-2026-001",
  "region": "MSK",
  "start": "2026-09-01T10:00:00",
  "end": "2026-09-01T10:30:00",
  "alarms": [
    {
      "ts": "2026-09-01T10:01:00",
      "integration": "Huawei I-PE",
      "system": "Huawei NE-Series",
      "alert_group": "mplsTunnelRerouted",
      "node": "pe1.msk",
      "node_alias": "pe1",
      "region": "MSK",
      "severity": 4
    }
  ],
  "topology_nodes": [
    {"id": "pe1", "node": "pe1.msk", "device_type": "router"}
  ],
  "topology_edges": [
    {"source": "pe1", "target": "pe2", "relation": "link"}
  ],
  "maintenance": [
    {"start": "2026-09-01T09:00:00", "end": "2026-09-01T11:00:00",
     "region": "MSK", "note": "planned works"}
  ]
}
```

Поля `topology_*` и `maintenance` необязательны. Топология используется как
контекст/ограничение для causal graph, окна обслуживания — как флаг риска.

## Эндпоинты

### POST /api/v1/topology/causal-graph

Строит направленный ациклический граф (DAG) по EWMA-сигналам.

Основные query-параметры:

| параметр | смысл |
|---|---|
| `step_seconds` | шаг интенсивности, по умолчанию 10 |
| `ewma_half_life_seconds` | период полураспада EWMA, по умолчанию 60 |
| `max_bins` / `max_series` | защита памяти; при переполнении шаг/ряды усекаются с warning |
| `max_lag_seconds` | максимальный лаг причинного влияния |
| `min_abs_corr` | порог корреляции для скелета |
| `min_causal_power` | порог силы ребра после conditional-independence |
| `max_cond_size` | размер conditioning set в PC-тесте |
| `remove_common_factor` | убирать общую активность сети |
| `use_topology`, `topology_strict` | использовать CMDB-связи как контекст/жёсткое ограничение |

Ответ:

```json
{
  "nodes": [
    {"id": "Huawei I-PE|Huawei NE-Series|mplsTunnelRerouted",
     "system": "Huawei NE-Series", "alert_group": "mplsTunnelRerouted",
     "events": 120, "devices": 22, "peak_ewma": 1.0}
  ],
  "edges": [
    {"from": "Huawei I-PE|Huawei NE-Series|mplsTunnelRerouted",
     "to": "Netcool GT - Syslog|Juniper|RPD-ISIS-ADJ",
     "causal_power": 0.78, "lag_bins": 2, "lag_seconds": 20,
     "corr": 0.78, "partial_corr": 0.78, "method": "lagged_pc"}
  ],
  "topological_order": ["...", "..."],
  "warnings": []
}
```

`causal_power` — сила связи после контроля общих факторов и conditioning set.
Это **не** формальный Average Treatment Effect из DoWhy; для продакшена
адаптер можно заменить на CausalNex/DoWhy, сохранив контракт ответа.

### POST /api/v1/anomalies/clusters

Группирует series по частотным и временным признакам: события, active bins,
duty, peak intensity, EWMA area, bursts, число устройств, severity.

По умолчанию используется `HDBSCAN` из scikit-learn. Если плотных кластеров
нет, автоматически включается `KMeans`. Можно явно выбрать `method=kmeans` или
`method=dbscan`.

Ответ содержит `clusters`, `noise`, `method_used`, `warnings` и признаки
каждого ряда. Кластер с `is_micro_storm=true` — кандидат на «микро-шторм»,
который ещё не стал глобальной аварией.

### POST /api/v1/agent/runbook

Оркестратор:

1. строит causal graph;
2. кластеризует текущий шторм;
3. превращает причинные рёбра в пары и передаёт их в существующий
   `PairRanker` (`XGBoost`, а при отсутствии модели — `heuristic-v1`);
4. подбирает runbook по `(system, alert_group)` из `RunbookRegistry`;
5. добавляет шаги проверки CMDB, микро-штормов и окон обслуживания.

Ответ:

```json
{
  "summary": "Наиболее вероятная первопричина: ...",
  "confidence": 0.91,
  "root_causes": [
    {"system": "Huawei NE-Series", "alert_group": "mplsTunnelRerouted",
     "probability": 0.91, "effects": ["..."]}
  ],
  "steps": [
    {"order": 1, "action": "Проверить ...", "source": "runbook:rb-mpls"}
  ],
  "evidence": {"causal": {}, "clusters": [], "ranker": {}, "maintenance": []},
  "generated_by": "local-rules-v1",
  "warnings": []
}
```

По умолчанию текст детерминированный и локальный. В `RunbookAgent` можно
передать `llm`-функцию (`Callable[[str], str]`), которая получит
структурированный prompt и заменит только нарратив, сохранив evidence и
проверяемые шаги.

## Как это связано с существующим «Рикошетом»

- `leadlag.py` не изменяется.
- `ranking/` переиспользуется: XGBoost/эвристика даёт уверенность для лидера.
- `precheck/runbooks.py`, `precheck/similar.py` переиспользуются как реестры.
- Batch-контур по-прежнему строит месячный граф; realtime-контур отвечает за
  горячее окно инцидента.

## Production-архитектура: Kafka/Vector → ClickHouse → FastAPI

Локальная реализация принимает пачку алармов в JSON, чтобы её можно было
тестировать без инфраструктуры. В production вход нужно заменить на адаптер:

```text
Netcool → Kafka/Vector → ClickHouse (10-30s bins, TTL 2h) → FastAPI realtime
                                      │
                                      └─ batch worker → monthly graph/retrain
```

Reference SQL для ClickHouse:

```sql
SELECT
    intDiv(toUnixTimestamp(AggregationFirst), 10) * 10 AS bucket,
    IntegrationName AS integration,
    Agent AS system,
    AlertGroup AS alert_group,
    count() AS intensity
FROM netcool_alarms
WHERE AggregationFirst >= now() - INTERVAL 15 MINUTE
  AND OriginalSeverity > 0
GROUP BY bucket, integration, system, alert_group
ORDER BY bucket;
```

Для EWMA в ClickHouse можно использовать оконные функции или materialized view
с состоянием; для локального режима EWMA считается в `pipeline.py`.

Рекомендуемое разделение нагрузки:

| слой | частота | что делает |
|---|---|---|
| ingest | непрерывно | Kafka/Vector, нормализация, запись в ClickHouse |
| hot inference | 10–30 сек | intensity/EWMA, HDBSCAN, causal graph, XGBoost predict |
| warm retrain | 10–30 мин | дообучение XGBoost по свежему feedback |
| cold rebuild | 1 час / 1 сутки | пересборка месячного causal graph и весов |

## Ограничения

- Короткое окно инцидента может дать ложные рёбра из-за периодичности или
  общего шторма. Поэтому есть `remove_common_factor`, PC-conditioning и
  topology constraint; всё равно нужна экспертная проверка.
- `causal_power` — не ATE и не доказательство причинности; это приоритет для
  проверки.
- Топология в Netcool заполнена не везде (`Tp1*`/`ObjectId` часто пусты).
  При отсутствии CMDB-связей ограничение отключается с предупреждением.
- Для LLM-режима обязательна маскировка `Node`, `NodeAlias`, IP, `POSITION_CODE`,
  `RAN_*` и других чувствительных полей.
- В hot path нельзя запускать `run_butterfly` на миллионах строк: используйте
  ClickHouse-агрегаты или кэш последнего batch-графа.
