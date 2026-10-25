# API «Рикошета»

База: `http://127.0.0.1:8077`. Все рабочие маршруты — под `/api/v1`.
Архитектура и ограничения сервиса — в `docs/EDA_Ricochet.ipynb`, раздел 12.
Сценарии онлайн-тестирования и замеры скорости — в `docs/API_TEST_SCENARIOS.md`.

## /health

```powershell
curl.exe http://127.0.0.1:8077/health
# {"status":"ok","alarms":4731050,"synthetic":false,"months":["2020-04","2020-05"],
#  "pairs_last":258,"maintenance_windows":0,"historical_incidents":81}
```

## Направление 1: «бабочка»

### POST /api/v1/butterfly/run

Запускает анализ (фоновая задача). Тело — переопределения параметров движка
(все поля опциональны; по умолчанию — проверенные значения «Рикошета»):

```json
{}
```

```json
{
  "min_abs_corr": 0.35,
  "min_devices": 5,
  "lags": 12,
  "holdout_last": 1,
  "ts_from": "2020-04-20T00:00:00",
  "ts_to": "2020-06-01T00:00:00"
}
```

Ответ: `{"job_id":"a1b2c3...","status":"running"}`

### GET /api/v1/butterfly/jobs/{job_id}

Статус задачи. `done` → `result.info` (сводка прогона) и число пар.

### GET /api/v1/butterfly/pairs

Список пар последнего прогона. Фильтры: `cls` (`BUTTERFLY_CANDIDATE` / `DUPLICATE` /
`AMBIGUOUS`), `min_corr`, `min_score`, `limit`. Пара содержит: `corr`, `corr0`, `lead`,
`lag_min`, `support`, `days_ok/days_total`, `regions_ok`, `devices_a/b`, `overlap`
(пересечение устройств), `shared_unshared` (корреляция на общих/несовпадающих
устройствах), `class`, `score`, `month_stable`.

### GET /api/v1/butterfly/graph

Граф зависимостей: `nodes` (ряды) и `edges` (пары с вердиктами). Фильтры: `min_score`, `cls`.

### POST /api/v1/butterfly/verify

Вердикт по произвольной паре рядов (без полного прогона):

```json
{
  "key_a": {"integration":"nsp","system":"nokia","alert_group":"dryContactAlarm"},
  "key_b": {"integration":"transport","system":"alu","alert_group":"externalAlarm"},
  "lag_bins": 0
}
```

→ `overlap` + `shared_unshared` (`DUPLICATE` / `CROSS_OBJECT` / `NO_COMMON_DEVICES` / ...).

## Направление 2: предпроверка

### POST /api/v1/precheck/incident

Тело — инцидент с авариями:

```json
{
  "incident_id": "INC-123",
  "region": "MSK",
  "alarms": [
    {"ts": "2026-01-10T12:00:00", "integration": "ipcore", "system": "syslog",
     "alert_group": "RPD-ISIS-ADJ", "node": "jmx001.msk", "region": "MSK", "severity": 4}
  ]
}
```

Ответ — собранное «дело» для агента:

- `maintenance` — попадание в окна обслуживания (`hit`, список окон);
- `enrichment` — полнота заполнения полей (по полям и общая);
- `root_cause_candidates` — кандидаты-первопричины из скрытых зависимостей:
  система/тип, лид, лаг, устойчивость, `in_incident` (лидер уже среди аварий
  инцидента или надо смотреть шире), `predicts_downstream` — ожидаемый эффект;
- `similar_incidents` — похожие исторические инциденты (Jaccard по профилю
  «система+тип» + бонус региона), с датой, ключами, note;
- `runbook` — подобранный runbook;
- `blast` — масштаб (узлы, регионы, максимальная критичность);
- `risk` — `score` 0–100, `level` LOW/MEDIUM/HIGH, `flags`:
  `MAINTENANCE_OVERLAP`, `BUTTERFLY_LINKED`, `LOW_ENRICHMENT`, `BLAST_RADIUS`,
  `HIGH_SEVERITY`, `REPEAT_OFFENDER`.

### POST /api/v1/precheck/alarm

То же для одиночной аварии (тело — один аларм).

### GET/POST /api/v1/precheck/runbooks

Реестр runbook'ов: `{id, system, alert_group, url, steps[], owner}`.
Матчинг: точный `(system, alert_group)` → по `system` → ничего.

## Окна обслуживания

- `GET /api/v1/maintenance/windows` — список
- `POST /api/v1/maintenance/windows` — добавить:
  `{"start": "...", "end": "...", "region": "", "system": "", "node": "", "note": ""}`
  (пустое поле = не фильтровать по нему)
- `DELETE /api/v1/maintenance/windows/{id}`

Аварии внутри окон исключаются из анализа «бабочки» и помечаются в предпроверке.
