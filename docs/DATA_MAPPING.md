# DATA_MAPPING: AIOps Challenge 2020 → каноническая схема

Источник: `AIOps挑战赛2020预赛数据.zip` (md5 `fac7fe1b4e048c81ef88874334b73534`).
Внешние пути китайские — переименовываются при распаковке, оригинал не меняется.

## Переименование путей

| оригинал | после extract |
|---|---|
| `AIOps挑战赛数据/` | `challenge_data/` |
| `故障整理（预赛）.csv` | `faults_preliminary.csv` |
| `平台指标/` (внутри дневного зипа) | `platform/` |
| `业务指标/` | `business/` |
| `调用链指标/` | `trace/` |

Дневные зипы `2020_MM_DD.zip` → `challenge_data/2020_MM_DD/2020_MM_DD/{platform,business,trace}/*.csv`.
Колонки внутри CSV уже английские (кроме документации в `data_release_v3.5.zip`).

## Состав датасета

15 дней: 2020-04-11, 2020-04-20…23, 2020-05-22…31 (UTC+8 в epoch-ms).

| слой | файлы | колонки |
|---|---|---|
| platform | `os_linux`, `db_oracle_11g`, `mw_redis`, `dcos_container`, `dcos_docker` | `itemid, name, bomc_id, timestamp(ms), value, cmdb_id` |
| business | `esb` | `serviceName, startTime(ms), avg_time, num, succee_num, succee_rate` |
| trace | `trace_{csf,osb,jdbc,local,remote_process,fly_remote}` | `callType, startTime(ms), elapsedTime, success, traceId, id, pid, cmdb_id, serviceName` |

`faults_preliminary.csv` (7.8 КБ): `index, object, fault_desrcibtion, kpi, name,
container, log_time, log_block, block, start_time, duration` — 190 размеченных отказов.

## Конвертация в аварии

Общие правила: время `epoch ms → ts (UTC)`; `cmdb_id → node` (и `node_alias` остаётся
пустым — в датасете нет второго идентификатора устройства); регион отсутствует → `""`.

### platform → аварии

По каждой паре `(name, cmdb_id)` считается скользящая медиана/MAD (окно 120 точек).
Выход за `median ± k·1.4826·MAD` — авария (только первый бин подряд, чтобы шторм
не раздувался):

- `alert_group = f"{name}_high" | f"{name}_low"` — метрика + направление
  (`CPU_free_pct_low` — «свободный CPU упал», `container_cpu_used_high` — «CPU контейнера вырос»)
- `system` = имя файла-источника (`os_linux`, `db_oracle_11g`, ...) — аналог пробы/Agent
- `integration = "platform"`, `severity = 4`

### business → аварии

ESB-сервисы (минутная гранулярность):

- `succee_rate < 0.98` → `esbLowSuccess` (sev 4)
- `avg_time` выше `median + k·MAD` → `esbSlowResponse` (sev 3)
- `node = serviceName`, `system = "esb"`, `integration = "business"`

### trace → аварии

- спан с `success == False` → `alert_group = f"{callType}_fail"` (`CSF_fail`, `LOCAL_fail`, `OSB_fail`)
- `elapsedTime` выше `median + k·MAD` (окно 2000 точек) → `f"{callType}_slow"`
- `system = serviceName` (сервис, который падает), `node = cmdb_id` (где исполнялся спан),
  `integration = "trace"`

Итог: ~425 тыс. аварий/день, суммарно ~6.4 млн за 15 дней.

## faults → исторические инциденты

Каждая строка `faults_preliminary.csv` → инцидент:

- `started_at = log_time`, `fault_type = fault_desrcibtion`,
  `fault_location = name`, `fault_detail = container`
- `keys` — профиль для поиска похожих: `(object→system, kpi_high)` + топ-8 фактических
  `(system, alert_group)` из аварий в течение часа после отказа
- используется предпроверкой (`similar_incidents`) и как ground-truth для валидации
  найденных «бабочек»

## Маппинг на термины Netcool

| канон | Netcool | AIOps 2020 |
|---|---|---|
| `integration` | `IntegrationName` | слой мониторинга |
| `system` | `Agent` | файл-источник / сервис |
| `alert_group` | `AlertGroup` | метрика+направление или тип фейла |
| `node`/`node_alias` | `Node`/`NodeAlias` | `cmdb_id` / — |
| `region` | `Tp1SuperRegion` | — |
| `severity` | `OriginalSeverity` | 3–4 |
