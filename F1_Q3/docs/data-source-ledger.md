# Data-source ledger

| Source ID | Institution | Dataset/file | Official URL | Access date | Access class | Format | Coverage | Unit | Keys | Core variables | Documentation | Terms and risks | Local raw path | Integrity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jolpica-qualifying | Jolpica-F1 / Ergast-compatible archive | Season qualifying results | https://api.jolpi.ca/ergast/f1/{season}/qualifying/ | 2026-08-08 UTC; exact timestamps in manifest | verified-downloadable | JSON | 2010-2019 Formula One | Driver x qualifying session | season, round, driverId | official position, Q1, Q2, Q3, driver, constructor, circuit | https://github.com/jolpica/jolpica-f1/blob/main/docs/endpoints/qualifying.md | Public volunteer-maintained API; custom User-Agent and pagination required; later historical corrections may change responses | data/raw/jolpica/qualifying_*.json | Per-page bytes and SHA-256 in manifest.json |
| jolpica-results | Jolpica-F1 / Ergast-compatible archive | Season race results | https://api.jolpi.ca/ergast/f1/{season}/results/ | 2026-08-08 UTC; exact timestamps in manifest | verified-downloadable | JSON | 2009-2019 Formula One | Driver x Grand Prix | season, round, driverId | finish position, points, starting grid, laps, status | https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md | Status definitions and historical corrections can affect counted starts | data/raw/jolpica/results_*.json | Per-page bytes and SHA-256 in manifest.json |

The documented unauthenticated limits are four requests per second and 500
requests per hour. The downloader uses 100-record pages, a custom User-Agent,
caching, a 0.32-second pause, and retry/backoff handling.
