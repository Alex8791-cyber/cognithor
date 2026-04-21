---
name: wetter-abfrage
description: "Weather query skill that retrieves current temperature, humidity, wind speed, and multi-day forecasts for a given city or coordinate. Use when checking the weather, getting a forecast, looking up temperature or wind conditions, or planning outdoor activities based on weather data."
---

# Wetter Abfrage

Retrieves current weather conditions (temperature, humidity, wind speed, conditions) and multi-day forecasts for a user-specified location.

## Steps

1. **Parse location** — extract city name or coordinates from user input:
   ```
   cognithor wetter_abfrage München
   ```
2. **Query weather API** — fetch current conditions and forecast data:
   ```python
   async with httpx.AsyncClient() as client:
       resp = await client.get(
           f"{API_BASE}/weather",
           params={"q": location, "units": "metric"}
       )
       data = resp.json()
   ```
3. **Format response** — present key metrics in a readable summary:
   ```
   München — 18°C, leicht bewölkt
   Luftfeuchtigkeit: 62% | Wind: 12 km/h NW
   Vorhersage: Mo 20°C ☀️ | Di 17°C 🌧️ | Mi 19°C ⛅
   ```
4. **Handle ambiguity** — if multiple locations match, present options:
   ```
   Meinten Sie: (1) München, DE  (2) München, AT?
   ```
5. **Return structured data** — `{"status": "ok", "result": <weather_data>}`

## Example

```
User > Wie wird das Wetter morgen in Berlin?
Cognithor > Berlin — morgen: 22°C, sonnig
         Luftfeuchtigkeit: 45% | Wind: 8 km/h SO
         UV-Index: 6 (hoch) — Sonnenschutz empfohlen
```

## Error Handling

| Error | Cause | Recovery |
|-------|-------|----------|
| Location not found | Typo or unknown city | Ask user to re-enter or provide coordinates |
| API timeout | Network issue or rate limit | Retry once after 2s; report failure if persistent |
| Missing data fields | Partial API response | Return available fields, note missing ones |
