# PSE Phase-2 Benchmark Report (phase2_wired on arc:hard)

Bundle hash: `sha256:ba75f39b5d9b02fa6ae4457dfdd36a0e4a5d794a9b0af61c3bf1f5ff273aab40`

## Aggregate

| Metric | Value |
| --- | --- |
| Tasks | 8 |
| Success rate | 12.5% |
| Cache-hit rate | 0.0% |
| Refined rate | 25.0% |
| Refinement uplift | 0.0% |
| P50 latency | 0.082s |
| P95 latency | 0.127s |
| Errors | 0 |

## Regression verdict

- OK: success_rate 12.5% (delta+0.0% vs baseline 12.5%; tolerance 10.0%)
- P50 latency: 0.082s (delta+0.001s vs baseline 0.081s)
- P95 latency: 0.127s (delta+0.001s vs baseline 0.126s)

## Per-task

| Task | Score | Elapsed | Terminated by | Refined | Path |
| --- | ---: | ---: | --- | :-: | --- |
| 0201_count_objects_to_color | 0.00 | 0.122s | no_solution |   | - |
| 0202_largest_object_only | 0.50 | 0.130s | refined_partial | ✓ | - |
| 0203_fill_enclosed_areas | 1.00 | 0.001s | phase1_success |   | - |
| 0204_grid_to_diagonal | 0.00 | 0.075s | no_solution |   | - |
| 0205_repeat_pattern_to_size | 0.00 | 0.087s | no_solution |   | - |
| 0206_remove_singletons | 0.00 | 0.076s | no_solution |   | - |
| 0207_color_by_size | 0.00 | 0.068s | no_solution |   | - |
| 0208_horizontal_then_vertical | 0.50 | 0.111s | refined_partial | ✓ | - |
