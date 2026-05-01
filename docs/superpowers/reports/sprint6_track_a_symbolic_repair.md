# Sprint-6 Track A — Symbolic-Repair-Advisor live (hard)

Bundle hash: `sha256:ba75f39b5d9b02fa6ae4457dfdd36a0e4a5d794a9b0af61c3bf1f5ff273aab40`

## Aggregate

| Metric | Value |
| --- | --- |
| Tasks | 8 |
| Success rate | 25.0% |
| Cache-hit rate | 0.0% |
| Refined rate | 12.5% |
| Refinement uplift | 100.0% |
| P50 latency | 0.080s |
| P95 latency | 0.127s |
| Errors | 0 |

## Per-task

| Task | Score | Elapsed | Terminated by | Refined | Path |
| --- | ---: | ---: | --- | :-: | --- |
| 0201_count_objects_to_color | 0.00 | 0.123s | no_solution |   | - |
| 0202_largest_object_only | 1.00 | 0.129s | refined_success | ✓ | symbolic_repair |
| 0203_fill_enclosed_areas | 1.00 | 0.001s | search_success |   | - |
| 0204_grid_to_diagonal | 0.00 | 0.073s | no_solution |   | - |
| 0205_repeat_pattern_to_size | 0.00 | 0.085s | no_solution |   | - |
| 0206_remove_singletons | 0.00 | 0.075s | no_solution |   | - |
| 0207_color_by_size | 0.00 | 0.067s | no_solution |   | - |
| 0208_horizontal_then_vertical | 0.50 | 0.108s | search_exhausted |   | - |
