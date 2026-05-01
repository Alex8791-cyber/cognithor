# Sprint-7 Track A1 — Cascade Generalisation (hard)

Bundle hash: `sha256:3dfac922848dc51fb8420416f1b87347b9654439cecf46610b5605ef2494d6fe`

## Aggregate

| Metric | Value |
| --- | --- |
| Tasks | 8 |
| Success rate | 50.0% |
| Cache-hit rate | 0.0% |
| Refined rate | 37.5% |
| Refinement uplift | 66.7% |
| P50 latency | 0.152s |
| P95 latency | 0.417s |
| Errors | 0 |

## Per-task

| Task | Score | Elapsed | Terminated by | Refined | Path |
| --- | ---: | ---: | --- | :-: | --- |
| 0201_count_objects_to_color | 0.50 | 0.455s | refined_partial | ✓ | cascade |
| 0202_largest_object_only | 1.00 | 0.130s | refined_success | ✓ | cascade |
| 0203_fill_enclosed_areas | 1.00 | 0.001s | search_success |   | - |
| 0204_grid_to_diagonal | 0.00 | 0.335s | no_solution |   | - |
| 0205_repeat_pattern_to_size | 0.00 | 0.347s | no_solution |   | - |
| 0206_remove_singletons | 1.00 | 0.074s | refined_success | ✓ | cascade |
| 0207_color_by_size | 0.00 | 0.174s | no_solution |   | - |
| 0208_horizontal_then_vertical | 1.00 | 0.069s | search_success |   | - |
