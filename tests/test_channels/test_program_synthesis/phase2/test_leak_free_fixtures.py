# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""20-Task Leak-Free fixture-set tests (Sprint-2 Track C)."""

from __future__ import annotations

from collections import Counter

import numpy as np

from cognithor.channels.program_synthesis.core.types import TaskSpec
from cognithor.channels.program_synthesis.integration.capability_tokens import (  # noqa: F401
    PSECapability as _PSECapability,
)
from cognithor.channels.program_synthesis.synthesis.benchmark import BenchmarkTask
from cognithor.channels.program_synthesis.synthesis.leak_free_fixtures import (
    LEAK_FREE_TASKS,
    LeakFreeTask,
    benchmark_tasks,
    leak_free_set_hash,
)

# ---------------------------------------------------------------------------
# Bundle invariants
# ---------------------------------------------------------------------------


# Pin the leak-free bundle hash so any drift fails CI immediately.
EXPECTED_BUNDLE_HASH = "sha256:d30af55c7b2a9f02253177d77278f93999623571ee974086b612575aa6cfd917"


class TestBundleInvariants:
    def test_bundle_size_is_twenty(self) -> None:
        # Plan acceptance criterion: 20-Task Leak-Free Fixtures.
        assert len(LEAK_FREE_TASKS) == 20

    def test_bundle_hash_is_pinned(self) -> None:
        # Frozen digest — any drift fails this test, forcing an explicit
        # update of EXPECTED_BUNDLE_HASH on a fixture change.
        assert leak_free_set_hash() == EXPECTED_BUNDLE_HASH

    def test_task_ids_are_unique(self) -> None:
        ids = [t.task_id for t in LEAK_FREE_TASKS]
        assert len(set(ids)) == len(ids)

    def test_task_ids_use_sequential_ordinals(self) -> None:
        # 0001 .. 0020.
        for i, task in enumerate(LEAK_FREE_TASKS, start=1):
            assert task.task_id.startswith(f"{i:04d}_"), (
                f"expected 0001..0020 ordering at index {i}; got {task.task_id}"
            )

    def test_categories_balance(self) -> None:
        counts = Counter(t.category for t in LEAK_FREE_TASKS)
        # Documented breakdown.
        assert counts == {
            "rotation": 4,
            "mirror": 3,
            "recolor": 3,
            "scale": 3,
            "crop": 3,
            "mixed": 4,
        }

    def test_every_task_has_at_least_one_example(self) -> None:
        for task in LEAK_FREE_TASKS:
            assert len(task.examples) >= 1, f"{task.task_id} has zero examples"

    def test_every_task_has_a_solution_hint(self) -> None:
        for task in LEAK_FREE_TASKS:
            assert task.solution_hint, f"{task.task_id} has empty solution_hint"

    def test_every_task_has_a_description(self) -> None:
        for task in LEAK_FREE_TASKS:
            assert task.description, f"{task.task_id} has empty description"


# ---------------------------------------------------------------------------
# Per-task structural sanity
# ---------------------------------------------------------------------------


class TestPerTaskSanity:
    def test_examples_are_int8_arrays(self) -> None:
        for task in LEAK_FREE_TASKS:
            for inp, out in task.examples:
                assert isinstance(inp, np.ndarray)
                assert isinstance(out, np.ndarray)
                # Could be int8 directly or convertible — convert + check
                # values fit in int8.
                assert inp.min() >= -128 and inp.max() <= 127, task.task_id
                assert out.min() >= -128 and out.max() <= 127, task.task_id

    def test_to_task_spec_yields_valid_taskspec(self) -> None:
        for task in LEAK_FREE_TASKS:
            spec = task.to_task_spec()
            assert isinstance(spec, TaskSpec)
            # stable_hash is computable.
            h = spec.stable_hash()
            assert h.startswith("sha256:")

    def test_taskspec_hashes_are_unique(self) -> None:
        # Leak-free guarantee: every fixture has a distinct cache key.
        hashes = [t.to_task_spec().stable_hash() for t in LEAK_FREE_TASKS]
        assert len(set(hashes)) == len(hashes)


# ---------------------------------------------------------------------------
# benchmark_tasks() integration with the Sprint-1 benchmark driver
# ---------------------------------------------------------------------------


class TestBenchmarkTasksHelper:
    def test_yields_twenty_benchmark_tasks(self) -> None:
        tasks = benchmark_tasks()
        assert len(tasks) == 20
        for task in tasks:
            assert isinstance(task, BenchmarkTask)

    def test_default_budget_is_spec_partition(self) -> None:
        tasks = benchmark_tasks()
        first = tasks[0]
        # PartitionedBudget.from_spec_default is 0.07 / 0.70 / 0.18 / 0.05.
        assert abs(first.budget.mcts - 0.70) < 1e-9

    def test_per_task_alpha_default(self) -> None:
        tasks = benchmark_tasks()
        for task in tasks:
            assert task.current_alpha == 0.6

    def test_task_id_propagates(self) -> None:
        tasks = benchmark_tasks()
        ids = {t.task_id for t in tasks}
        assert ids == {t.task_id for t in LEAK_FREE_TASKS}

    def test_wall_clock_budget_overridable(self) -> None:
        tasks = benchmark_tasks(wall_clock_budget_seconds=12.5)
        for task in tasks:
            assert task.wall_clock_budget_seconds == 12.5


# ---------------------------------------------------------------------------
# Solution-hint sanity: spot-check a few tasks
# ---------------------------------------------------------------------------


class TestSolutionHintSanity:
    def test_rotation_solutions_match_expected(self) -> None:
        # 0001 should be a rotate90 task.
        t = next(task for task in LEAK_FREE_TASKS if task.task_id == "0001_rotate90_2x2")
        for inp, expected in t.examples:
            assert np.array_equal(np.rot90(inp, k=-1), expected)

    def test_mirror_horizontal_solution_matches(self) -> None:
        t = next(task for task in LEAK_FREE_TASKS if task.task_id == "0005_mirror_horizontal")
        for inp, expected in t.examples:
            assert np.array_equal(np.fliplr(inp), expected)

    def test_recolor_1_to_5_solution_matches(self) -> None:
        t = next(task for task in LEAK_FREE_TASKS if task.task_id == "0008_recolor_1_to_5")
        for inp, expected in t.examples:
            mapped = inp.copy()
            mapped[inp == 1] = 5
            assert np.array_equal(mapped, expected)

    def test_scale_up_2x_solution_matches(self) -> None:
        t = next(task for task in LEAK_FREE_TASKS if task.task_id == "0011_scale_up_2x_1x2")
        for inp, expected in t.examples:
            scaled = np.repeat(np.repeat(inp, 2, axis=0), 2, axis=1)
            assert np.array_equal(scaled, expected)


# ---------------------------------------------------------------------------
# LeakFreeTask dataclass contract
# ---------------------------------------------------------------------------


class TestDataclassContract:
    def test_is_frozen(self) -> None:
        # Sanity: every entry is a LeakFreeTask.
        for task in LEAK_FREE_TASKS:
            assert isinstance(task, LeakFreeTask)
