"""Tests for the Image V2 domain (Sprint-26.4)."""

from __future__ import annotations

import pytest

from cognithor.channels.program_synthesis.domains.image_v2 import (
    IMAGE_V2_PRIMITIVE_NAMES,
    ImageV2Catalog,
    ImageV2Domain,
    ImageV2Primitive,
    ImageV2VerifierError,
    build_image_v2_catalog,
    register_image_v2_domain,
)
from cognithor.channels.program_synthesis.domains.registry import DomainRegistry


class TestImageV2Catalog:
    def test_builds(self) -> None:
        cat = build_image_v2_catalog()
        assert isinstance(cat, ImageV2Catalog)
        assert len(cat) == len(IMAGE_V2_PRIMITIVE_NAMES)

    def test_at_least_12_primitives(self) -> None:
        assert len(IMAGE_V2_PRIMITIVE_NAMES) >= 12

    def test_invalid_primitive_name(self) -> None:
        with pytest.raises(ValueError, match="Invalid Image-v2"):
            ImageV2Primitive(name="bad-!", fn=lambda x: x, cost=0.1)

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            ImageV2Primitive(name="p", fn=lambda x: x, cost=-1.0)


class TestSymmetry:
    def test_mirror_h(self) -> None:
        cat = build_image_v2_catalog()
        out = cat.get("mirror_h").fn([[1, 0], [0, 1]])
        assert out == ((0, 1), (1, 0))

    def test_mirror_v(self) -> None:
        cat = build_image_v2_catalog()
        out = cat.get("mirror_v").fn([[1, 2], [3, 4]])
        assert out == ((3, 4), (1, 2))

    def test_rotate_180(self) -> None:
        cat = build_image_v2_catalog()
        out = cat.get("rotate_180").fn([[1, 2], [3, 4]])
        assert out == ((4, 3), (2, 1))

    def test_transpose(self) -> None:
        cat = build_image_v2_catalog()
        out = cat.get("transpose").fn([[1, 2, 3], [4, 5, 6]])
        assert out == ((1, 4), (2, 5), (3, 6))

    def test_complete_symmetric_h(self) -> None:
        cat = build_image_v2_catalog()
        # Left half has data, right half is 0; complete fills the right.
        out = cat.get("complete_symmetric_h").fn([[1, 2, 0, 0]], fill_color=0)
        # Mirror partner of col 0 is col 3, of col 1 is col 2.
        # Filling cells with mirror partners → row becomes (1, 2, 2, 1)
        assert out == ((1, 2, 2, 1),)


class TestAnchors:
    def test_find_anchor_present(self) -> None:
        cat = build_image_v2_catalog()
        out = cat.get("find_anchor").fn([[0, 0], [0, 5]], color=5)
        assert out == (1, 1)

    def test_find_anchor_missing(self) -> None:
        cat = build_image_v2_catalog()
        out = cat.get("find_anchor").fn([[0, 0]], color=5)
        assert out is None

    def test_align_to_anchor(self) -> None:
        cat = build_image_v2_catalog()
        # 5 is at (1, 1); align it to (0, 0)
        out = cat.get("align_to_anchor").fn(
            [[0, 0, 0], [0, 5, 0], [0, 0, 0]],
            anchor=(1, 1),
            target=(0, 0),
            fill=0,
        )
        # Shift up-left by 1: cell (1,1) lands at (0,0).
        assert out[0][0] == 5


class TestConditionalFill:
    def test_fill_if_color(self) -> None:
        cat = build_image_v2_catalog()
        out = cat.get("fill_if_color").fn([[1, 2], [3, 1]], target=1, replacement=9)
        assert out == ((9, 2), (3, 9))

    def test_flood_fill_protected(self) -> None:
        cat = build_image_v2_catalog()
        # 0 background, 1 = barrier; barrier fully encloses the right
        # cell so flood from (0,0) cannot reach (0,2).
        out = cat.get("flood_fill_protected").fn(
            [[0, 1, 0], [0, 1, 1], [0, 0, 1]],
            start=(0, 0),
            new_color=7,
            barrier=1,
        )
        assert out[0][0] == 7
        assert out[2][1] == 7
        # (0,2) is fully walled off → stays 0
        assert out[0][2] == 0

    def test_flood_fill_no_op_when_origin_is_barrier(self) -> None:
        cat = build_image_v2_catalog()
        out = cat.get("flood_fill_protected").fn(
            [[1, 1], [0, 0]], start=(0, 0), new_color=7, barrier=1
        )
        # Started on a barrier — grid unchanged.
        assert out == ((1, 1), (0, 0))


class TestPattern:
    def test_find_period_h(self) -> None:
        cat = build_image_v2_catalog()
        # Period-2 pattern in row 0
        out = cat.get("find_period_h").fn([[1, 2, 1, 2, 1, 2]])
        assert out == 2

    def test_find_period_no_repeat(self) -> None:
        cat = build_image_v2_catalog()
        out = cat.get("find_period_h").fn([[1, 2, 3, 4]])
        assert out == 4

    def test_tile_pattern_h(self) -> None:
        cat = build_image_v2_catalog()
        out = cat.get("tile_pattern_h").fn([[1, 2]], target_cols=5)
        assert out == ((1, 2, 1, 2, 1),)

    def test_self_tile_by_mask(self) -> None:
        cat = build_image_v2_catalog()
        # Tile = [[1]], Mask = [[0, 1], [1, 0]]
        # Output 2x2 (mask) × 1x1 (tile) = 2x2
        out = cat.get("self_tile_by_mask").fn([[1]], mask=[[0, 1], [1, 0]], background=0)
        assert out == ((0, 1), (1, 0))


class TestObject:
    def test_connected_components_two_blobs(self) -> None:
        cat = build_image_v2_catalog()
        components = cat.get("connected_components").fn(
            [
                [1, 0, 0, 2],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        assert len(components) == 2

    def test_connected_components_empty(self) -> None:
        cat = build_image_v2_catalog()
        components = cat.get("connected_components").fn([[0, 0], [0, 0]])
        assert components == []

    def test_bounding_box(self) -> None:
        cat = build_image_v2_catalog()
        bb = cat.get("bounding_box").fn([(1, 2), (3, 4), (1, 4)])
        assert bb == (1, 2, 3, 4)

    def test_bounding_box_empty(self) -> None:
        cat = build_image_v2_catalog()
        bb = cat.get("bounding_box").fn([])
        assert bb == (0, 0, -1, -1)


class TestImageV2Domain:
    def test_metadata(self) -> None:
        d = ImageV2Domain()
        m = d.metadata
        assert m.name == "image_v2"
        assert m.benchmark_name == "arc-agi-1-training"

    def test_register(self) -> None:
        reg = DomainRegistry()
        register_image_v2_domain(reg)
        assert isinstance(reg.get("image_v2"), ImageV2Domain)

    def test_verify_pipeline(self) -> None:
        d = ImageV2Domain()
        ok = d.verify(
            [{"primitive": "mirror_h", "args": {}}],
            [
                {
                    "input": ((1, 0), (0, 1)),
                    "output": ((0, 1), (1, 0)),
                },
            ],
        )
        assert ok

    def test_verify_list_grid_equals_tuple_grid(self) -> None:
        d = ImageV2Domain()
        # list-input but tuple-output should still match via grid normalisation
        ok = d.verify(
            [{"primitive": "mirror_h", "args": {}}],
            [
                {
                    "input": [[1, 0], [0, 1]],
                    "output": [[0, 1], [1, 0]],  # list, not tuple
                },
            ],
        )
        assert ok

    def test_verify_mismatch_raises(self) -> None:
        d = ImageV2Domain()
        with pytest.raises(ImageV2VerifierError, match="!= expected"):
            d.verify(
                [{"primitive": "mirror_h", "args": {}}],
                [{"input": [[1, 0]], "output": [[5, 5]]}],
            )

    def test_verify_unknown_primitive(self) -> None:
        d = ImageV2Domain()
        with pytest.raises(ImageV2VerifierError, match="Unknown Image"):
            d.verify(
                [{"primitive": "no_such", "args": {}}],
                [{"input": [[1]], "output": [[1]]}],
            )

    def test_program_must_be_list_or_dict(self) -> None:
        d = ImageV2Domain()
        with pytest.raises(ImageV2VerifierError, match="must be"):
            d.verify("nope", [])

    def test_dict_program_shape(self) -> None:
        d = ImageV2Domain()
        ok = d.verify(
            {"program": [{"primitive": "rotate_180", "args": {}}]},
            [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}],
        )
        assert ok
