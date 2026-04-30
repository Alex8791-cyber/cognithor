# Cognithor PSE — ARC-DSL Reference

_Auto-generated. PSE version `1.2.0-draft`, DSL version `1.2.0`._

**61 primitives** registered, plus 13 predicate constructors and the closed Lambda / AlignMode / SortKey enums.

Run `cognithor pse dsl describe <name>` for any primitive to see its full record (signature + cost + description + examples).

## Catalog

### Output type: `Grid`

| Name | Signature | Cost | Description |
|---|---|---|---|
| `bounding_box` | `(Object) → Grid` | 1.50 | Render the object as a tight grid of size = bbox dimensions. Pixels inside the object get its color, pixels outside get 0. |
| `crop_bbox` | `(Grid) → Grid` | 1.50 | Crop to the bounding box of all non-background pixels (background = most-common color). Returns a 1×1 grid containing the background color if the grid is uniformly background. |
| `frame` | `(Grid, Color) → Grid` | 1.80 | Draw a 1-pixel border of *color* around the grid edge, leaving the interior unchanged. Grid must be at least 1×1. |
| `gravity_down` | `(Grid) → Grid` | 2.00 | Pull all non-background pixels in each column toward the bottom edge. |
| `gravity_left` | `(Grid) → Grid` | 2.00 | Pull all non-background pixels in each row toward the left edge. |
| `gravity_right` | `(Grid) → Grid` | 2.00 | Pull all non-background pixels in each row toward the right edge. |
| `gravity_up` | `(Grid) → Grid` | 2.00 | Pull all non-background pixels in each column toward the top edge. |
| `identity` | `(Grid) → Grid` | 0.10 | Return the grid unchanged. Cheap building block for branches. |
| `mask_apply` | `(Grid, Mask, Color) → Grid` | 2.00 | Set every cell of the grid where *mask* is True to *color*. Mask shape must match the grid shape exactly. |
| `mirror_antidiagonal` | `(Grid) → Grid` | 1.20 | Mirror across the anti-diagonal (top-right to bottom-left). |
| `mirror_diagonal` | `(Grid) → Grid` | 1.20 | Mirror across the main diagonal. Equivalent to transpose for square grids. |
| `mirror_horizontal` | `(Grid) → Grid` | 1.00 | Flip the grid left-to-right (mirror across the vertical axis). |
| `mirror_vertical` | `(Grid) → Grid` | 1.00 | Flip the grid top-to-bottom (mirror across the horizontal axis). |
| `overlay` | `(Grid, Grid, Color) → Grid` | 2.50 | Overlay *top* onto *base*: cells of *top* equal to *transparent_color* are skipped, all other cells overwrite *base*. Both grids must have the same shape. |
| `pad_with` | `(Grid, Color, Int) → Grid` | 1.80 | Pad the grid on all four sides with *width* pixels of *color*. Width must be ≥ 0. |
| `recolor` | `(Grid, Color, Color) → Grid` | 1.50 | Replace every occurrence of color *src* with color *dst*. |
| `render_objects` | `(ObjectSet, Grid) → Grid` | 2.00 | Paint every object in the set onto a copy of *base*. Cells outside the grid are silently dropped (clip-to-edge). Later objects overwrite earlier ones at overlapping cells. |
| `replace_background` | `(Grid, Color) → Grid` | 1.50 | Replace the background (most-common color) with the given color. Equivalent to ``recolor(grid, most_common_color(grid), new)``. |
| `rotate180` | `(Grid) → Grid` | 1.00 | Rotate the grid 180°. |
| `rotate270` | `(Grid) → Grid` | 1.00 | Rotate the grid 270° clockwise (= 90° counter-clockwise). |
| `rotate90` | `(Grid) → Grid` | 1.00 | Rotate the grid 90° clockwise. |
| `scale_down_2x` | `(Grid) → Grid` | 2.00 | Scale the grid down by 2× by sampling the top-left pixel of each 2×2 block. Odd dimensions are truncated. Only valid for grids with shape ≥ 2×2. |
| `scale_up_2x` | `(Grid) → Grid` | 2.00 | Scale the grid up by 2× (each pixel becomes a 2×2 block). |
| `scale_up_3x` | `(Grid) → Grid` | 2.00 | Scale the grid up by 3× (each pixel becomes a 3×3 block). |
| `shift` | `(Grid, Int, Int) → Grid` | 2.00 | Shift the grid by (dy, dx). Pixels that fall off the edge are dropped, exposed cells are filled with the background (most-common color). Range is unrestricted; large shifts collapse the output to all-background. |
| `stack_horizontal` | `(Grid, Grid) → Grid` | 2.00 | Stack two grids side-by-side (left-to-right). Row counts must match; output cols = left.cols + right.cols. |
| `stack_vertical` | `(Grid, Grid) → Grid` | 2.00 | Stack two grids top-to-bottom. Column counts must match; output rows = top.rows + bottom.rows. |
| `swap_colors` | `(Grid, Color, Color) → Grid` | 1.50 | Swap two colors throughout the grid. |
| `tile_2x` | `(Grid) → Grid` | 2.00 | Tile the grid in a 2×2 pattern (output dimensions = input × 2). |
| `transpose` | `(Grid) → Grid` | 1.00 | Transpose: swap rows and columns (flip across main diagonal). |
| `wrap_shift` | `(Grid, Int, Int) → Grid` | 2.20 | Shift the grid by (dy, dx) with toroidal wrap-around (numpy.roll). |

### Output type: `Color`

| Name | Signature | Cost | Description |
|---|---|---|---|
| `const_color_0` | `() → Color` | 0.50 | Constant color 0. |
| `const_color_1` | `() → Color` | 0.50 | Constant color 1. |
| `const_color_2` | `() → Color` | 0.50 | Constant color 2. |
| `const_color_3` | `() → Color` | 0.50 | Constant color 3. |
| `const_color_4` | `() → Color` | 0.50 | Constant color 4. |
| `const_color_5` | `() → Color` | 0.50 | Constant color 5. |
| `const_color_6` | `() → Color` | 0.50 | Constant color 6. |
| `const_color_7` | `() → Color` | 0.50 | Constant color 7. |
| `const_color_8` | `() → Color` | 0.50 | Constant color 8. |
| `const_color_9` | `() → Color` | 0.50 | Constant color 9. |
| `least_common_color` | `(Grid) → Color` | 1.00 | Return the least-frequent color present in the grid. Colors with zero occurrence are ignored; ties broken by lowest index. |
| `most_common_color` | `(Grid) → Color` | 1.00 | Return the most-frequent color in the grid (ties broken by lowest index). |

### Output type: `Mask`

| Name | Signature | Cost | Description |
|---|---|---|---|
| `mask_and` | `(Mask, Mask) → Mask` | 1.50 | Pixel-wise logical AND of two masks of equal shape. |
| `mask_eq` | `(Grid, Color) → Mask` | 1.50 | Return a boolean mask: True where the grid equals *color*. |
| `mask_ne` | `(Grid, Color) → Mask` | 1.50 | Return a boolean mask: True where the grid is *not* color. |
| `mask_not` | `(Mask) → Mask` | 1.20 | Pixel-wise logical NOT (involution: mask_not(mask_not(x)) == x). |
| `mask_or` | `(Mask, Mask) → Mask` | 1.50 | Pixel-wise logical OR of two masks of equal shape. |
| `mask_xor` | `(Mask, Mask) → Mask` | 1.50 | Pixel-wise logical XOR of two masks of equal shape. |

### Output type: `Object`

| Name | Signature | Cost | Description |
|---|---|---|---|
| `align_to` | `(Object, Object, AlignMode) → Object` | 3.00 | Translate object A so its bounding box aligns with B's per *mode*. CENTER aligns both axes; the four edges align that axis and centre the other; corners align both axes simultaneously. |
| `largest_object` | `(ObjectSet) → Object` | 1.50 | Object with the largest pixel count in the set. Ties broken by discovery order (first occurrence wins). |
| `smallest_object` | `(ObjectSet) → Object` | 1.50 | Object with the smallest pixel count in the set. Ties broken by discovery order (first occurrence wins). |

### Output type: `ObjectSet`

| Name | Signature | Cost | Description |
|---|---|---|---|
| `connected_components_4` | `(Grid) → ObjectSet` | 2.50 | 4-connectivity flood-fill of all non-background pixels. Background = most-common color (excluded from output). |
| `connected_components_8` | `(Grid) → ObjectSet` | 2.50 | 8-connectivity flood-fill of all non-background pixels. Diagonal neighbours count; otherwise identical to ``connected_components_4``. |
| `filter_objects` | `(ObjectSet, Predicate) → ObjectSet` | 2.50 | Keep only objects for which *pred* is True. The predicate's is_largest_in / is_smallest_in receive the original ObjectSet as context so 'largest' refers to the input set, not the filtered output. |
| `map_objects` | `(ObjectSet, Lambda) → ObjectSet` | 3.00 | Apply *fn* to every object in the set; return the resulting ObjectSet in the same order. Pure, no in-place mutation. |
| `objects_of_color` | `(Grid, Color) → ObjectSet` | 2.00 | Return the 4-connected components whose color matches the argument. Treats the requested color as foreground regardless of background. |
| `sort_objects` | `(ObjectSet, SortKey) → ObjectSet` | 2.50 | Stable-sort the set by *key*. Ties break by discovery order so the result is reproducible across runs (cache-stable). |

### Output type: `Lambda`

| Name | Signature | Cost | Description |
|---|---|---|---|
| `branch` | `(Predicate, Lambda, Lambda) → Lambda` | 3.50 | Build a conditional Lambda: ``λobj. then_fn(obj) if pred(obj) else else_fn(obj)``. Sub-tiefe ≤ 1 — nested ``branch`` forbidden in Phase 1 (spec §7.5). |

### Output type: `Int`

| Name | Signature | Cost | Description |
|---|---|---|---|
| `color_count` | `(Grid) → Int` | 1.00 | Number of distinct colors present in the grid (0..10). |
| `object_count` | `(ObjectSet) → Int` | 1.00 | Number of objects in the set (≥ 0). |

## Predicate constructors (closed set)

Higher-order primitives like `filter_objects` accept a `Predicate` argument. The constructor names below are the only predicates the search engine may construct (free Python lambdas are forbidden — sandbox guarantee, see spec §6.4).

| Constructor | Arity | Notes |
|---|---|---|
| `and` | 2 | combinator |
| `color_eq` | 1 |  |
| `color_in` | 1 |  |
| `is_largest_in` | 1 | needs ObjectSet context |
| `is_rectangle` | 0 |  |
| `is_smallest_in` | 1 | needs ObjectSet context |
| `is_square` | 0 |  |
| `not` | 1 | combinator |
| `or` | 2 | combinator |
| `size_eq` | 1 |  |
| `size_gt` | 1 |  |
| `size_lt` | 1 |  |
| `touches_border` | 0 | needs grid_shape context |

