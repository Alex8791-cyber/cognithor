# Cognithor PSE — ARC-DSL Reference

_Auto-generated. PSE version `1.2.0`, DSL version `1.2.0`._

**130 primitives** registered, plus 13 predicate constructors and the closed Lambda / AlignMode / SortKey enums.

Run `cognithor pse dsl describe <name>` for any primitive to see its full record (signature + cost + description + examples).

## Catalog

### Output type: `Grid`

| Name | Signature | Cost | Description |
|---|---|---|---|
| `bounding_box` | `(Object) → Grid` | 1.50 | Render the object as a tight grid of size = bbox dimensions. Pixels inside the object get its color, pixels outside get 0. |
| `complete_symmetry_antidiag` | `(Grid) → Grid` | 2.70 | Fill in the (square) grid so it is symmetric across its anti-diagonal (top-right to bottom-left). Non-square grids fall back to the input unchanged. Existing non-zero cells are preserved; zero cells are filled from their anti-transposed partner. |
| `complete_symmetry_d` | `(Grid) → Grid` | 2.70 | Fill in the (square) grid so it is symmetric across its main diagonal (transpose mirror). Non-square grids fall back to the input unchanged — Phase-1 search must guard with the shape check upstream. Existing non-zero cells are preserved; zero cells are filled from their transposed partner. |
| `complete_symmetry_h` | `(Grid) → Grid` | 2.50 | Fill in the grid so it is symmetric across its vertical axis (left-right mirror). Existing non-zero cells are preserved; zero cells are filled from their horizontal partner if that partner is non-zero. Solves ARC tasks with horizontally-defaced symmetric figures. |
| `complete_symmetry_v` | `(Grid) → Grid` | 2.50 | Fill in the grid so it is symmetric across its horizontal axis (top-bottom mirror). Existing non-zero cells are preserved; zero cells are filled from their vertical partner if that partner is non-zero. Solves ARC tasks with vertically-defaced symmetric figures. |
| `count_components` | `(Grid) → Grid` | 2.50 | Count the number of 4-connected non-zero components and return a 1×1 grid containing that count as its single colour. Counts saturate at 9 (the ARC colour range). |
| `crop_bbox` | `(Grid) → Grid` | 1.50 | Crop to the bounding box of all non-background pixels (background = most-common color). Returns a 1×1 grid containing the background color if the grid is uniformly background. |
| `crop_largest_component` | `(Grid) → Grid` | 2.50 | Find the largest 4-connected non-zero component and return its bounding-box subgrid (other cells in the bbox stay zero). Differs from `crop_bbox` (which crops to the bbox of every non-background cell jointly): solves ARC tasks where the rule is 'extract the dominant shape, drop the rest'. |
| `crop_smallest_component` | `(Grid) → Grid` | 2.50 | Find the smallest 4-connected non-zero component and return its bounding-box subgrid. Mirror of `crop_largest_component`. Solves ARC tasks where the rule is 'extract the rare/marker shape, drop the noise'. |
| `crop_to_least_common_color_cells` | `(Grid) → Grid` | 2.50 | Find the rarest non-zero colour and return the bounding-box subgrid of cells of that colour (other cells in the bbox stay in their original colour). The rarest colour breaks ties by lowest index. Solves ARC tasks where the rule is 'find the marker / odd one out and crop around it'. |
| `fill_with_most_common_color` | `(Grid) → Grid` | 1.50 | Return a grid of the same shape as the input, filled with its most-frequent colour (ties broken by lowest index, matching `most_common_color`). Solves ARC tasks of the 5582e5ca family where the rule is 'collapse the input to its dominant colour'. |
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
| `neighbor_count_grid` | `(Grid) → Grid` | 2.00 | Replace each cell with the count of its 8-connected non-zero neighbours (including itself if non-zero), capped at 9. Output has the same shape as the input. Solves ARC tasks where the rule is 'mark each cell with its local density'. |
| `overlay` | `(Grid, Grid, Color) → Grid` | 2.50 | Overlay *top* onto *base*: cells of *top* equal to *transparent_color* are skipped, all other cells overwrite *base*. Both grids must have the same shape. |
| `pad_with` | `(Grid, Color, Int) → Grid` | 1.80 | Pad the grid on all four sides with *width* pixels of *color*. Width must be ≥ 0. |
| `recolor` | `(Grid, Color, Color) → Grid` | 1.50 | Replace every occurrence of color *src* with color *dst*. |
| `recolor_by_component_size` | `(Grid) → Grid` | 3.00 | Recolour every 4-connected non-zero component so its colour equals its size, capped at 9. Background cells (colour 0) are preserved. |
| `remove_singletons` | `(Grid) → Grid` | 2.50 | Replace every cell whose colour has no orthogonal same-colour neighbour with 0. Background cells (colour 0) are preserved. |
| `render_objects` | `(ObjectSet, Grid) → Grid` | 2.00 | Paint every object in the set onto a copy of *base*. Cells outside the grid are silently dropped (clip-to-edge). Later objects overwrite earlier ones at overlapping cells. |
| `replace_background` | `(Grid, Color) → Grid` | 1.50 | Replace the background (most-common color) with the given color. Equivalent to ``recolor(grid, most_common_color(grid), new)``. |
| `rotate180` | `(Grid) → Grid` | 1.00 | Rotate the grid 180°. |
| `rotate270` | `(Grid) → Grid` | 1.00 | Rotate the grid 270° clockwise (= 90° counter-clockwise). |
| `rotate90` | `(Grid) → Grid` | 1.00 | Rotate the grid 90° clockwise. |
| `scale_down_2x` | `(Grid) → Grid` | 2.00 | Scale the grid down by 2× by sampling the top-left pixel of each 2×2 block. Odd dimensions are truncated. Only valid for grids with shape ≥ 2×2. |
| `scale_up_2x` | `(Grid) → Grid` | 2.00 | Scale the grid up by 2× (each pixel becomes a 2×2 block). |
| `scale_up_3x` | `(Grid) → Grid` | 2.00 | Scale the grid up by 3× (each pixel becomes a 3×3 block). |
| `self_tile_by_mask` | `(Grid) → Grid` | 3.00 | Fractal self-tile: tile the grid by itself using its non-zero cells as a placement mask. Output shape = (R*R, C*C) for an R×C input. For each input cell (i, j), if grid[i, j] != 0 the entire input is stamped at output block (i*R..i*R+R, j*C..j*C+C); otherwise that block stays zero. Solves ARC tasks of the 007bbfb7 family. |
| `shift` | `(Grid, Int, Int) → Grid` | 2.00 | Shift the grid by (dy, dx). Pixels that fall off the edge are dropped, exposed cells are filled with the background (most-common color). Range is unrestricted; large shifts collapse the output to all-background. |
| `stack_horizontal` | `(Grid, Grid) → Grid` | 2.00 | Stack two grids side-by-side (left-to-right). Row counts must match; output cols = left.cols + right.cols. |
| `stack_vertical` | `(Grid, Grid) → Grid` | 2.00 | Stack two grids top-to-bottom. Column counts must match; output rows = top.rows + bottom.rows. |
| `swap_colors` | `(Grid, Color, Color) → Grid` | 1.50 | Swap two colors throughout the grid. |
| `tile_2x` | `(Grid) → Grid` | 2.00 | Tile the grid in a 2×2 pattern (output dimensions = input × 2). |
| `tile_3x` | `(Grid) → Grid` | 2.50 | Tile the grid in a 3×3 pattern (output dimensions = input × 3). |
| `transpose` | `(Grid) → Grid` | 1.00 | Transpose: swap rows and columns (flip across main diagonal). |
| `unique_colors_diagonal` | `(Grid) → Grid` | 3.00 | Extract the sorted set of unique non-zero colours in the input and return an N×N grid whose main diagonal contains those colours (N = number of unique non-zero colours). The off-diagonal cells are zero. When the input has no non-zero colours, returns a 1×1 zero grid. |
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
| `int_abs` | `(Int) → Int` | 1.00 | Return \|n\|. |
| `int_decrement` | `(Int) → Int` | 1.00 | Return n - 1. |
| `int_double` | `(Int) → Int` | 1.00 | Return 2 * n. |
| `int_half` | `(Int) → Int` | 1.00 | Return n // 2 (integer division, rounds toward negative infinity). |
| `int_identity` | `(Int) → Int` | 0.00 | Return the integer unchanged. Useful as a no-op leaf. |
| `int_increment` | `(Int) → Int` | 1.00 | Return n + 1. |
| `int_list_first` | `(IntList) → Int` | 1.00 | Return the first integer; raises on empty list. |
| `int_list_length` | `(IntList) → Int` | 1.00 | Return the number of elements in the int list. |
| `int_list_max` | `(IntList) → Int` | 1.00 | Return the maximum integer; raises on empty list. |
| `int_list_min` | `(IntList) → Int` | 1.00 | Return the minimum integer; raises on empty list. |
| `int_list_sum` | `(IntList) → Int` | 1.00 | Return the sum of the integers; 0 for an empty list. |
| `int_negate` | `(Int) → Int` | 1.00 | Return -n. |
| `int_square` | `(Int) → Int` | 1.00 | Return n * n. |
| `int_triple` | `(Int) → Int` | 1.00 | Return 3 * n. |
| `object_count` | `(ObjectSet) → Int` | 1.00 | Number of objects in the set (≥ 0). |
| `string_length` | `(String) → Int` | 1.00 | Return the number of characters in the string. |
| `string_list_count_nonempty` | `(StringList) → Int` | 1.00 | Return the number of elements that are not the empty string. |
| `string_list_length` | `(StringList) → Int` | 1.00 | Return the number of elements in the list. |
| `string_to_int` | `(String) → Int` | 1.00 | Parse the string as a base-10 integer. Raises ``ValueError`` on non-numeric input so the executor sandbox prunes the candidate. |

### Output type: other

| Name | Signature | Cost | Description |
|---|---|---|---|
| `int_list_reverse` | `(IntList) → IntList` | 1.00 | Return the list of ints in reverse order. |
| `int_list_sort` | `(IntList) → IntList` | 1.00 | Return the list of ints sorted ascending. |
| `int_to_string` | `(Int) → String` | 1.00 | Return the base-10 decimal string representation of the integer. |
| `string_capitalize` | `(String) → String` | 1.00 | Return the string with the first char uppercased and rest lowercased. |
| `string_collapse_spaces` | `(String) → String` | 1.00 | Replace any run of whitespace with a single space (no leading/trailing strip). |
| `string_extract_email` | `(String) → String` | 1.00 | Return the first email-like token (\S+@\S+\.\S+) or '' if none. |
| `string_extract_url` | `(String) → String` | 1.00 | Return the first http(s):// URL-like token, or '' if none. |
| `string_first_digit_run` | `(String) → String` | 1.00 | Return the first contiguous run of digit characters, or '' if none. |
| `string_first_word` | `(String) → String` | 1.00 | Return the first whitespace-delimited word, or '' for an empty input. |
| `string_identity` | `(String) → String` | 0.00 | Return the string unchanged. Useful as a no-op leaf. |
| `string_join_comma` | `(StringList) → String` | 1.00 | Join a list of strings with comma+space. |
| `string_join_space` | `(StringList) → String` | 1.00 | Join a list of strings with single spaces. |
| `string_keep_alphanumeric` | `(String) → String` | 1.00 | Return only the letters and digits of the string, in original order. |
| `string_keep_digits` | `(String) → String` | 1.00 | Return only the digit characters of the string, in original order. |
| `string_keep_letters` | `(String) → String` | 1.00 | Return only the alphabetic characters of the string, in original order. |
| `string_last_digit_run` | `(String) → String` | 1.00 | Return the last contiguous run of digit characters, or '' if none. |
| `string_last_word` | `(String) → String` | 1.00 | Return the last whitespace-delimited word, or '' for an empty input. |
| `string_list_first` | `(StringList) → String` | 1.00 | Return the first element, or '' for an empty list. |
| `string_list_last` | `(StringList) → String` | 1.00 | Return the last element, or '' for an empty list. |
| `string_lower` | `(String) → String` | 1.00 | Return the string in all-lowercase. |
| `string_remove_digits` | `(String) → String` | 1.00 | Return the string with every digit character removed. |
| `string_remove_letters` | `(String) → String` | 1.00 | Return the string with every alphabetic character removed. |
| `string_remove_punctuation` | `(String) → String` | 1.00 | Return the string with every non-alphanumeric, non-whitespace character removed. |
| `string_remove_spaces` | `(String) → String` | 1.00 | Return the string with every whitespace character removed. |
| `string_replace_dash_with_space` | `(String) → String` | 1.00 | Replace every '-' with a space. Common in slug → title transforms. |
| `string_replace_space_with_dash` | `(String) → String` | 1.00 | Replace every space with a '-'. Title → slug. |
| `string_replace_space_with_underscore` | `(String) → String` | 1.00 | Replace every space with a '_'. Title → snake_case. |
| `string_replace_underscore_with_space` | `(String) → String` | 1.00 | Replace every '_' with a space. snake_case → words. |
| `string_reverse` | `(String) → String` | 1.00 | Return the string with characters in reverse order. |
| `string_strip` | `(String) → String` | 1.00 | Return the string with leading/trailing whitespace removed. |
| `string_upper` | `(String) → String` | 1.00 | Return the string in all-uppercase. |
| `string_list_reverse` | `(StringList) → StringList` | 1.00 | Return the list with elements in reverse order. |
| `string_list_sort` | `(StringList) → StringList` | 1.00 | Return the list sorted lexicographically (Python default str ordering). |
| `string_list_unique` | `(StringList) → StringList` | 1.00 | Deduplicate the list, preserving first-seen order. |
| `string_split_comma` | `(String) → StringList` | 1.00 | Split on commas. Empty fields kept (use string_strip on parts to clean). |
| `string_split_space` | `(String) → StringList` | 1.00 | Split on any whitespace run; collapse consecutive separators. |

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

