"""
ARC-AGI-3 SDK Validation — Phase 0.

Run: python -m jarvis.arc.validate_sdk

Validates all assumptions about the ARC-AGI-3 SDK before any implementation.
Results saved to arc_agi3_validation.json for programmatic use.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path


def validate() -> dict:
    print("=" * 60)
    print("  Cognithor ARC-AGI-3 SDK Validation (Phase 0)")
    print("=" * 60)

    results: dict = {}
    errors: list[str] = []

    # ------------------------------------------------------------------ 1
    print("\n[1/11] arc-agi import...")
    try:
        import arc_agi

        results["arc_agi_version"] = getattr(arc_agi, "__version__", "unknown")
        print(f"  [OK] arc_agi {results['arc_agi_version']}")
    except ImportError as e:
        print(f"  [FAIL] {e}")
        print("    -> pip install arc-agi")
        _save({"import_error": str(e)}, [str(e)])
        sys.exit(1)

    try:
        import arcengine

        results["arcengine_version"] = getattr(arcengine, "__version__", "unknown")
        print(f"  [OK] arcengine {results['arcengine_version']}")
    except ImportError as e:
        print(f"  [FAIL] arcengine: {e}")
        errors.append(f"arcengine import: {e}")

    # ------------------------------------------------------------------ 2
    print("\n[2/11] GameAction + GameState enums...")
    try:
        from arcengine import GameAction, GameState

        all_actions = list(GameAction)
        results["actions"] = []
        for a in all_actions:
            info: dict = {"name": str(a), "value": a.value if hasattr(a, "value") else str(a)}
            if hasattr(a, "is_simple"):
                info["is_simple"] = a.is_simple()
            if hasattr(a, "is_complex"):
                info["is_complex"] = a.is_complex()
            results["actions"].append(info)
            simple = info.get("is_simple", "?")
            cpx = info.get("is_complex", "?")
            print(f"  {a}: simple={simple}, complex={cpx}")

        all_states = [s for s in dir(GameState) if not s.startswith("_") and s.isupper()]
        results["game_states"] = all_states
        print(f"  GameState values: {all_states}")

        # Check for RESET
        has_reset = any("RESET" in str(a).upper() for a in all_actions)
        results["has_reset_action"] = has_reset
        print(f"  Has RESET action: {has_reset}")

    except Exception as e:
        print(f"  [FAIL] {e}")
        errors.append(f"enums: {e}")

    # ------------------------------------------------------------------ 3
    print("\n[3/11] Arcade class...")
    try:
        import arc_agi

        arc = arc_agi.Arcade()
        methods = [m for m in dir(arc) if not m.startswith("_")]
        results["arcade_methods"] = methods
        print(f"  Methods: {methods}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        errors.append(f"Arcade: {e}")
        _save(results, errors)
        sys.exit(1)

    # ------------------------------------------------------------------ 4
    print("\n[4/11] Create environment 'ls20'...")
    env = None
    try:
        env = arc.make("ls20")
        if env is None:
            print("  [FAIL] arc.make('ls20') returned None")
            errors.append("env is None")
        else:
            env_methods = [m for m in dir(env) if not m.startswith("_")]
            results["env_type"] = type(env).__name__
            results["env_methods"] = env_methods
            print(f"  [OK] type={type(env).__name__}")
            print(f"  Methods: {env_methods}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        errors.append(f"make: {e}")

    if env is None:
        print("\n  Cannot continue without environment.")
        _save(results, errors)
        sys.exit(1)

    # ------------------------------------------------------------------ 5
    print("\n[5/11] env.reset() — observation format...")
    obs = None
    try:
        obs = env.reset()
        results["obs_type"] = type(obs).__name__
        # Collect all non-callable attributes
        attrs = {}
        for a in dir(obs):
            if a.startswith("_"):
                continue
            val = getattr(obs, a, None)
            if callable(val):
                continue
            attrs[a] = type(val).__name__
        results["obs_attributes"] = attrs
        print(f"  [OK] type={type(obs).__name__}")
        print(f"  Attributes: {attrs}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        errors.append(f"reset: {e}")

    # ------------------------------------------------------------------ 6
    print("\n[6/11] Grid format identification...")
    if obs is not None:
        import numpy as np

        grid_found = False
        for attr_name in ["frame", "frame_data", "grid", "pixels", "data", "image"]:
            val = getattr(obs, attr_name, None)
            if val is None:
                continue
            try:
                arr = np.array(val)
                results["grid_attribute"] = attr_name
                results["grid_shape"] = str(arr.shape)
                results["grid_dtype"] = str(arr.dtype)
                results["grid_min"] = int(arr.min()) if arr.size > 0 else None
                results["grid_max"] = int(arr.max()) if arr.size > 0 else None
                results["grid_size"] = int(arr.size)
                results["grid_first_values"] = (
                    arr.flat[:10].tolist() if arr.size >= 10 else arr.flat[:].tolist()
                )
                print(f"  [OK] Found in obs.{attr_name}")
                print(f"    shape={arr.shape}, dtype={arr.dtype}")
                print(f"    min={arr.min()}, max={arr.max()}")
                print(f"    first 10 values: {results['grid_first_values']}")
                grid_found = True
                break
            except Exception as e:
                print(f"  obs.{attr_name} exists but conversion failed: {e}")

        if not grid_found:
            print("  [FAIL] No grid attribute found!")
            print(f"    All attributes: {results.get('obs_attributes', {})}")
            errors.append("No grid attribute")
            results["grid_attribute"] = "NOT_FOUND"
    else:
        print("  [SKIP] No observation available")

    # ------------------------------------------------------------------ 7
    print("\n[7/11] obs.state value...")
    if obs is not None:
        state_val = getattr(obs, "state", "MISSING")
        results["obs_state"] = str(state_val)
        print(f"  obs.state = {state_val} (type: {type(state_val).__name__})")
    else:
        print("  [SKIP]")

    # ------------------------------------------------------------------ 8
    print("\n[8/11] obs.levels_completed...")
    if obs is not None:
        lc = getattr(obs, "levels_completed", "MISSING")
        results["has_levels_completed"] = lc != "MISSING"
        print(f"  obs.levels_completed = {lc}")
        if lc == "MISSING":
            # Check for alternative names
            for alt in ["level", "current_level", "levels_won", "completed_levels"]:
                alt_val = getattr(obs, alt, None)
                if alt_val is not None:
                    print(f"  [INFO] Found alternative: obs.{alt} = {alt_val}")
                    results["levels_completed_alt"] = alt
    else:
        print("  [SKIP]")

    # ------------------------------------------------------------------ 9
    print("\n[9/11] env.action_space...")
    if hasattr(env, "action_space"):
        asp = env.action_space
        results["action_space"] = str(asp)
        results["action_space_type"] = type(asp).__name__
        print(f"  [OK] type={type(asp).__name__}")
        print(f"  value: {asp}")
    else:
        print("  [WARN] env.action_space not found")
        results["action_space"] = "NOT_FOUND"
        errors.append("action_space missing")

    # ------------------------------------------------------------------ 10
    print("\n[10/11] env.step() return format...")
    try:
        from arcengine import GameAction

        obs2 = env.step(GameAction.ACTION1)
        results["step_return_type"] = type(obs2).__name__
        step_attrs = {}
        for a in dir(obs2):
            if a.startswith("_"):
                continue
            val = getattr(obs2, a, None)
            if callable(val):
                continue
            step_attrs[a] = type(val).__name__
        results["step_return_attrs"] = step_attrs
        print(f"  [OK] type={type(obs2).__name__}")
        print(f"  Attributes: {step_attrs}")
        if hasattr(obs2, "state"):
            print(f"  state after ACTION1: {obs2.state}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        errors.append(f"step: {e}")

    # ------------------------------------------------------------------ 11
    print("\n[11/11] Scorecard format...")
    try:
        sc = arc.get_scorecard()
        if sc is not None:
            results["scorecard_type"] = type(sc).__name__
            sc_attrs = {}
            for a in dir(sc):
                if a.startswith("_"):
                    continue
                val = getattr(sc, a, None)
                if callable(val):
                    continue
                sc_attrs[a] = str(val)[:200]
            results["scorecard_attrs"] = sc_attrs
            print(f"  [OK] type={type(sc).__name__}")
            print(f"  Attributes: {sc_attrs}")
        else:
            results["scorecard_type"] = "None"
            print("  Scorecard is None (expected before game completion)")
    except Exception as e:
        print(f"  [FAIL] {e}")
        errors.append(f"scorecard: {e}")

    # ------------------------------------------------------------------ extra
    print("\n[EXTRA] arc.make() signature...")
    try:
        sig = inspect.signature(arc.make)
        params = list(sig.parameters.keys())
        results["make_params"] = params
        results["has_save_recording"] = "save_recording" in params
        print(f"  Parameters: {params}")
        print(f"  Has save_recording: {results['has_save_recording']}")
    except Exception as e:
        print(f"  Could not inspect: {e}")

    # ------------------------------------------------------------------ summary
    print("\n" + "=" * 60)
    if errors:
        print(f"  [WARN] {len(errors)} issues found:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  [OK] All checks passed!")
    print("=" * 60)

    _save(results, errors)
    return results


def _save(results: dict, errors: list[str]) -> None:
    output = {"results": results, "errors": errors}
    output_path = Path("arc_agi3_validation.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    validate()
