"""CNN Action Predictor — optional PyTorch-based online learner for ARC-AGI-3.

This module is OPTIONAL. All torch imports are guarded so the module stays
importable even when torch is not installed.
"""

from __future__ import annotations

__all__ = [
    "_TORCH_AVAILABLE",
    "ActionPredictor",
    "OnlineTrainer",
]

import hashlib
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Optional torch import
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_COLORS: int = 13  # colour indices 0-12
_N_ACTIONS: int = 8  # RESET + ACTION1-7
_GRID_H: int = 64
_GRID_W: int = 64

# ---------------------------------------------------------------------------
# ActionPredictor
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class ActionPredictor(nn.Module):  # type: ignore[misc]
        """CNN that predicts action probabilities and coordinate heat-maps.

        Input shape:  ``(batch, n_colors, 64, 64)`` — one-hot encoded grid.
        Outputs:
            action_logits: ``(batch, n_actions)``  — passed through Sigmoid.
            coord_map:     ``(batch, 1, 64, 64)``  — passed through Sigmoid.
        """

        def __init__(
            self,
            n_colors: int = _N_COLORS,
            n_actions: int = _N_ACTIONS,
        ) -> None:
            super().__init__()
            self.n_colors = n_colors
            self.n_actions = n_actions

            # Encoder: stride-2 downsampling  64→32→16→8
            self.encoder = nn.Sequential(
                # Block 1: n_colors → 32,  64×64 → 32×32
                nn.Conv2d(n_colors, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                # Block 2: 32 → 64,  32×32 → 16×16
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                # Block 3: 64 → 128,  16×16 → 8×8
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                # Block 4: 128 → 128,  8×8 → 4×4  (kept at 8×8 via stride=1)
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            # Spatial size after encoder: 8×8  (three stride-2 + one stride-1)
            _flat_size = 128 * 8 * 8

            # Action head
            self.action_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(_flat_size, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, n_actions),
                nn.Sigmoid(),
            )

            # Coordinate head: upsample 8×8 back to 64×64
            self.coord_head = nn.Sequential(
                # 8 → 16
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                # 16 → 32
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                # 32 → 64
                nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Run forward pass.

            Parameters
            ----------
            x:
                Float tensor of shape ``(batch, n_colors, 64, 64)``.

            Returns
            -------
            action_probs:
                ``(batch, n_actions)`` probabilities in [0, 1].
            coord_map:
                ``(batch, 1, 64, 64)`` heat-map in [0, 1].
            """
            features = self.encoder(x)
            action_probs = self.action_head(features)
            coord_map = self.coord_head(features)
            return action_probs, coord_map

    # -----------------------------------------------------------------------
    # OnlineTrainer
    # -----------------------------------------------------------------------

    class OnlineTrainer:
        """Online experience-replay trainer for :class:`ActionPredictor`.

        Parameters
        ----------
        device:
            Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
        buffer_size:
            Maximum number of experiences kept in the replay buffer.
        """

        def __init__(
            self,
            device: str = "cuda",
            buffer_size: int = 200_000,
        ) -> None:
            self._raw_device = device
            self._buffer_size = buffer_size
            self._device = torch.device(device if torch.cuda.is_available() else "cpu")

            self.model = ActionPredictor().to(self._device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

            # Replay buffer: each entry is (grid, action_idx, coord_or_None)
            self._buffer: deque[tuple[np.ndarray, int, tuple[int, int] | None]] = deque(
                maxlen=buffer_size
            )
            self._seen_hashes: set[str] = set()

            self.train_interval: int = 32
            self.batch_size: int = 64
            self._steps_since_train: int = 0

        # ------------------------------------------------------------------
        # Public API
        # ------------------------------------------------------------------

        def add_experience(
            self,
            grid: np.ndarray,
            action_idx: int,
            coord: tuple[int, int] | None = None,
            frame_changed: bool = False,
        ) -> None:
            """Add one step-experience to the replay buffer.

            Experiences are deduplicated by ``(grid_hash, action_idx, coord)``.
            After every :attr:`train_interval` unique additions, a training step
            is triggered automatically.

            Parameters
            ----------
            grid:
                2-D int8 array of shape ``(64, 64)`` or ``(H, W)``.
            action_idx:
                Integer action index in ``[0, n_actions)``.
            coord:
                Optional ``(row, col)`` coordinate hint.
            frame_changed:
                Reserved for caller context; currently unused internally.
            """
            key = self._make_key(grid, action_idx, coord)
            if key in self._seen_hashes:
                return
            self._seen_hashes.add(key)

            self._buffer.append((grid.copy(), action_idx, coord))
            self._steps_since_train += 1

            if (
                self._steps_since_train >= self.train_interval
                and len(self._buffer) >= self.batch_size
            ):
                self._train_step()
                self._steps_since_train = 0

        def predict(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Return predicted action and coordinate probabilities for *grid*.

            Parameters
            ----------
            grid:
                2-D int8 array of shape ``(64, 64)``.

            Returns
            -------
            action_probs:
                1-D float32 array of shape ``(n_actions,)`` in [0, 1].
            coord_probs:
                2-D float32 array of shape ``(64, 64)`` in [0, 1].
            """
            self.model.eval()
            with torch.no_grad():
                tensor = self._grid_to_tensor(grid).unsqueeze(0).to(self._device)
                action_probs_t, coord_map_t = self.model(tensor)
            action_probs = action_probs_t.squeeze(0).cpu().numpy().astype(np.float32)
            coord_probs = coord_map_t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            return action_probs, coord_probs

        def reset_for_new_level(self) -> None:
            """Clear the replay buffer and reinitialise the model and optimiser."""
            self._buffer.clear()
            self._seen_hashes.clear()
            self._steps_since_train = 0

            self.model = ActionPredictor().to(self._device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # ------------------------------------------------------------------
        # Private helpers
        # ------------------------------------------------------------------

        def _train_step(self) -> float:
            """Sample a mini-batch and perform one gradient update.

            Returns
            -------
            float
                The scalar loss value for monitoring.
            """
            self.model.train()

            actual_batch = min(self.batch_size, len(self._buffer))
            indices = np.random.choice(len(self._buffer), size=actual_batch, replace=False)
            batch = [list(self._buffer)[i] for i in indices]

            grids, action_targets, coord_targets = [], [], []
            for g, a, c in batch:
                grids.append(self._grid_to_tensor(g))
                # One-hot action target
                act_t = torch.zeros(_N_ACTIONS)
                if 0 <= a < _N_ACTIONS:
                    act_t[a] = 1.0
                action_targets.append(act_t)
                # Coordinate target: Gaussian blob or zeros
                coord_t = torch.zeros(1, _GRID_H, _GRID_W)
                if c is not None:
                    row, col = int(c[0]), int(c[1])
                    row = max(0, min(row, _GRID_H - 1))
                    col = max(0, min(col, _GRID_W - 1))
                    coord_t[0, row, col] = 1.0
                coord_targets.append(coord_t)

            grid_batch = torch.stack(grids).to(self._device)
            act_batch = torch.stack(action_targets).to(self._device)
            coord_batch = torch.stack(coord_targets).to(self._device)

            pred_actions, pred_coords = self.model(grid_batch)

            loss_actions = nn.functional.binary_cross_entropy(pred_actions, act_batch)
            loss_coords = nn.functional.binary_cross_entropy(pred_coords, coord_batch)
            loss = loss_actions + loss_coords

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return float(loss.item())

        def _grid_to_tensor(self, grid: np.ndarray) -> torch.Tensor:
            """One-hot encode *grid* into a ``(n_colors, 64, 64)`` float32 tensor.

            Parameters
            ----------
            grid:
                2-D int8 array of shape ``(H, W)``; values clipped to ``[0, n_colors)``.

            Returns
            -------
            torch.Tensor
                Shape ``(13, 64, 64)`` float32.
            """
            g = np.asarray(grid, dtype=np.int64)
            if g.ndim == 3:
                g = g[0]
            # Pad or crop to (64, 64)
            h, w = g.shape
            canvas = np.zeros((_GRID_H, _GRID_W), dtype=np.int64)
            canvas[: min(h, _GRID_H), : min(w, _GRID_W)] = g[: min(h, _GRID_H), : min(w, _GRID_W)]
            # Clip values to valid color range
            canvas = np.clip(canvas, 0, _N_COLORS - 1)
            # One-hot: (13, 64, 64)
            one_hot = np.zeros((_N_COLORS, _GRID_H, _GRID_W), dtype=np.float32)
            for c in range(_N_COLORS):
                one_hot[c] = (canvas == c).astype(np.float32)
            return torch.from_numpy(one_hot)

        @staticmethod
        def _make_key(
            grid: np.ndarray,
            action_idx: int,
            coord: tuple[int, int] | None,
        ) -> str:
            """Return a short deduplication hash for ``(grid, action_idx, coord)``."""
            grid_bytes = np.asarray(grid, dtype=np.int8).tobytes()
            grid_hash = hashlib.md5(grid_bytes, usedforsecurity=False).hexdigest()
            return f"{grid_hash}:{action_idx}:{coord}"

else:
    # torch not available — provide stub classes that raise on instantiation

    class ActionPredictor:  # type: ignore[no-redef]
        """Stub: raises ImportError when torch is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "ActionPredictor requires PyTorch. Install it with: pip install torch"
            )

    class OnlineTrainer:  # type: ignore[no-redef]
        """Stub: raises ImportError when torch is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("OnlineTrainer requires PyTorch. Install it with: pip install torch")
