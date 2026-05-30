"""Thermal monitoring and cooldown controls for training loops."""

from __future__ import annotations

import time
from pathlib import Path

from synthoseis_pre_train._checkpoint import _save_checkpoint
from synthoseis_pre_train.gpu_utils import get_cpu_temperature_c, get_thermal_pressure_level


class ThermalGuard:
    def __init__(self, max_c: float, cooldown_sec: int,
                 check_every_batches: int, output_dir: Path,
                 pressure_trip_level: str = "serious"):
        self.max_c = max_c
        self.cooldown_sec = max(0, cooldown_sec)
        self.check_every_batches = max(1, check_every_batches)
        self.output_dir = output_dir
        self.pressure_trip_level = (pressure_trip_level or "off").strip().lower()
        self._pressure_order = {
            "nominal": 0,
            "fair": 1,
            "serious": 2,
            "critical": 3,
        }
        self._pressure_trip_idx = (
            None if self.pressure_trip_level == "off"
            else self._pressure_order[self.pressure_trip_level]
        )
        self.last_temp_c = None
        self.last_pressure_level = None

    def sample_temperature(self, batch_idx: int):
        """Sample CPU temperature at the configured periodic interval."""
        if self.max_c <= 0 and self._pressure_trip_idx is None:
            return None
        if batch_idx % self.check_every_batches != 0:
            return self.last_temp_c
        self.last_temp_c = get_cpu_temperature_c()
        self.last_pressure_level = get_thermal_pressure_level()
        return self.last_temp_c

    def maybe_pause(self, epoch: int, ds_idx: int, batch_idx: int,
                    model, optimizer, scaler,
                    train_paths: list, val_paths: list,
                    temp_c: float | None = None,
                    ema_state: dict | None = None) -> bool:
        """Checkpoint and pause training when CPU temperature is too high."""
        if self.max_c <= 0 and self._pressure_trip_idx is None:
            return False
        if temp_c is None:
            temp_c = self.last_temp_c

        pressure_trip = False
        if self._pressure_trip_idx is not None and self.last_pressure_level is not None:
            pressure_idx = self._pressure_order.get(self.last_pressure_level.strip().lower())
            pressure_trip = pressure_idx is not None and pressure_idx >= self._pressure_trip_idx

        if temp_c is not None and temp_c >= self.max_c:
            trip_reason = f"CPU {temp_c:.1f}C >= {self.max_c:.1f}C"
        elif pressure_trip:
            trip_reason = f"thermal pressure {self.last_pressure_level}"
        else:
            return False

        ckpt_path = self.output_dir / "thermal_latest.pt"
        print(
            f"\nThermal pause: {trip_reason} "
            f"(epoch {epoch + 1}, dataset {ds_idx + 1}, batch {batch_idx})"
        )
        _save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scaler,
            epoch,
            train_loss=float("nan"),
            val_loss=float("nan"),
            train_paths=train_paths,
            val_paths=val_paths,
            ds_idx=ds_idx,
            ema_state=ema_state,
        )
        print(f"  Saved thermal checkpoint: {ckpt_path}")
        if self.cooldown_sec > 0:
            print(f"  Cooling down for {self.cooldown_sec} seconds...")
            time.sleep(self.cooldown_sec)
            print("  Resuming training after cooldown.")
        return True


def _print_thermal_monitor_status(max_c: float, pressure_trip_level: str) -> None:
    """Print whether CPU thermal monitoring is available for this run."""
    pressure_trip_level = (pressure_trip_level or "off").strip().lower()
    if max_c <= 0 and pressure_trip_level == "off":
        print("Thermal monitor: disabled")
        return

    temp_c = get_cpu_temperature_c()
    pressure = get_thermal_pressure_level()
    if temp_c is None and pressure is None:
        print("Thermal monitor: unavailable (powermetrics output could not be parsed)")
        print("  Hint: run 'sudo -v' before starting training to enable automatic thermal pausing.")
        return

    if pressure_trip_level == "off":
        pressure_msg = "off"
    else:
        pressure_msg = pressure_trip_level.capitalize()

    if temp_c is not None:
        print(f"Thermal monitor: available (current CPU {temp_c:.1f}C, threshold {max_c:.1f}C)")
        if pressure is not None:
            print(f"  Thermal pressure: {pressure}")
        print(f"  Pressure trip level: {pressure_msg}")
    else:
        print(f"Thermal monitor: available via thermal pressure only ({pressure})")
        if pressure_trip_level == "off":
            print("  Pressure-based pausing is disabled; only CPU temperature can trigger a pause.")
        else:
            print(f"  Pause trigger uses pressure levels >= {pressure_msg} when CPU temperature is unavailable.")
