"""Parameters for reproducible EEG feature measurement."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class EEGConfig:
    window_seconds: float = 4.0
    welch_seconds: float = 4.0
    min_spectral_seconds: float = 1.0
    welch_overlap_fraction: float = 0.5
    alpha_band: tuple[float, float] = (8.0, 13.0)
    alpha_min_prominence_db: float = 6.0
    line_min_prominence_db: float = 10.0
    line_peak_half_width_hz: float = 1.0
    line_baseline_inner_hz: float = 2.0
    line_baseline_outer_hz: float = 4.0
    low_frequency_band: tuple[float, float] = (0.5, 4.0)
    high_frequency_band: tuple[float, float] = (30.0, 100.0)
    total_power_band: tuple[float, float] = (0.5, 100.0)
    background_fit_band: tuple[float, float] = (2.0, 40.0)
    large_amplitude_uv: float = 500.0
    flat_step_uv: float = 0.01

    def __post_init__(self):
        bands = ("alpha_band", "low_frequency_band", "high_frequency_band",
                 "total_power_band", "background_fit_band")
        for name in bands:
            value = getattr(self, name)
            if (not isinstance(value, (tuple, list)) or len(value) != 2
                    or any(isinstance(x, bool) or not isinstance(x, (int, float))
                           or not math.isfinite(x) for x in value)
                    or not 0 < value[0] < value[1]):
                raise ValueError(f"{name} must contain two increasing positive frequencies")
            object.__setattr__(self, name, tuple(float(x) for x in value))
        for name, value in asdict(self).items():
            if name in bands:
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if min(self.window_seconds, self.welch_seconds, self.min_spectral_seconds) <= 0:
            raise ValueError("window and spectral durations must be positive")
        if self.min_spectral_seconds > self.welch_seconds:
            raise ValueError("min_spectral_seconds must be at most welch_seconds")
        if not 0 <= self.welch_overlap_fraction < 1:
            raise ValueError("welch_overlap_fraction must be in [0, 1)")
        if not (0 < self.line_peak_half_width_hz < self.line_baseline_inner_hz
                < self.line_baseline_outer_hz):
            raise ValueError("line peak and baseline widths must increase")
        if self.large_amplitude_uv <= 0:
            raise ValueError("large_amplitude_uv must be positive")

    def to_dict(self):
        return asdict(self)
