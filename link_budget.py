"""Satellite link budget calculations using dB-domain engineering formulas."""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Dict, Any

BOLTZMANN_DBW_PER_K_HZ = -228.6  # 10*log10(k), W/K/Hz
FSPL_CONSTANT_GHZ_KM = 92.45


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero.")
    return value


def fspl_db(frequency_ghz: float, distance_km: float) -> float:
    """Free-space path loss in dB for frequency in GHz and distance in km."""
    f = _positive("frequency_ghz", frequency_ghz)
    d = _positive("distance_km", distance_km)
    return FSPL_CONSTANT_GHZ_KM + 20 * math.log10(f) + 20 * math.log10(d)


def cn0_db_hz(eirp_dbw: float, gt_db_per_k: float,
              path_loss_db: float, other_losses_db: float = 0.0) -> float:
    """Carrier-to-noise-density ratio C/N0 in dB-Hz."""
    vals = (eirp_dbw, gt_db_per_k, path_loss_db, other_losses_db)
    if not all(math.isfinite(float(v)) for v in vals):
        raise ValueError("All link budget values must be finite numbers.")
    return float(eirp_dbw) + float(gt_db_per_k) - float(path_loss_db) \
        - float(other_losses_db) - BOLTZMANN_DBW_PER_K_HZ


def cn_db_from_cn0(cn0: float, bandwidth_hz: float) -> float:
    """Convert C/N0 (dB-Hz) to C/N (dB) for noise bandwidth in Hz."""
    b = _positive("bandwidth_hz", bandwidth_hz)
    if not math.isfinite(float(cn0)):
        raise ValueError("cn0 must be a finite number.")
    return float(cn0) - 10 * math.log10(b)


def ebn0_db_from_cn0(cn0: float, bit_rate_bps: float) -> float:
    """Compute Eb/N0 (dB) from C/N0 (dB-Hz) and bit rate (bit/s)."""
    r = _positive("bit_rate_bps", bit_rate_bps)
    if not math.isfinite(float(cn0)):
        raise ValueError("cn0 must be a finite number.")
    return float(cn0) - 10 * math.log10(r)


def combine_cn_db(*cn_values_db: float) -> float:
    """Combine independent noise contributions expressed as C/N ratios in dB."""
    if not cn_values_db:
        raise ValueError("Provide at least one C/N value.")
    values = [float(v) for v in cn_values_db]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("C/N values must be finite numbers.")
    # 1/(C/N_total) = sum(1/(C/N_i)); use log-sum-exp for numerical stability.
    min_exp = min(-v / 10 for v in values)
    sum_linear_scaled = sum(10 ** ((-v / 10) - min_exp) for v in values)
    return -10 * (min_exp + math.log10(sum_linear_scaled))


@dataclass
class HopInput:
    frequency_ghz: float
    eirp_dbw: float
    gt_db_per_k: float
    distance_km: float
    other_losses_db: float = 0.0

    def calculate(self, bandwidth_mhz: float, bit_rate_mbps: float) -> Dict[str, float]:
        bw_hz = _positive("bandwidth_mhz", bandwidth_mhz) * 1e6
        rate_bps = _positive("bit_rate_mbps", bit_rate_mbps) * 1e6
        loss = fspl_db(self.frequency_ghz, self.distance_km)
        c0 = cn0_db_hz(self.eirp_dbw, self.gt_db_per_k, loss, self.other_losses_db)
        return {
            "fspl_db": loss,
            "cn0_db_hz": c0,
            "cn_db": cn_db_from_cn0(c0, bw_hz),
            "ebn0_db": ebn0_db_from_cn0(c0, rate_bps),
        }


def calculate_link(uplink: HopInput, downlink: HopInput,
                   bandwidth_mhz: float, bit_rate_mbps: float) -> Dict[str, Any]:
    """Calculate uplink, downlink, and combined C/N for a transparent bent-pipe link.

    Simplified model: assumes uplink and downlink C/N values refer to the same
    equivalent noise bandwidth. Transponder-added noise, interference, rain
    statistics, nonlinear effects, and MODCOD thresholds are not modeled.
    """
    up = uplink.calculate(bandwidth_mhz, bit_rate_mbps)
    down = downlink.calculate(bandwidth_mhz, bit_rate_mbps)
    combined = combine_cn_db(up["cn_db"], down["cn_db"])
    return {
        "inputs": {
            "uplink": asdict(uplink),
            "downlink": asdict(downlink),
            "bandwidth_mhz": float(bandwidth_mhz),
            "bit_rate_mbps": float(bit_rate_mbps),
        },
        "uplink": up,
        "downlink": down,
        "combined_cn_db": combined,
        "model_notes": [
            "Free-space path loss only; additional losses are user-supplied.",
            "Uplink and downlink C/N are combined as independent noise contributions.",
            "Same equivalent noise bandwidth is assumed for both hops.",
            "Transponder noise, interference, rain-fade statistics, pointing/polarization details, "
            "nonlinear operation and MODCOD thresholds are not modeled.",
        ],
    }
