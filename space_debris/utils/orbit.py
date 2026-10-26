"""Tracking filters and trajectory/orbit estimation for detected objects.

Two layers:

1. ``ConstantVelocityFilter`` / ``TrajectoryBank`` -- per-track Kalman filter in
   normalised image coordinates. Used to smooth detections and to bridge
   frames where the detector drops the object.

2. ``estimate_trajectory`` -- turns a smoothed pixel track into rates (px/s,
   deg/s) and, *if the observer-to-object range is supplied*, into a physical
   speed and a circular-orbit consistency check.

Everything here is numpy-only (no scipy/filterpy) and reports the assumptions
it relied on, because a camera image alone cannot determine an orbit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MU_EARTH = 398600.4418   # km^3 / s^2 (Earth gravitational parameter)
R_EARTH = 6371.0         # km


class ConstantVelocityFilter:
    """Kalman filter over state [cx, cy, vx, vy] (normalised image units).

    Measurement model observes position only; velocity is inferred.
    """

    def __init__(self, cx: float, cy: float, dt: float = 1.0,
                 process_var: float = 1e-3, meas_var: float = 1e-4):
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 1.0
        self.Q = np.eye(4) * process_var
        self.R = np.eye(2) * meas_var
        self.dt = dt
        self._F = np.eye(4)

    def _transition(self, dt: float) -> np.ndarray:
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def update(self, cx: float, cy: float, dt: float | None = None) -> np.ndarray:
        """Predict with ``dt`` then correct with the new measurement."""
        dt = self.dt if dt is None else dt
        F = self._transition(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        z = np.array([cx, cy])
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        return self.position

    def predict_only(self, dt: float) -> np.ndarray:
        """Roll the filter forward without a measurement (occluded frames)."""
        F = self._transition(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.position

    @property
    def position(self) -> np.ndarray:
        return self.x[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:4]

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def to_dict(self) -> dict:
        pos = self.position
        vel = self.velocity
        return {
            "cx": float(pos[0]),
            "cy": float(pos[1]),
            "vx": float(vel[0]),
            "vy": float(vel[1]),
            "speed_norm_per_s": self.speed,
        }


class TrajectoryBank:
    """Holds one filter + observation history per track id."""

    def __init__(self, **filter_kwargs):
        self.filters: dict[int, ConstantVelocityFilter] = {}
        self.history: dict[int, list[tuple[float, float, float]]] = {}
        self.filter_kwargs = filter_kwargs

    def update(self, track_id: int, cx: float, cy: float,
               t: float, dt: float | None = None) -> ConstantVelocityFilter:
        if track_id not in self.filters:
            filt = ConstantVelocityFilter(cx, cy, dt=dt or 1.0, **self.filter_kwargs)
            self.filters[track_id] = filt
            self.history[track_id] = [(t, cx, cy)]
            return filt
        if dt is None:
            prev_t = self.history[track_id][-1][0]
            dt = max(t - prev_t, 1e-6)
        self.history[track_id].append((t, cx, cy))
        return self.filters[track_id].update(cx, cy, dt)

    def miss(self, track_id: int, dt: float = 1.0) -> np.ndarray | None:
        """Advance a lost track without a measurement."""
        filt = self.filters.get(track_id)
        return None if filt is None else filt.predict_only(dt)

    def points(self, track_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (times, positions) for a track."""
        hist = self.get_history(track_id)
        if not hist:
            return np.zeros(0), np.zeros((0, 2))
        arr = np.asarray(hist, dtype=float)
        return arr[:, 0], arr[:, 1:3]

    def get_history(self, track_id: int) -> list[tuple[float, float, float]]:
        return self.history.get(track_id, [])

    def ids(self) -> list[int]:
        return list(self.filters.keys())


def fov_vertical_deg(fov_horizontal_deg: float, width: int, height: int) -> float:
    """Vertical FOV implied by a horizontal FOV and the sensor aspect."""
    half = np.radians(fov_horizontal_deg) / 2.0
    aspect = width / max(height, 1)
    return float(np.degrees(2.0 * np.arctan(np.tan(half) / aspect)))


def estimate_trajectory(
    times: np.ndarray,
    points: np.ndarray,
    *,
    width: int,
    height: int,
    fov_deg: float = 60.0,
    range_km: float | None = None,
    min_seconds: float = 0.0,
) -> dict:
    """Estimate motion from a normalised (cx, cy) pixel track.

    Args:
        times: seconds since the first frame.
        points: (N, 2) normalised image coordinates.
        width/height: image size in pixels (for px/s reporting).
        fov_deg: horizontal field of view of the camera.
        range_km: observer-to-object range. Leave None if unknown -- the
            physical speed and orbit checks are then omitted instead of
            guessed.

    Returns a JSON-serialisable dict. Any value that needs an assumption the
    caller did not supply is ``None`` and listed under ``missing``.
    """
    times = np.asarray(times, dtype=float)
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    result = {
        "n_points": int(len(points)),
        "duration_s": None,
        "pixel_velocity_px_s": None,
        "angular_rate_deg_s": None,
        "speed_km_s": None,
        "circular_orbit_altitude_km": None,
        "circular_orbit_speed_km_s": None,
        "consistent_with_circular_orbit": None,
        "missing": [],
        "assumptions": {
            "fov_deg_horizontal": fov_deg,
            "range_km": range_km,
        },
    }
    if len(points) < 2:
        result["missing"].append("needs_at_least_2_points")
        return result

    duration = float(times[-1] - times[0])
    result["duration_s"] = duration
    if duration < max(min_seconds, 1e-6):
        result["missing"].append("track_too_short")
        return result

    # Least-squares straight-line fit: p(t) = p0 + v * t
    A = np.vstack([np.ones_like(times), times]).T
    coef, *_ = np.linalg.lstsq(A, points, rcond=None)
    v_norm = coef[1]  # normalised units / s
    residual = points - A @ coef
    rms = float(np.sqrt(np.mean(residual ** 2)))

    v_px = v_norm * np.array([width, height], dtype=float)
    result["pixel_velocity_px_s"] = [float(v_px[0]), float(v_px[1])]
    result["pixel_speed_px_s"] = float(np.hypot(*v_px))
    result["fit_rms_norm"] = rms

    fov_v = fov_vertical_deg(fov_deg, width, height)
    omega_deg = [v_norm[0] * fov_deg, v_norm[1] * fov_v]
    omega_rad = np.radians(omega_deg)
    result["angular_rate_deg_s"] = [float(omega_deg[0]), float(omega_deg[1])]
    result["angular_rate_rad_s"] = [float(omega_rad[0]), float(omega_rad[1])]

    if range_km is None:
        result["missing"].append("range_km_needed_for_physical_speed")
        return result

    speed = float(np.linalg.norm(omega_rad) * range_km)
    result["speed_km_s"] = speed

    r = R_EARTH + max(range_km, 1.0)
    v_circ = float(np.sqrt(MU_EARTH / r))
    result["circular_orbit_speed_km_s"] = v_circ
    result["circular_orbit_altitude_km"] = float(MU_EARTH / (speed ** 2) - R_EARTH) if speed > 0 else None
    # 10% band: is the measured speed compatible with a circular orbit here?
    result["consistent_with_circular_orbit"] = bool(abs(speed - v_circ) <= 0.1 * v_circ)
    result["assumptions"]["mu_earth_km3_s2"] = MU_EARTH
    return result


def summarize_track(track_id: int, times: np.ndarray, points: np.ndarray,
                    **kwargs) -> dict:
    """Per-track report: Kalman end-state + trajectory estimate."""
    est = estimate_trajectory(times, points, **kwargs)
    report = {"track_id": track_id, "trajectory": est}
    if len(points) >= 1:
        filt = ConstantVelocityFilter(float(points[-1][0]), float(points[-1][1]))
        for i in range(1, len(points)):
            dt = float(times[i] - times[i - 1]) if len(times) == len(points) else 1.0
            filt.update(float(points[i, 0]), float(points[i, 1]), max(dt, 1e-6))
        report["kalman"] = filt.to_dict()
    return report


def orbit_from_jsonl(path, fov_deg: float = 60.0, range_km: float | None = None,
                     width: int | None = None, height: int | None = None,
                     out_json: str | None = None) -> dict:
    """Offline orbit stage over a JSONL produced by ``models.track.track``.

    Groups each track's ``(t, cx, cy)`` observations and returns a per-track
    report (plus a file, if ``out_json`` is given).
    """
    import json
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if width is None or height is None:
        for rec in records:
            if rec.get("image_size"):
                width, height = (int(v) for v in rec["image_size"])
                break
    if width is None or height is None:
        for rec in records:
            src = rec.get("source")
            if not src:
                continue
            try:
                import cv2
                img = cv2.imread(str(src))
                if img is not None:
                    height, width = img.shape[:2]
                    break
            except Exception:
                continue
    if width is None or height is None:
        raise ValueError("image size unknown: pass --width/--height or ensure "
                         "records have readable 'source' paths")

    per_track: dict[int, list[tuple[float, float, float]]] = {}
    for rec in records:
        t = float(rec.get("t", rec.get("frame", 0)))
        for det in rec.get("detections", []):
            tid = det.get("track_id")
            if tid is None:
                continue
            cx, cy = float(det["cx"]), float(det["cy"])
            # tolerate legacy records that stored pixel centres
            if abs(cx) > 1.5 or abs(cy) > 1.5:
                cx, cy = cx / width, cy / height
            per_track.setdefault(int(tid), []).append((t, cx, cy))

    report = []
    for tid in sorted(per_track):
        obs = per_track[tid]
        times = np.asarray([o[0] for o in obs], dtype=float)
        points = np.asarray([[o[1], o[2]] for o in obs], dtype=float)
        if len(points) < 2:
            continue
        report.append(summarize_track(
            tid, times, points, width=int(width), height=int(height),
            fov_deg=fov_deg, range_km=range_km,
        ))

    if out_json:
        outp = Path(out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2))
    return {"tracks": len(report), "frames": len(records),
            "image_size": [int(width), int(height)], "report": report}
