"""Audit the current Lab 1 CSVs; leave logs and motion commands unchanged."""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from filePlotter import read_log, scan_points

OUT = ROOT / 'plots' / 'verification'
OUT.mkdir(parents=True, exist_ok=True)
results = {}

def mean_bins(t, values):
    centers, means = [], []
    for start in np.arange(np.floor(t[0]), t[-1], 1.0):
        mask = (t >= start) & (t < start + 1)
        if np.any(mask):
            centers.append(np.mean(t[mask]))
            means.append(np.mean(values[mask]))
    return centers, means

for motion in ('circle', 'spiral', 'line'):
    logs = {s: read_log(ROOT / 'data' / f'{s}_content_{motion}.csv')
            for s in ('odom', 'imu', 'laser')}
    origin = min(log.stamps[0] for log in logs.values())
    odom, imu, laser = (logs[s] for s in ('odom', 'imu', 'laser'))
    t, it = odom.elapsed(origin), imu.elapsed(origin)
    x, y = odom.columns['x'], odom.columns['y']
    heading = np.unwrap(odom.columns['th'])
    dt, tm = np.diff(t), (t[:-1] + t[1:]) / 2
    dx, dy = np.diff(x), np.diff(y)
    speed = np.hypot(dx, dy) / dt
    hm = (heading[:-1] + heading[1:]) / 2
    forward = (dx * np.cos(hm) + dy * np.sin(hm)) / dt
    omega = np.diff(heading) / dt
    gyro = imu.columns['angular_z']
    gyro_at_odom = np.interp(tm, it, gyro)
    onset = tm[np.flatnonzero(speed > .02)[0]]
    window = (tm > onset + 1) & (tm < tm[-1] - .2)
    steady = window.copy()
    if motion == 'line':
        steady &= tm > onset + 5.5
    a, b = max(t[0], it[0]), min(t[-1], it[-1])
    integration_time = np.r_[a, it[(it > a) & (it < b)], b]
    integrated = np.trapz(np.interp(integration_time, it, gyro), integration_time)
    yaw_change = np.interp(b, t, heading) - np.interp(a, t, heading)
    m = {
        'inferred_movement_onset_s': float(onset),
        'analysis_window_s': [float(onset + 1), float(tm[-1] - .2)],
        'steady_window_s': [float(max(onset + 1, onset + 5.5 if motion == 'line' else 0)), float(tm[-1] - .2)],
        'median_speed_m_s': float(np.median(speed[steady])),
        'mean_speed_m_s': float(np.mean(speed[steady])),
        'median_omega_rad_s': float(np.median(omega[steady])),
        'gyro_odom_rate_correlation': float(np.corrcoef(omega[window], gyro_at_odom[window])[0, 1]),
        'gyro_odom_rate_rmse_rad_s': float(np.sqrt(np.mean((omega[window] - gyro_at_odom[window]) ** 2))),
        'yaw_change_rad': float(yaw_change),
        'gyro_integrated_yaw_rad': float(integrated),
        'gyro_minus_odom_yaw_deg': float(np.degrees(integrated - yaw_change)),
        'path_length_m': float(np.sum(np.hypot(dx, dy))),
        'logs': {},
    }
    for sensor, log in logs.items():
        differences = np.diff(log.stamps) / 1e9
        m['logs'][sensor] = {
            'file': str(log.path.relative_to(ROOT)),
            'sha256': hashlib.sha256(log.path.read_bytes()).hexdigest(),
            'samples': len(log.stamps),
            'duration_s': float((log.stamps[-1] - log.stamps[0]) / 1e9),
            'median_sample_rate_hz': float(1 / np.median(differences)),
            'largest_gap_s': float(np.max(differences)),
        }
    lx, ly, discarded = scan_points(laser, 0)
    ranges = np.asarray([laser.columns[f'range_{i}'][0] for i in range(720)])
    valid = np.isfinite(ranges) & (ranges > 0)
    m['lidar'] = {
        'valid_first_scan_beams': int(np.sum(valid)), 'discarded_first_scan_beams': discarded,
        'cartesian_radius_max_error_m': float(np.max(np.abs(np.hypot(lx, ly) - ranges[valid]))),
        'angle_increment_deg': float(np.degrees(laser.columns['angle_increment'][0])),
        'valid_range_min_m': float(np.min(ranges[valid])),
        'valid_range_max_m': float(np.max(ranges[valid])),
    }
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    fig.suptitle(f'{motion.title()} — engineering checks')
    axes[0].plot(tm, speed, color='#0072B2', label='Speed from odometry')
    # Commands were not logged. First lidar + half a timer period is only a guide.
    guide_start = laser.elapsed(origin)[0] + .05
    elapsed = np.maximum(tm - guide_start, 0)
    commanded_v = np.minimum(.1 * elapsed, .5) if motion == 'line' else np.full_like(tm, .3)
    commanded_v[tm < guide_start] = 0
    commanded_w = np.zeros_like(tm) if motion == 'line' else (
        .3 / np.minimum(.5 + .01 * elapsed, 1.5) if motion == 'spiral' else np.full_like(tm, .6))
    commanded_w[tm < guide_start] = 0
    axes[0].plot(tm, commanded_v, '--', color='#D55E00', label='Current-code command guide (start estimated)')
    axes[0].set(ylabel='Speed (m/s)', title='Translation: measured versus intended')
    axes[1].plot(tm, omega, color='#0072B2', alpha=.8, label='Yaw rate from odometry')
    axes[1].plot(it, gyro, color='#009E73', linewidth=.8, alpha=.75, label='IMU gyro z')
    axes[1].plot(tm, commanded_w, '--', color='#D55E00', label='Current-code command guide')
    axes[1].set(ylabel='Yaw rate (rad/s)', title='Rotation: odometry and IMU comparison')
    if motion == 'circle':
        cx, cy, c = np.linalg.lstsq(np.c_[2*x, 2*y, np.ones_like(x)], x*x + y*y, rcond=None)[0]
        radius = np.sqrt(c + cx*cx + cy*cy)
        radial = np.hypot(x - cx, y - cy)
        m['fitted_radius_m'] = float(radius)
        m['circle_radial_rmse_m'] = float(np.sqrt(np.mean((radial - radius)**2)))
        imask = (it > onset + 1) & (it < tm[-1] - .2)
        m['moving_imu_mean_ax_m_s2'] = float(np.mean(imu.columns['acc_x'][imask]))
        m['moving_imu_mean_ay_m_s2'] = float(np.mean(imu.columns['acc_y'][imask]))
        m['median_odom_centripetal_acceleration_m_s2'] = float(np.median((forward*omega)[window]))
        bt, by = mean_bins(it, imu.columns['acc_y'])
        axes[2].plot(bt, by, 'o-', color='#009E73', label='IMU a_y: 1 s sample means')
        bt, by = mean_bins(tm, forward * omega)
        axes[2].plot(bt, by, 's-', color='#0072B2', label='Odometry v × omega: 1 s sample means')
        axes[2].axhline(.18, linestyle='--', color='#D55E00', label='Ideal centripetal acceleration 0.18 m/s²')
        axes[2].set(ylabel='Acceleration (m/s²)', title='Lateral acceleration (assuming IMU axes aligned to robot)')
    elif motion == 'spiral':
        radius_mask = window & (speed > .27) & (omega > .18)
        radius = speed[radius_mask] / omega[radius_mask]
        slope, intercept = np.polyfit(tm[radius_mask] - onset, radius, 1)
        m['moving_radius_fit_slope_m_s'] = float(slope)
        m['moving_radius_fit_intercept_m'] = float(intercept)
        m['moving_radius_fit_rmse_m'] = float(np.sqrt(np.mean((radius - (slope*(tm[radius_mask]-onset)+intercept))**2)))
        low = window & (speed < .15)
        m['low_speed_duration_s'] = float(np.sum(dt[low]))
        m['low_speed_time_fraction'] = float(np.sum(dt[low])/np.sum(dt[window]))
        m['radius_fit_samples'] = int(np.sum(radius_mask))
        axes[2].scatter(tm[radius_mask], radius, s=4, color='#0072B2', label='v / omega while moving (v > 0.27, omega > 0.18)')
        axes[2].plot(tm, np.minimum(.5 + .01 * elapsed, 1.5), '--', color='#D55E00', label='Current-code turning radius guide')
        axes[2].set(ylabel='Turning radius (m)', title='Radius growth; low-speed intervals excluded only from this panel')
    else:
        cross = -(x-x[0])*np.sin(heading[0]) + (y-y[0])*np.cos(heading[0])
        m['end_cross_track_from_initial_heading_m'] = float(cross[-1])
        ramp = (tm > onset + .5) & (tm < onset + 4.5)
        slope, intercept = np.polyfit(tm[ramp] - onset, speed[ramp], 1)
        m['ramp_fit_acceleration_m_s2'] = float(slope)
        ramp_imu = (it > onset + .5) & (it < onset + 4.5)
        m['ramp_imu_mean_ax_m_s2'] = float(np.mean(imu.columns['acc_x'][ramp_imu]))
        axes[2].plot(t, cross, color='#0072B2', label='Deviation from initial-heading straight line')
        axes[2].axhline(0, color='#D55E00', linestyle='--', label='Ideal straight line')
        axes[2].set(ylabel='Cross-track error (m)', title='Negative deviation is to the right of the initial heading')
    for ax in axes:
        ax.set_xlabel('Time since first logged sensor sample (s)')
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.savefig(OUT / f'{motion}_checks.png', dpi=160)
    plt.close(fig)
    results[motion] = m

with (OUT / 'metrics.json').open('w') as handle:
    json.dump(results, handle, indent=2, allow_nan=False)
print(json.dumps({motion: {k:v for k,v in m.items() if k not in ('logs',)}
                  for motion,m in results.items()}, indent=2))
