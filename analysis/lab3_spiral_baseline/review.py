"""Review the recorded baseline without altering data or running robot code."""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mte544_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kalman_filter import kalman_filter

OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'test_results/lab3_spiral_baseline/robot_pose.csv'
raw_bytes = SOURCE.read_bytes()
source_hash = hashlib.sha256(raw_bytes).hexdigest()
with SOURCE.open() as handle:
    reader = csv.reader(handle)
    headers = [value.strip() for value in next(reader) if value.strip()]
    records = [[value.strip() for value in row if value.strip()]
               for row in reader if any(value.strip() for value in row)]
assert all(len(row) == len(headers) for row in records)
data = {key: np.array([int(row[i]) if 'stamp' in key else float(row[i]) for row in records])
        for i, key in enumerate(headers)}
assert all(np.all(np.isfinite(values)) for values in data.values())
time = (data['stamp'] - data['stamp'][0]) / 1e9
intervals = np.diff(data['stamp']) / 1e9
assert np.all(intervals > 0)
settings = json.loads(SOURCE.with_suffix('.settings.json').read_text())
kf = kalman_filter(settings['initial_P'], settings['Q'], settings['R'], settings['initial_state'], settings['nominal_dt_s'])
replay_error = np.zeros(6)
covariance_eigenvalue_minimum = float('inf')
for i in range(len(time)):
    if i:
        kf.predict(dt=intervals[i - 1])
    kf.update([data[key][i] for key in ['odom_vx', 'odom_w', 'imu_ax', 'imu_ay']])
    logged = np.array([data[key][i] for key in ['kf_x', 'kf_y', 'kf_th', 'kf_w', 'kf_vx', 'kf_ax']])
    replay_error = np.maximum(replay_error, np.abs(kf.x - logged))
    covariance_eigenvalue_minimum = min(covariance_eigenvalue_minimum, float(np.linalg.eigvalsh(kf.P).min()))
assert np.max(replay_error) < 1e-12

position_difference = np.hypot(data['kf_x'] - data['odom_x'], data['kf_y'] - data['odom_y'])
heading_difference = np.arctan2(np.sin(data['kf_th'] - data['odom_th']), np.cos(data['kf_th'] - data['odom_th']))
steady = (time >= 15) & (time <= 26)
forward_acceleration = np.gradient(data['odom_vx'], time)
pairs = [('odom_vx', 'kf_vx'), ('odom_w', 'kf_w'), ('imu_ax', 'kf_ax'), ('imu_ay', 'kf_ay')]
metrics = {
    'source': str(SOURCE.relative_to(ROOT)),
    'source_sha256': source_hash,
    'settings_sha256': hashlib.sha256(SOURCE.with_suffix('.settings.json').read_bytes()).hexdigest(),
    'filter_code_sha256': hashlib.sha256((ROOT / 'kalman_filter.py').read_bytes()).hexdigest(),
    'samples': len(time), 'duration_sim_s': float(time[-1]),
    'finite_all_columns': True,
    'interval_s_min_median_max': [float(np.min(intervals)), float(np.median(intervals)), float(np.max(intervals))],
    'effective_rate_hz': float((len(time) - 1) / time[-1]),
    'sensor_stamp_separation_ms_max': float(np.max(np.abs(data['stamp'] - data['imu_stamp'])) / 1e6),
    'replay_max_absolute_error_state_order': replay_error.tolist(),
    'replayed_covariance_min_eigenvalue': covariance_eigenvalue_minimum,
    'replayed_final_covariance_diagonal': np.diag(kf.P).tolist(),
    'position_difference_rms_cm': float(100 * np.sqrt(np.mean(position_difference ** 2))),
    'position_difference_max_cm': float(100 * position_difference.max()),
    'position_difference_final_cm': float(100 * position_difference[-1]),
    'heading_difference_rms_deg': float(np.rad2deg(np.sqrt(np.mean(heading_difference ** 2)))),
    'heading_difference_max_abs_deg': float(np.rad2deg(np.max(np.abs(heading_difference)))),
    'steady_window_s': [15, 26],
    'steady_samples': int(steady.sum()),
    'steady_comparisons': {},
    'steady_imu_ax_mean_m_s2': float(data['imu_ax'][steady].mean()),
    'steady_odom_dv_dt_mean_m_s2': float(forward_acceleration[steady].mean()),
    'comparison_limit': 'Odometry is not independent ground truth. Estimate differences are not absolute localization errors.',
}
for raw, estimated in pairs:
    metrics['steady_comparisons'][raw] = {
        'estimated_column': estimated,
        'raw_std': float(data[raw][steady].std()),
        'estimated_std': float(data[estimated][steady].std()),
        'rms_difference': float(np.sqrt(np.mean((data[estimated][steady] - data[raw][steady]) ** 2))),
    }
(OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9})
fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
ax = axes[0, 0]
ax.plot(data['odom_x'], data['odom_y'], color='tab:blue', label='Raw odometry', lw=1.7)
ax.plot(data['kf_x'], data['kf_y'], color='tab:orange', label='EKF estimate', lw=1.2, ls='--')
ax.scatter([data['odom_x'][0]], [data['odom_y'][0]], color='black', marker='o', s=20, label='Start', zorder=4)
ax.scatter([data['odom_x'][-1]], [data['odom_y'][-1]], color='black', marker='s', s=20, label='End', zorder=4)
ax.set(title='Recorded path', xlabel='x (m)', ylabel='y (m)', aspect='equal')
ax.legend(fontsize=8)
ax = axes[0, 1]
ax.plot(time, position_difference * 100, label='Distance between EKF and odometry', color='tab:purple')
ax.set(title='Position disagreement (odometry is not ground truth)', xlabel='Elapsed simulation time (s)', ylabel='Distance (cm)')
ax.legend(fontsize=8)
for ax, raw, estimated, title, unit in [
    (axes[1, 0], 'odom_vx', 'kf_vx', 'Forward velocity', 'm/s'),
    (axes[1, 1], 'odom_w', 'kf_w', 'Angular velocity', 'rad/s'),
    (axes[2, 0], 'imu_ax', 'kf_ax', 'Forward acceleration', 'm/s²'),
    (axes[2, 1], 'imu_ay', 'kf_ay', 'Lateral acceleration', 'm/s²'),
]:
    ax.plot(time, data[raw], color='tab:blue', lw=0.9, alpha=0.75, label='Odometry' if raw.startswith('odom') else 'IMU')
    ax.plot(time, data[estimated], color='tab:orange', lw=1.0, label='EKF')
    if raw == 'imu_ax':
        ax.plot(time, forward_acceleration, color='tab:green', lw=0.8, alpha=0.8, label='Odometry dv/dt (finite difference)')
    if raw == 'imu_ay':
        ax.plot(time, data['odom_vx'] * data['odom_w'], color='tab:green', lw=0.8, ls='--', label='Odometry v × omega')
    ax.set(title=title, xlabel='Elapsed simulation time (s)', ylabel=f'{title} ({unit})')
    ax.legend(fontsize=8)
for ax in axes.flat:
    ax.grid(alpha=0.3)
fig.suptitle('Lab 3 spiral baseline | Q = 0.5 I6, R = 0.5 I4 | All 795 samples retained', fontsize=13)
fig.savefig(OUT / 'baseline_overview.png', dpi=180)
fig.savefig(OUT / 'baseline_overview.pdf')
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), constrained_layout=True)
for ax, (raw, estimated), title, unit in zip(axes.flat, pairs,
        ['Forward velocity', 'Angular velocity', 'Forward acceleration', 'Lateral acceleration'],
        ['m/s', 'rad/s', 'm/s²', 'm/s²']):
    ax.plot(time, data[raw], color='tab:blue', label='Odometry' if raw.startswith('odom') else 'IMU')
    ax.plot(time, data[estimated], color='tab:orange', label='EKF', ls='--')
    ax.set(title=title, xlabel='Elapsed simulation time (s)', ylabel=unit, xlim=(15, 18))
    if raw == 'odom_vx':
        ax.set_ylim(0.9, 1.1)
    elif raw == 'odom_w':
        ax.set_ylim(0.9, 1.1)
    ax.legend()
    ax.grid(alpha=0.3)
fig.suptitle('Steady-motion detail | IMU fluctuations also influence estimated velocities', fontsize=13)
fig.savefig(OUT / 'steady_motion_detail.png', dpi=180)
plt.close(fig)

assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash, 'Source CSV changed during review'
print(json.dumps(metrics, indent=2))
print('Saved plots and metrics in', OUT)
