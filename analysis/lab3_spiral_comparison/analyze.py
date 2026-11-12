"""Compare recorded spiral trials and isolate covariance effects by replay.

Recorded-trial results and counterfactual offline replays are kept separate.
This script never changes recordings or runtime code and does not run ROS.
"""

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
from plot_lab3 import load_recording

OUT = Path(__file__).resolve().parent
CONFIGS = {
    'Baseline': (0.5, 0.5),
    'Lower Q': (0.1, 0.5),
    'Higher Q': (1.0, 0.5),
    'Lower R': (0.5, 0.1),
    'Higher R': (0.5, 1.0),
}
PREFIXES = {
    'Baseline': 'lab3_spiral_baseline',
    'Lower Q': 'lab3_spiral_q_low_*',
    'Higher Q': 'lab3_spiral_q_high_*',
    'Lower R': 'lab3_spiral_r_low_*',
    'Higher R': 'lab3_spiral_r_high_*',
}
RUNTIME = ['decisions.py', 'localization.py', 'kalman_filter.py', 'controller.py', 'planner.py', 'pid.py', 'utilities.py']
filter_hash = hashlib.sha256((ROOT / 'kalman_filter.py').read_bytes()).hexdigest()
runs = {}
for label, pattern in PREFIXES.items():
    matches = sorted(p for p in (ROOT / 'test_results').glob(pattern) if (p / 'robot_pose.csv').exists())
    assert len(matches) == 1, f'Choose an explicit run for {label}: {matches}'
    folder = matches[0]
    _, data, time, source_hash = load_recording(folder)
    settings = json.loads((folder / 'robot_pose.settings.json').read_text())
    q, r = CONFIGS[label]
    np.testing.assert_array_equal(settings['Q'], np.eye(6) * q)
    np.testing.assert_array_equal(settings['R'], np.eye(4) * r)
    np.testing.assert_array_equal(settings['initial_P'], np.eye(6) * 0.1)
    trial_file = folder / 'trial.json'
    trial = json.loads(trial_file.read_text()) if trial_file.exists() else None
    if trial:
        assert trial['status'] == 'duration_limit_reached'
        assert trial['stop_command_succeeded'] and trial['process_returncode'] == 0
        assert trial['use_sim_time'] and trial['motion'] == 'spiral'
        for name, expected in trial['source_sha256'].items():
            assert hashlib.sha256((folder / 'code' / name).read_bytes()).hexdigest() == expected
        assert trial['source_sha256']['kalman_filter.py'] == filter_hash
    else:
        baseline_audit = json.loads((ROOT / 'analysis/lab3_spiral_baseline/metrics.json').read_text())
        assert baseline_audit['filter_code_sha256'] == filter_hash
        assert baseline_audit['source_sha256'] == source_hash
    runs[label] = {'folder': folder, 'data': data, 'time': time, 'settings': settings,
                   'trial': trial, 'source_sha256': source_hash}
new_trials = [run['trial'] for run in runs.values() if run['trial']]
assert all(all(trial['source_sha256'][name] == new_trials[0]['source_sha256'][name]
               for name in RUNTIME) for trial in new_trials)
end = float(min(run['time'][-1] for run in runs.values()))


def replay(run, q, r, full=False):
    data, time, settings = run['data'], run['time'], run['settings']
    count = len(time) if full else int(np.sum(time <= end))
    kf = kalman_filter(settings['initial_P'], q * np.eye(6), r * np.eye(4), settings['initial_state'], settings['nominal_dt_s'])
    states = []
    minimum_eigenvalue = float('inf')
    for i in range(count):
        if i:
            kf.predict(dt=(int(data['stamp'][i]) - int(data['stamp'][i - 1])) / 1e9)
        kf.update([data[key][i] for key in ['odom_vx', 'odom_w', 'imu_ax', 'imu_ay']])
        states.append(kf.x.copy())
        minimum_eigenvalue = min(minimum_eigenvalue, float(np.linalg.eigvalsh(kf.P).min()))
    assert minimum_eigenvalue > 0
    return np.asarray(states), minimum_eigenvalue


def comparison_metrics(run, states):
    count = len(states)
    data, time = run['data'], run['time'][:count]
    steady = (time >= 15) & (time <= end)
    position = np.hypot(states[:, 0] - data['odom_x'][:count], states[:, 1] - data['odom_y'][:count])
    heading = np.arctan2(np.sin(states[:, 2] - data['odom_th'][:count]), np.cos(states[:, 2] - data['odom_th'][:count]))
    return {
        'common_samples': count,
        'position_rms_difference_cm': float(np.sqrt(np.mean(position ** 2)) * 100),
        'position_max_difference_cm': float(position.max() * 100),
        'heading_rms_difference_deg': float(np.rad2deg(np.sqrt(np.mean(heading ** 2)))),
        'velocity_rms_difference_m_s': float(np.sqrt(np.mean((states[:, 4] - data['odom_vx'][:count]) ** 2))),
        'angular_velocity_rms_difference_rad_s': float(np.sqrt(np.mean((states[:, 3] - data['odom_w'][:count]) ** 2))),
        'steady_v_std_m_s': float(states[steady, 4].std()),
        'steady_w_std_rad_s': float(states[steady, 3].std()),
        'steady_odom_v_std_m_s': float(data['odom_vx'][:count][steady].std()),
        'steady_odom_w_std_rad_s': float(data['odom_w'][:count][steady].std()),
        'steady_raw_ax_mean_m_s2': float(data['imu_ax'][:count][steady].mean()),
        'steady_estimated_ax_mean_m_s2': float(states[steady, 5].mean()),
        'steady_odom_dv_dt_mean_m_s2': float(np.gradient(data['odom_vx'][:count], time)[steady].mean()),
        'steady_raw_ay_std_m_s2': float(data['imu_ay'][:count][steady].std()),
        'steady_estimated_ay_std_m_s2': float((states[steady, 4] * states[steady, 3]).std()),
    }


recorded = []
for label, run in runs.items():
    q, r = CONFIGS[label]
    states, minimum_eigenvalue = replay(run, q, r, full=True)
    logged = np.column_stack([run['data'][key] for key in ['kf_x', 'kf_y', 'kf_th', 'kf_w', 'kf_vx', 'kf_ax']])
    replay_difference = float(np.max(np.abs(states - logged)))
    assert replay_difference < 1e-12
    mask = run['time'] <= end
    row = {
        'setting': label, 'q': q, 'r': r, 'run': run['folder'].name,
        'source_sha256': run['source_sha256'],
        'full_samples': len(run['time']), 'full_duration_sim_s': float(run['time'][-1]),
        'interval_s_min': float(np.min(np.diff(run['data']['stamp'])) / 1e9),
        'interval_s_max': float(np.max(np.diff(run['data']['stamp'])) / 1e9),
        'max_sensor_stamp_separation_ms': float(np.max(np.abs(run['data']['stamp'] - run['data']['imu_stamp'])) / 1e6),
        'all_logged_states_replay_max_abs_difference': replay_difference,
        'replayed_covariance_minimum_eigenvalue': minimum_eigenvalue,
        **comparison_metrics(run, logged[mask]),
    }
    recorded.append(row)

counterfactual = []
for setting, (q, r) in CONFIGS.items():
    for input_label, run in runs.items():
        states, _ = replay(run, q, r)
        counterfactual.append({'setting': setting, 'input_recording': input_label,
                               **comparison_metrics(run, states)})
replay_means = {}
for label in CONFIGS:
    rows = [row for row in counterfactual if row['setting'] == label]
    replay_means[label] = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]
                           if key not in ['setting', 'input_recording']}
for filename, rows in [('recorded_comparison.csv', recorded), ('offline_replay_comparison.csv', counterfactual)]:
    with (OUT / filename).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
summary = {
    'common_elapsed_sensor_time_s': [0, end], 'common_samples_per_run': recorded[0]['common_samples'],
    'steady_window_s': [15, end],
    'recorded_trials': recorded,
    'offline_replay_mean_metrics': replay_means,
    'offline_replay_method': 'Apply all five settings to every recorded input sequence, holding initial P, initial state and timing fixed per sequence. These are counterfactual analyses, not additional experiments.',
    'limitations': [
        'Odometry is a comparison source, not independent ground truth.',
        'Only one actual recording exists per covariance setting; raw IMU realizations differ.',
        'All runs retain a forward-acceleration discrepancy at steady odometry speed.',
        'Point control and physical-robot performance have not been tested by this analysis.',
    ],
    'provisional_point_test': {'q': 0.5, 'r': 1.0, 'reason': 'Moderate smoothing compromise; requires confirmation in point control.'},
    'source_filter_sha256': filter_hash,
}
(OUT / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')

colors = dict(zip(CONFIGS, ['#333333', '#0072B2', '#D55E00', '#009E73', '#CC79A7']))
fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
for label, run in runs.items():
    d, t = run['data'], run['time']
    mask = t <= end
    t = t[mask]
    color = colors[label]
    dp = np.hypot(d['kf_x'][mask] - d['odom_x'][mask], d['kf_y'][mask] - d['odom_y'][mask])
    dh = np.rad2deg(np.arctan2(np.sin(d['kf_th'][mask] - d['odom_th'][mask]), np.cos(d['kf_th'][mask] - d['odom_th'][mask])))
    values = [dp * 100, dh, d['kf_vx'][mask] - d['odom_vx'][mask],
              d['kf_w'][mask] - d['odom_w'][mask], d['kf_ax'][mask],
              d['imu_ay'][mask] - d['odom_vx'][mask] * d['odom_w'][mask]]
    for ax, y in zip(axes.flat, values):
        ax.plot(t, y, color=color, lw=0.85, alpha=0.8, label=label)
for ax, title, unit in zip(axes.flat,
        ['Position disagreement', 'Heading disagreement', 'Forward-velocity difference (steady motion)',
         'Angular-velocity difference (steady motion)', 'Estimated forward acceleration (steady motion)',
         'Lateral acceleration residual: IMU − odometry vω'],
        ['cm', 'deg', 'm/s', 'rad/s', 'm/s²', 'm/s²']):
    ax.set(title=title, xlabel='Elapsed sensor time (s)', ylabel=unit)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
for ax in [axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]:
    ax.set_xlim(15, end)
fig.suptitle(f'Recorded spiral trials | Common window 0–{end:.3f} s\nDifferences from odometry are not ground-truth errors', fontsize=12)
fig.savefig(OUT / 'recorded_run_comparison.png', dpi=180)
fig.savefig(OUT / 'recorded_run_comparison.pdf')
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
for ax, key, title, unit in zip(axes,
        ['position_rms_difference_cm', 'steady_v_std_m_s', 'steady_w_std_rad_s'],
        ['Position agreement with odometry', 'Forward-velocity fluctuations', 'Angular-velocity fluctuations'],
        ['RMS difference (cm)', 'Standard deviation (m/s)', 'Standard deviation (rad/s)']):
    means = [replay_means[label][key] for label in CONFIGS]
    ax.bar(range(5), means, color=list(colors.values()), alpha=0.7, label='Mean over five input recordings')
    for i, label in enumerate(CONFIGS):
        points = [row[key] for row in counterfactual if row['setting'] == label]
        ax.scatter(np.full(len(points), i), points, color='black', s=12, alpha=0.65)
    ax.set_xticks(range(5), [label.replace(' ', '\n') for label in CONFIGS])
    ax.set(title=title, ylabel=unit)
    ax.grid(axis='y', alpha=0.3)
fig.suptitle('Offline replay: each setting applied to every same input recording\nDots = individual recordings; bars = means. These are not additional Gazebo trials.', fontsize=11)
fig.savefig(OUT / 'offline_replay_comparison.png', dpi=180)
fig.savefig(OUT / 'offline_replay_comparison.pdf')
plt.close(fig)

lines = [
    '# Lab 3 spiral covariance comparison', '',
    f'Compared all five recordings over 0–{end:.3f} seconds ({recorded[0]["common_samples"]} samples each).',
    'All CSVs are finite with strictly increasing timestamps; every recorded EKF state reproduces exactly.',
    'All four new trials completed their duration limit, exited with code 0 and confirmed a zero-velocity publication.',
    'The saved covariance matrices, initial P and code snapshots were verified.', '',
    '| Setting | q | r | Position RMS difference (cm) | Maximum difference (cm) | Heading RMS difference (deg) |',
    '|---|---:|---:|---:|---:|---:|',
]
for row in recorded:
    lines.append(f'| {row["setting"]} | {row["q"]:g} | {row["r"]:g} | {row["position_rms_difference_cm"]:.2f} | {row["position_max_difference_cm"]:.2f} | {row["heading_rms_difference_deg"]:.2f} |')
lines += [
    '', '## Interpretation', '',
    'Higher R has the lowest recorded pose disagreement, but separate recordings contain different IMU fluctuations. It is not proof of best absolute accuracy.',
    'The controlled offline replay applies every setting to every sensor recording. Lower Q reduces steady velocity fluctuations most, while increasing pose disagreement with odometry. Higher R gives moderate smoothing with a smaller agreement penalty. Lower R follows measurements more strongly and yields more velocity jitter on the same inputs.',
    'In the offline replay means, higher R reduces forward/angular velocity standard deviation by approximately 8%/9% relative to the baseline. Lower Q reduces them by approximately 20%/22%.',
    'The common steady-motion IMU forward-acceleration means remain about −0.054 to −0.058 m/s² although odometry forward speed is nearly constant. Covariance changes do not remove this sensor/model discrepancy.',
    '', '## Next experiment', '',
    'Test point control using Q = 0.5 I6 and R = 1.0 I4 as a provisional smoothing compromise. Do not label it a final optimum. Reset Gazebo first:', '',
    '```bash',
    'python3 run_lab3_trial.py --sim --motion point --name lab3_point_r_high --q 0.5 --r 1.0 --duration 120',
    '```', '',
    'If time permits, run a point baseline (q = 0.5, r = 0.5) under the same reset conditions. Compare actual odometry distance to the goal as well as EKF distance at termination. Investigate the IMU forward-acceleration discrepancy before final physical-robot tuning.',
    '', 'The recording CSVs and runtime code were not modified. See comparison.json and both CSV tables for full metrics and provenance.',
]
(OUT / 'REVIEW.md').write_text('\n'.join(lines) + '\n')
for run in runs.values():
    assert hashlib.sha256((run['folder'] / 'robot_pose.csv').read_bytes()).hexdigest() == run['source_sha256']
print('\n'.join(lines[:16]))
print('Saved analysis:', OUT)
print('Offline replay mean metrics:', json.dumps({label: {k: row[k] for k in ['position_rms_difference_cm', 'steady_v_std_m_s', 'steady_w_std_rad_s']} for label, row in replay_means.items()}, indent=2))
