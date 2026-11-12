"""Plot completed Lab 3 recordings by CSV column name; no ROS required."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

REQUIRED = {
    'stamp', 'imu_stamp', 'imu_ax', 'imu_ay', 'kf_ax', 'kf_ay',
    'odom_vx', 'odom_w', 'kf_vx', 'kf_w',
    'odom_x', 'odom_y', 'odom_th', 'kf_x', 'kf_y', 'kf_th',
}


def load_recording(path):
    """Retain every sample and preserve integer ns until subtracting origin."""
    path = Path(path)
    source = path / 'robot_pose.csv' if path.is_dir() else path
    payload = source.read_bytes()
    lines = payload.decode().splitlines()
    reader = csv.reader(lines, skipinitialspace=True)
    try:
        headers = [value.strip() for value in next(reader)]
    except StopIteration:
        raise ValueError(f'{source}: empty CSV') from None
    while headers and not headers[-1]:
        headers.pop()
    if len(headers) != len(set(headers)) or not REQUIRED.issubset(headers):
        raise ValueError(f'{source}: missing required columns or duplicate headers')
    columns = {key: [] for key in headers}
    for line_number, row in enumerate(reader, start=2):
        if not any(value.strip() for value in row):
            continue
        row = [value.strip() for value in row]
        while row and not row[-1]:
            row.pop()
        if len(row) != len(headers):
            raise ValueError(f'{source}:{line_number}: wrong number of fields')
        for key, value in zip(headers, row):
            columns[key].append(int(value) if 'stamp' in key else float(value))
    data = {key: np.asarray(values) for key, values in columns.items()}
    if len(data['stamp']) < 2:
        raise ValueError(f'{source}: need at least two recorded samples')
    if not all(np.all(np.isfinite(values)) for values in data.values()):
        raise ValueError(f'{source}: non-finite CSV values')
    if np.any(np.diff(data['stamp']) <= 0):
        raise ValueError(f'{source}: timestamps must increase strictly')
    elapsed = (data['stamp'] - data['stamp'][0]) / 1e9
    return source.resolve(), data, elapsed, hashlib.sha256(payload).hexdigest()


def plot_run(path):
    """Write PNG/PDF comparisons and numerical summary beside a recording."""
    # Select a file backend before importing pyplot, also on headless PCs.
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mte544_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    source, data, elapsed, checksum = load_recording(path)
    run_dir = source.parent
    out = run_dir / 'plots'
    out.mkdir(exist_ok=True)
    intervals = np.diff(data['stamp']) / 1e9
    position_difference = np.hypot(data['kf_x'] - data['odom_x'], data['kf_y'] - data['odom_y'])
    heading_difference = np.arctan2(
        np.sin(data['kf_th'] - data['odom_th']), np.cos(data['kf_th'] - data['odom_th']),
    )
    settings_file = source.with_suffix('.settings.json')
    settings = json.loads(settings_file.read_text()) if settings_file.exists() else None
    summary = {
        'source_csv': source.name,
        'source_sha256': checksum,
        'samples': len(elapsed), 'elapsed_sim_s': float(elapsed[-1]),
        'effective_sample_rate_hz': float((len(elapsed) - 1) / elapsed[-1]),
        'interval_s_min_median_max': [float(intervals.min()), float(np.median(intervals)), float(intervals.max())],
        'max_sensor_stamp_separation_ms': float(np.max(np.abs(data['stamp'] - data['imu_stamp'])) / 1e6),
        'position_difference_rms_cm': float(100 * np.sqrt(np.mean(position_difference ** 2))),
        'position_difference_max_cm': float(100 * position_difference.max()),
        'heading_difference_rms_deg': float(np.rad2deg(np.sqrt(np.mean(heading_difference ** 2)))),
        'Q': settings['Q'] if settings else None,
        'R': settings['R'] if settings else None,
        'comparison_limit': 'Differences from odometry are not errors against independent ground truth.',
        'preprocessing': 'All samples retained; no smoothing. Headings unwrapped only for continuous time plots.',
    }
    pairs = [
        ('odom_vx', 'kf_vx', 'Forward velocity', 'm/s', 'Odometry'),
        ('odom_w', 'kf_w', 'Angular velocity', 'rad/s', 'Odometry'),
        ('imu_ax', 'kf_ax', 'Forward acceleration', 'm/s²', 'IMU'),
        ('imu_ay', 'kf_ay', 'Lateral acceleration', 'm/s²', 'IMU'),
    ]
    summary['rms_measurement_estimate_difference'] = {
        raw: float(np.sqrt(np.mean((data[estimated] - data[raw]) ** 2)))
        for raw, estimated, *_ in pairs
    }
    suffix = ''
    if settings:
        q, r = np.asarray(settings['Q']), np.asarray(settings['R'])
        if np.allclose(q, np.eye(6) * q[0, 0]) and np.allclose(r, np.eye(4) * r[0, 0]):
            suffix = f' | Q = {q[0, 0]:g} I6, R = {r[0, 0]:g} I4'
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (raw, estimated, title, unit, sensor) in zip(axes.flat, pairs):
        ax.plot(elapsed, data[raw], color='tab:blue', lw=1, alpha=0.8, label=sensor)
        ax.plot(elapsed, data[estimated], color='tab:orange', ls='--', lw=1.1, label='EKF estimate')
        ax.set(title=title, xlabel='Elapsed sensor time (s)', ylabel=f'{title} ({unit})')
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f'{run_dir.name}: measurements and estimates{suffix}', fontsize=12)
    for extension in ['png', 'pdf']:
        fig.savefig(out / f'measurements_vs_estimates.{extension}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(data['odom_x'], data['odom_y'], color='tab:blue', label='Odometry')
    ax.plot(data['kf_x'], data['kf_y'], color='tab:orange', ls='--', label='EKF estimate')
    ax.scatter([data['kf_x'][0]], [data['kf_y'][0]], color='black', marker='o', s=22, label='EKF start', zorder=4)
    ax.scatter([data['kf_x'][-1]], [data['kf_y'][-1]], color='black', marker='s', s=22, label='EKF end', zorder=4)
    ax.set(title='Recorded path', xlabel='x (m)', ylabel='y (m)', aspect='equal')
    for ax, component, title, unit in [
        (axes[0, 1], 'x', 'x position', 'm'),
        (axes[1, 0], 'y', 'y position', 'm'),
        (axes[1, 1], 'th', 'Heading (unwrapped)', 'rad'),
    ]:
        raw, estimate = data[f'odom_{component}'], data[f'kf_{component}']
        if component == 'th':
            raw, estimate = np.unwrap(raw), np.unwrap(estimate)
        ax.plot(elapsed, raw, color='tab:blue', label='Odometry')
        ax.plot(elapsed, estimate, color='tab:orange', ls='--', label='EKF estimate')
        ax.set(title=title, xlabel='Elapsed sensor time (s)', ylabel=unit)
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f'{run_dir.name}: pose comparison{suffix}', fontsize=12)
    for extension in ['png', 'pdf']:
        fig.savefig(out / f'pose_comparison.{extension}', dpi=180)
    plt.close(fig)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recordings', type=Path, nargs='+', help='Run folders or robot_pose.csv files')
    args = parser.parse_args()
    for recording in args.recordings:
        try:
            output = plot_run(recording)
        except (OSError, ValueError) as error:
            parser.exit(1, f'Could not plot {recording}: {error}\n')
        print(f'Saved plots: {output}')


if __name__ == '__main__':
    main()
