"""Plot the Lab 1 sensor logs by motion; see PLOTTING.md for usage."""

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


ROOT = Path(__file__).resolve().parent
MOTIONS = ('circle', 'spiral', 'line')
SENSORS = ('odom', 'imu', 'laser')
COLORS = ('#0072B2', '#D55E00', '#009E73')


@dataclass
class SensorLog:
    path: Path
    stamps: np.ndarray
    columns: dict

    def elapsed(self, origin):
        # Subtract integer nanoseconds before converting to avoid epoch rounding.
        return (self.stamps - origin).astype(float) / 1e9


def read_log(path):
    """Read our numeric CSV format, including its trailing empty column."""
    with path.open(newline='') as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        headers = [value.strip() for value in next(reader, [])]
        while headers and not headers[-1]:
            headers.pop()
        if not headers or len(set(headers)) != len(headers) or 'stamp' not in headers:
            raise ValueError(f'{path}: expected unique column names and a stamp column')
        stamp_index = headers.index('stamp')
        values, stamps = [], []
        for line_number, row in enumerate(reader, start=2):
            while row and not row[-1].strip():
                row.pop()
            if not row:
                continue
            if len(row) != len(headers):
                raise ValueError(f'{path}:{line_number}: row width does not match header')
            try:
                stamps.append(int(row[stamp_index]))
                values.append([float(value) for i, value in enumerate(row)
                               if i != stamp_index])
            except ValueError as error:
                raise ValueError(f'{path}:{line_number}: invalid numeric value') from error
    if not values:
        raise ValueError(f'{path}: no sensor samples')
    stamps = np.asarray(stamps, dtype=np.int64)
    if np.any(np.diff(stamps) < 0):
        raise ValueError(f'{path}: timestamps go backward; select a single uninterrupted run')
    matrix = np.asarray(values, dtype=float)
    names = [name for name in headers if name != 'stamp']
    return SensorLog(path, stamps, dict(zip(names, matrix.T)))


def require_columns(log, names):
    for name in names:
        if name not in log.columns:
            raise ValueError(f'{log.path}: missing {name} column')
        if not np.all(np.isfinite(log.columns[name])):
            raise ValueError(f'{log.path}: nonfinite values in {name}')


def scan_points(log, index):
    """Convert one scan to local Cartesian points; first beam points along +x."""
    if not 0 <= index < len(log.stamps):
        raise ValueError(f'{log.path}: scan index must be 0..{len(log.stamps) - 1}')
    beams = sorted((int(name[6:]), name) for name in log.columns
                   if re.fullmatch(r'range_\d+', name))
    if not beams or 'angle_increment' not in log.columns:
        raise ValueError(f'{log.path}: expected range_<index> and angle_increment columns')
    increment = log.columns['angle_increment'][index]
    if not np.isfinite(increment) or increment <= 0:
        raise ValueError(f'{log.path}: invalid angle_increment at scan {index}')
    ranges = np.asarray([log.columns[name][index] for _, name in beams])
    # Keep original beam indices: removing a range must not rotate later beams.
    angles = np.asarray([beam for beam, _ in beams]) * increment
    valid = np.isfinite(ranges) & (ranges > 0)
    if not np.any(valid):
        raise ValueError(f'{log.path}: scan {index} has no finite positive ranges')
    x = ranges[valid] * np.cos(angles[valid])
    y = ranges[valid] * np.sin(angles[valid])
    return x, y, int(np.sum(~valid))


def label_axes(ax, title, xlabel, ylabel):
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)


def draw_trajectory(ax, log, motion):
    x, y = log.columns['x'], log.columns['y']
    ax.plot(x, y, color=COLORS[0], marker='o', markersize=3,
            markevery=max(1, len(x) // 18), linewidth=1.4, label='Odometry path')
    ax.scatter(x[0], y[0], color=COLORS[2], marker='^', s=65, label='Start', zorder=4)
    ax.scatter(x[-1], y[-1], color=COLORS[1], marker='X', s=65, label='End', zorder=4)
    label_axes(ax, f'{motion.title()} — odometry trajectory', 'x (m)', 'y (m)')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=8)


def draw_odometry(axes, log, motion, origin):
    time = log.elapsed(origin)
    for name, color, marker in [('x', COLORS[0], 'o'), ('y', COLORS[1], 's')]:
        axes[0].plot(time, log.columns[name], color=color, marker=marker,
                     markersize=3, markevery=max(1, len(time) // 18), label=name)
    label_axes(axes[0], f'{motion.title()} — position versus time', '', 'Position (m)')
    # Yaw was converted from quaternions when logging. Unwrap ±pi jumps so
    # completed turns are readable, without changing the recorded CSV.
    axes[1].plot(time, np.unwrap(log.columns['th']), color=COLORS[2], marker='^',
                 markersize=3, markevery=max(1, len(time) // 18), label='Yaw (unwrapped)')
    label_axes(axes[1], 'Heading versus time', 'Elapsed time (s)', 'Yaw (rad)')
    for ax in axes:
        ax.legend(fontsize=8)


def draw_imu(axes, log, motion, origin):
    time = log.elapsed(origin)
    for name, label, color, marker in [
            ('acc_x', 'a_x', COLORS[0], 'o'), ('acc_y', 'a_y', COLORS[1], 's')]:
        axes[0].plot(time, log.columns[name], color=color, marker=marker,
                     markersize=2.5, markevery=max(1, len(time) // 18),
                     linewidth=0.8, label=label)
    label_axes(axes[0], f'{motion.title()} — IMU acceleration', '', 'Acceleration (m/s²)')
    axes[1].plot(time, log.columns['angular_z'], color=COLORS[2], marker='^',
                 markersize=2.5, markevery=max(1, len(time) // 18),
                 linewidth=0.8, label='w_z')
    label_axes(axes[1], 'IMU angular velocity', 'Elapsed time (s)', 'Angular velocity (rad/s)')
    for ax in axes:
        ax.legend(fontsize=8)


def draw_scan(ax, log, motion, origin, index):
    x, y, removed = scan_points(log, index)
    time = log.elapsed(origin)[index]
    ax.scatter(x, y, color=COLORS[0], s=9, marker='.', label='Finite lidar returns')
    ax.scatter(0, 0, color=COLORS[1], s=55, marker='^', label='Lidar origin', zorder=4)
    label_axes(ax, f'{motion.title()} — lidar scan {index} at {time:.2f} s',
               'Local x (m)', 'Local y (m)')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=8)
    ax.text(0.02, 0.02, f'{len(x)} valid beams; {removed} discarded',
            transform=ax.transAxes, fontsize=8,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))


def make_figures(logs, motion, origin, scan_index):
    figures = {}
    if 'odom' in logs:
        require_columns(logs['odom'], ('x', 'y', 'th'))
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        draw_trajectory(ax, logs['odom'], motion)
        figures['odom_trajectory'] = fig
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6), constrained_layout=True)
        draw_odometry(axes, logs['odom'], motion, origin)
        figures['odom_timeseries'] = fig
    if 'imu' in logs:
        require_columns(logs['imu'], ('acc_x', 'acc_y', 'angular_z'))
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6), constrained_layout=True)
        draw_imu(axes, logs['imu'], motion, origin)
        figures['imu_timeseries'] = fig
    if 'laser' in logs:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        draw_scan(ax, logs['laser'], motion, origin, scan_index)
        figures['lidar_scan'] = fig
    if all(sensor in logs for sensor in SENSORS):
        fig = plt.figure(figsize=(13, 9), constrained_layout=True)
        fig.suptitle(f'Lab 1 — {motion.title()} sensor data', fontsize=16)
        # One grid avoids overlapping nested layouts on Matplotlib 3.5 (Humble).
        grid = fig.add_gridspec(4, 2)
        draw_trajectory(fig.add_subplot(grid[:2, 0]), logs['odom'], motion)
        for row, column, sensor, draw in [(0, 1, 'odom', draw_odometry),
                                          (2, 0, 'imu', draw_imu)]:
            upper = fig.add_subplot(grid[row, column])
            lower = fig.add_subplot(grid[row + 1, column], sharex=upper)
            draw((upper, lower), logs[sensor], motion, origin)
        draw_scan(fig.add_subplot(grid[2:, 1]), logs['laser'], motion, origin, scan_index)
        figures['overview'] = fig
    return figures


def selected_paths(args):
    if args.files:
        paths = []
        for filename in args.files:
            path = Path(filename)
            if not path.is_file() and not path.is_absolute():
                path = args.data_dir / path
            paths.append(path)
        return paths
    motions = MOTIONS if args.motion == 'all' else (args.motion,)
    return [args.data_dir / f'{sensor}_content_{motion}.csv'
            for motion in motions for sensor in SENSORS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--motion', choices=('all',) + MOTIONS, default='all')
    selection.add_argument('--files', nargs='+', help='Plot selected sensor CSVs (original CLI)')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'plots')
    parser.add_argument('--scan-index', type=int, default=0, help='Zero-based lidar row; default: first scan')
    parser.add_argument('--show', action='store_true', help='Also open interactive plot windows')
    args = parser.parse_args()
    grouped = {}
    try:
        for path in selected_paths(args):
            match = re.fullmatch(r'(odom|imu|laser)_content_(circle|spiral|line)\.csv', path.name)
            if not match:
                raise ValueError(f'{path}: expected <sensor>_content_<motion>.csv')
            sensor, motion = match.groups()
            if sensor in grouped.setdefault(motion, {}):
                raise ValueError(f'{motion}: multiple {sensor} files selected')
            grouped[motion][sensor] = read_log(path)
        # Validate every selected file before exporting any figures.
        for logs in grouped.values():
            if 'odom' in logs:
                require_columns(logs['odom'], ('x', 'y', 'th'))
            if 'imu' in logs:
                require_columns(logs['imu'], ('acc_x', 'acc_y', 'angular_z'))
            if 'laser' in logs:
                scan_points(logs['laser'], args.scan_index)
    except (OSError, ValueError, OverflowError) as error:
        parser.error(str(error))

    summary = []
    for motion, logs in grouped.items():
        # Use one time origin per motion so separate sensors remain aligned.
        origin = min(int(log.stamps[0]) for log in logs.values())
        figures = make_figures(logs, motion, origin, args.scan_index)
        folder = args.output_dir / motion
        folder.mkdir(parents=True, exist_ok=True)
        with PdfPages(folder / 'plots.pdf') as pdf:
            for name, fig in figures.items():
                fig.savefig(folder / f'{name}.png', dpi=200)
                pdf.savefig(fig)
                if not args.show:
                    plt.close(fig)
        for sensor, log in logs.items():
            duration = (int(log.stamps[-1]) - int(log.stamps[0])) / 1e9
            invalid = ''
            if sensor == 'laser':
                _, _, invalid = scan_points(log, args.scan_index)
            summary.append([motion, sensor, str(log.path), len(log.stamps),
                            f'{duration:.6f}', int(np.sum(np.diff(log.stamps) == 0)),
                            args.scan_index if sensor == 'laser' else '', invalid])
            print(f'{motion}/{sensor}: {len(log.stamps)} samples, {duration:.3f} s')
        print(f'Saved {len(figures)} figures and plots.pdf in {folder}')
    with (args.output_dir / 'validation_summary.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['motion', 'sensor', 'source', 'samples', 'duration_s',
                         'duplicate_timestamps', 'scan_index', 'discarded_scan_beams'])
        writer.writerows(summary)
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
