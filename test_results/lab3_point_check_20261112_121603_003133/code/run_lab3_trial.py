"""Record one Lab 3 trial into a new folder, stop motion, then save plots."""

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parent
RUNTIME_FILES = [
    'decisions.py', 'localization.py', 'kalman_filter.py', 'controller.py',
    'planner.py', 'pid.py', 'utilities.py',
]


def build_command(options, run_dir):
    """Execute the saved code snapshot, with every CSV relative to its run."""
    return [
        sys.executable, '-u', str(run_dir / 'code' / 'decisions.py'),
        '--motion', options.motion, '--ros-args',
        '-p', f'use_sim_time:={str(options.sim).lower()}',
        '-p', f'q_scale:={float(options.q)}', '-p', f'r_scale:={float(options.r)}',
        '-p', f'odom_topic:={options.odom_topic}', '-p', f'imu_topic:={options.imu_topic}',
        '-p', f'log_file:={run_dir / "robot_pose.csv"}',
        '-r', f'/cmd_vel:={options.cmd_vel_topic}',
    ]


def terminate_child(child):
    """Allow normal shutdown first; escalate only for this child group."""
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]:
        if child.poll() is not None:
            return
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue


def stop_motion(topic, run_dir):
    """Send the same explicit zero Twist used by the former shell helper."""
    command = [
        'ros2', 'topic', 'pub', '--once', topic, 'geometry_msgs/msg/Twist',
        '{linear: {x: 0.0}, angular: {z: 0.0}}',
    ]
    try:
        with (run_dir / 'stop_command.log').open('w') as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=8)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f'Stop command failed: {error}. Send a zero Twist manually.', file=sys.stderr)
        return False


def record_trial(options):
    """Archive source/metadata, supervise one run, and retain partial logs too."""
    # Timestamp plus microseconds makes repetitions separate recordings.
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    label = options.name or f'lab3_{options.motion}'
    run_dir = options.output_dir.resolve() / f'{label}_{stamp}'
    run_dir.mkdir(parents=True, exist_ok=False)
    code_dir = run_dir / 'code'
    code_dir.mkdir()
    for filename in RUNTIME_FILES + ['run_lab3_trial.py', 'plot_lab3.py']:
        shutil.copy2(ROOT / filename, code_dir / filename)
    command = build_command(options, run_dir)
    metadata = {
        'started_at': datetime.now().astimezone().isoformat(),
        'motion': options.motion, 'q_scale': options.q, 'r_scale': options.r,
        'use_sim_time': options.sim, 'duration_limit_wall_s': options.duration,
        'command': command, 'working_directory': str(run_dir),
        'ROS_DOMAIN_ID': os.environ.get('ROS_DOMAIN_ID', '0'),
        'ROS_LOCALHOST_ONLY': os.environ.get('ROS_LOCALHOST_ONLY'),
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(code_dir.glob('*.py'))},
        'status': 'starting',
    }
    metadata_path = run_dir / 'trial.json'
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Recording to: {run_dir}', flush=True)
    print('Gazebo must already be running. Ctrl+C ends this recording and saves its plots.', flush=True)
    child = None
    reader = None
    try:
        with (run_dir / 'console.log').open('w') as console:
            child = subprocess.Popen(
                command, cwd=run_dir, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
            )

            def tee_output():
                for line in child.stdout:
                    console.write(line)
                    console.flush()
                    print(line, end='', flush=True)

            reader = threading.Thread(target=tee_output, daemon=True)
            reader.start()
            try:
                child.wait(timeout=options.duration)
                metadata['status'] = 'exited' if child.returncode == 0 else 'process_failed'
            except subprocess.TimeoutExpired:
                metadata['status'] = 'duration_limit_reached'
            except KeyboardInterrupt:
                metadata['status'] = 'interrupted'
            finally:
                terminate_child(child)
                reader.join(timeout=2)
                metadata['process_returncode'] = child.returncode
    except OSError as error:
        metadata['status'] = 'launch_failed'
        metadata['error'] = str(error)
        print(f'Could not launch trial: {error}', file=sys.stderr)
    finally:
        if child is not None:
            metadata['stop_command_succeeded'] = stop_motion(options.cmd_vel_topic, run_dir)
        metadata['finished_at'] = datetime.now().astimezone().isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')

    if not options.no_plot:
        from plot_lab3 import plot_run
        try:
            output = plot_run(run_dir)
            print(f'Saved plots: {output}')
        except (OSError, ValueError) as error:
            metadata['plot_error'] = str(error)
            metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
            print(f'Plots unavailable: {error}', file=sys.stderr)
    print(f'Trial status: {metadata["status"]}. Files retained in {run_dir}')
    if metadata['status'] in {'launch_failed', 'process_failed'}:
        return 1
    if child is not None and not metadata.get('stop_command_succeeded', False):
        print('Stop publication was not confirmed; see stop_command.log.', file=sys.stderr)
        return 1
    return 1 if metadata.get('plot_error') else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', choices=['spiral', 'point', 'trajectory'], default='spiral')
    parser.add_argument('--name', help='Trial label; a timestamp is always appended')
    parser.add_argument('--q', type=float, default=0.5, help='Q = q * I6')
    parser.add_argument('--r', type=float, default=0.5, help='R = r * I4')
    parser.add_argument('--duration', type=float, default=30, help='Wall-clock seconds; point runs can use 120')
    parser.add_argument('--sim', action='store_true', help='Use simulation time (Gazebo)')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'test_results')
    parser.add_argument('--odom-topic', default='/odom')
    parser.add_argument('--imu-topic', default='/imu')
    parser.add_argument('--cmd-vel-topic', default='/cmd_vel')
    parser.add_argument('--no-plot', action='store_true', help='Record now and plot later')
    options = parser.parse_args(argv)
    if not math.isfinite(options.q) or options.q < 0:
        parser.error('--q must be finite and nonnegative')
    if not math.isfinite(options.r) or options.r <= 0:
        parser.error('--r must be finite and positive')
    if not math.isfinite(options.duration) or options.duration <= 0:
        parser.error('--duration must be finite and positive')
    if options.name and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', options.name):
        parser.error('--name must contain only letters, numbers, underscores or hyphens')
    return record_trial(options)


if __name__ == '__main__':
    raise SystemExit(main())
