"""Trial orchestration and timestamp checks without running ROS motion."""

import csv
import io
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from plot_lab3 import REQUIRED, load_recording
import plot_lab3
import run_lab3_trial as runner


def recording(path, *, bad_value=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Nonstandard column order exercises name-based parsing. Integer epoch
    # timestamps differ by ns amounts that cannot survive float conversion.
    headers = sorted(REQUIRED)
    with path.open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(headers + [''])
        for i in range(3):
            values = {key: 0.0 for key in headers}
            values['stamp'] = 1_700_000_000_000_000_007 + i * 34_000_001
            values['imu_stamp'] = values['stamp'] - 3_000_000
            values['odom_x'] = i * 0.01
            values['kf_x'] = i * 0.009
            if bad_value:
                values['imu_ax'] = float('nan')
            writer.writerow([values[key] for key in headers] + [''])


def test_loader_keeps_all_samples_and_integer_timestamp_precision(tmp_path):
    source = tmp_path / 'robot_pose.csv'
    recording(source)
    _, data, elapsed, _ = load_recording(tmp_path)
    assert len(elapsed) == 3
    assert data['stamp'].dtype.kind in 'iu'
    assert data['stamp'][0] == 1_700_000_000_000_000_007
    np.testing.assert_allclose(elapsed, [0, 0.034000001, 0.068000002], rtol=0, atol=1e-15)
    np.testing.assert_allclose(data['odom_x'], [0, 0.01, 0.02])


def test_loader_rejects_nonfinite_measurements(tmp_path):
    source = tmp_path / 'robot_pose.csv'
    recording(source, bad_value=True)
    with pytest.raises(ValueError, match='non-finite'):
        load_recording(source)


@pytest.mark.parametrize('duration_expired', [False, True])
def test_runner_saves_snapshot_logs_and_stops_before_plotting(tmp_path, monkeypatch, duration_expired):
    launched, stops, plotted, signals = [], [], [], []

    class FakeChild:
        pid = 12345

        def __init__(self):
            self.stdout = io.StringIO('synthetic child output\n')
            self.returncode = None
            self.first_wait = True

        def wait(self, timeout=None):
            if duration_expired and self.first_wait:
                self.first_wait = False
                raise subprocess.TimeoutExpired('synthetic', timeout)
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

    def fake_popen(command, cwd, **kwargs):
        launched.append(command)
        directory = Path(cwd)
        recording(directory / 'robot_pose.csv')
        (directory / 'linear.csv').write_text('e,stamp\n0,1\n')
        (directory / 'angular.csv').write_text('e,stamp\n0,1\n')
        return FakeChild()

    def fake_stop(command, **kwargs):
        stops.append(command)
        return SimpleNamespace(returncode=0)

    def fake_plot(path):
        assert stops, 'Robot stop must be attempted before plotting'
        plotted.append(path)
        return path / 'plots'

    monkeypatch.setattr(runner.subprocess, 'Popen', fake_popen)
    monkeypatch.setattr(runner.subprocess, 'run', fake_stop)
    monkeypatch.setattr(runner.os, 'killpg', lambda pid, sig: signals.append(sig))
    monkeypatch.setattr(plot_lab3, 'plot_run', fake_plot)
    code = runner.main([
        '--motion', 'point', '--sim', '--q', '0.1', '--r', '1.0',
        '--duration', '120', '--name', 'point_check', '--output-dir', str(tmp_path),
    ])
    assert code == 0
    folders = list(tmp_path.iterdir())
    assert len(folders) == 1
    folder = folders[0]
    metadata = json.loads((folder / 'trial.json').read_text())
    assert metadata['status'] == ('duration_limit_reached' if duration_expired else 'exited')
    assert metadata['stop_command_succeeded']
    assert metadata['q_scale'] == 0.1 and metadata['r_scale'] == 1.0
    assert 'use_sim_time:=true' in launched[0]
    assert 'q_scale:=0.1' in launched[0]
    assert 'r_scale:=1.0' in launched[0]
    assert Path(launched[0][2]) == folder / 'code' / 'decisions.py'
    assert (folder / 'linear.csv').exists() and (folder / 'angular.csv').exists()
    assert 'synthetic child output' in (folder / 'console.log').read_text()
    assert len(metadata['source_sha256']) == 9
    assert plotted == [folder]
    assert signals == ([signal.SIGINT] if duration_expired else [])


def test_runner_rejects_invalid_parameters_before_creating_run(tmp_path):
    with pytest.raises(SystemExit) as error:
        runner.main(['--q', 'nan', '--output-dir', str(tmp_path)])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []
