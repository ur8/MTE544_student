"""Regression checks for timestamp precision and lidar geometry (no ROS needed)."""

from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from filePlotter import SensorLog, draw_odometry, make_figures, read_log, scan_points


class PlotDataTests(unittest.TestCase):
    def test_epoch_timestamps_keep_nanosecond_differences_and_first_row(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'odom_content_circle.csv'
            path.write_text('x, y, th, stamp, \n'
                            '1, 2, 0, 1700000000000000000, \n'
                            '3, 4, 0, 1700000000000000001, \n')
            log = read_log(path)
            np.testing.assert_array_equal(log.columns['x'], [1, 3])
            np.testing.assert_allclose(log.elapsed(log.stamps[0]), [0, 1e-9], atol=0)

    def test_invalid_beams_do_not_change_remaining_beam_angles(self):
        log = SensorLog(Path('scan.csv'), np.array([0]), {
            'range_0': np.array([2.0]), 'range_1': np.array([np.nan]),
            'range_2': np.array([3.0]), 'range_3': np.array([np.inf]),
            'angle_increment': np.array([np.pi / 2]),
        })
        x, y, removed = scan_points(log, 0)
        np.testing.assert_allclose(x, [2, -3], atol=1e-12)
        np.testing.assert_allclose(y, [0, 0], atol=1e-12)
        self.assertEqual(removed, 2)

    def test_no_valid_lidar_returns_is_reported(self):
        log = SensorLog(Path('scan.csv'), np.array([0]), {
            'range_0': np.array([np.inf]), 'angle_increment': np.array([0.1]),
        })
        with self.assertRaisesRegex(ValueError, 'no finite positive ranges'):
            scan_points(log, 0)

    def test_backward_timestamps_and_truncated_rows_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.csv'
            for text, message in [
                    ('x,stamp\n1,2\n2,1\n', 'timestamps go backward'),
                    ('x,stamp\n1\n', 'row width'),
                    ('x,stamp\n', 'no sensor samples')]:
                path.write_text(text)
                with self.assertRaisesRegex(ValueError, message):
                    read_log(path)

    def test_heading_plot_unwraps_pi_jump_and_labels_units(self):
        log = SensorLog(Path('odom.csv'), np.array([0, 1_000_000_000]), {
            'x': np.array([0., 1.]), 'y': np.array([0., 0.]),
            'th': np.array([3., -3.]),
        })
        fig, axes = plt.subplots(2, 1)
        try:
            draw_odometry(axes, log, 'circle', 0)
            np.testing.assert_allclose(axes[1].lines[0].get_ydata(), [3., 2*np.pi-3.])
            np.testing.assert_allclose(axes[1].lines[0].get_xdata(), [0., 1.])
            self.assertEqual(axes[1].get_ylabel(), 'Yaw (rad)')
        finally:
            plt.close(fig)

    def test_overview_panels_do_not_overlap(self):
        stamps = np.array([0, 1_000_000_000])
        logs = {
            'odom': SensorLog(Path('odom.csv'), stamps, {
                'x': np.array([0., 1.]), 'y': np.array([0., 1.]),
                'th': np.array([0., 0.5]),
            }),
            'imu': SensorLog(Path('imu.csv'), stamps, {
                'acc_x': np.array([0., 0.1]), 'acc_y': np.array([0., 0.2]),
                'angular_z': np.array([0., 0.5]),
            }),
            'laser': SensorLog(Path('scan.csv'), stamps, {
                'range_0': np.array([1., 1.]), 'range_1': np.array([2., 2.]),
                'angle_increment': np.array([np.pi / 2, np.pi / 2]),
            }),
        }
        figures = make_figures(logs, 'circle', 0, 0)
        try:
            overview = figures['overview']
            overview.canvas.draw()
            boxes = [ax.get_position() for ax in overview.axes]
            self.assertEqual(len(boxes), 6)
            for i, box in enumerate(boxes):
                for other in boxes[i + 1:]:
                    self.assertFalse(box.overlaps(other))
        finally:
            for fig in figures.values():
                plt.close(fig)


if __name__ == '__main__':
    unittest.main()
