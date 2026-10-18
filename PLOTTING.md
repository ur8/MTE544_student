# Lab 1 plotting (Part 7)

Run from the repository directory after collecting the CSVs in `data/`:

```bash
python3 filePlotter.py                  # All three motions; save figures
python3 filePlotter.py --motion circle # One motion
python3 filePlotter.py --show           # Also open plot windows
```

The original selection option still works:

```bash
python3 filePlotter.py --files data/imu_content_spiral.csv
```

Use `--data-dir` for an archived run, `--output-dir` for a different destination,
and `--scan-index 30` to select another lidar message (zero-based; default 0).

## Outputs and rubric coverage

Each motion gets a folder under `plots/` with these PNGs and a `plots.pdf`
containing the same figures. The CSV logs are unchanged.

| Figure | What is plotted |
|---|---|
| `odom_trajectory.png` | x versus y, equal metre scales, with start/end markers |
| `odom_timeseries.png` | x/y and yaw versus elapsed seconds; separate axes for metres and radians |
| `imu_timeseries.png` | acceleration x/y and angular velocity z versus seconds; separate axes for their different units |
| `lidar_scan.png` | One scan as Cartesian points, with equal metre scales |
| `overview.png` | All the required plots together for that motion; produced when all three sensors are selected |

Titles, labelled axes with units, legends, distinct colours/markers, and grids
follow the [rubric](rubrics.md). Individual figures and the overview can be
used in the report; the exported PDFs are supporting figures, not the final
report (the README limits the report body to two pages).

## Conversion and validation

- Keep timestamps as integer nanoseconds, subtract the earliest selected sensor
  timestamp for that motion, then divide by `1e9`. This aligns sensor plots and
  avoids losing precision with real robot timestamps. Time zero is the first
  logged sample, not necessarily the start of movement.
- Odometry `th` is already yaw in radians. Unwrap the ±pi boundary for readable
  multiple turns; no smoothing is applied to positions or IMU readings.
- For lidar beam `i`, use `theta = i * angle_increment`, then
  `x = range * cos(theta)` and `y = range * sin(theta)`. Remove `inf`, `NaN`, and
  nonpositive ranges **without renumbering the remaining beams**. As permitted
  by Tutorial 4, the first beam points along local +x; this is one local scan,
  not a global map. Our logs do not contain `angle_min` or a global scan pose.
- Reject empty files, inconsistent rows, missing required columns, backward
  timestamps, and nonfinite odometry/IMU values. Lidar filtering is reported on
  its figure. `plots/validation_summary.csv` records sample counts, durations,
  duplicate timestamps, and discarded beams for the selected scan.

## Results checked on 2026-10-09

See [MOTION_ANALYSIS.md](MOTION_ANALYSIS.md) for quantitative engineering checks,
motion deviations, and reproducible comparison figures for these runs.

All nine current CSVs loaded with consistent rows and increasing timestamps.
Generated figures were visually inspected for each motion, including panel
layout, labels, units, and equal spatial scaling.

| Motion | Odometry / IMU / lidar samples | Odometry duration | Observations |
|---|---|---|---|
| Circle | 467 / 775 / 160 | 23.392 s | Circular path; fitted odometry radius approximately 0.504 m, consistent with the commanded 0.5 m |
| Spiral | 1,554 / 2,562 / 548 | 77.743 s | Expanding loops; IMU turning rate trends downward |
| Line | 191 / 316 / 61 | 9.485 s | Mostly straight path, approximately 2.53 m travelled; heading drifts approximately −12° |

The first lidar scans retained 448, 397, and 418 of 720 beams respectively.
IMU noise and brief turning-rate dips remain visible; discuss these measured
features in the report rather than treating the motion as ideal. These checks
verify the plotting and consistency of the logs, not independent ground truth
for the physical trajectory.

Regression checks cover timestamp precision, retained first samples, lidar
beam angles after filtering, bad input, yaw wrapping, and overview layout:

```bash
python3 -m unittest discover -s tests -v
```
