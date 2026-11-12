# Lab 3 spiral covariance comparison

Compared all five recordings over 0–24.786 seconds (730 samples each).
All CSVs are finite with strictly increasing timestamps; every recorded EKF state reproduces exactly.
All four new trials completed their duration limit, exited with code 0 and confirmed a zero-velocity publication.
The saved covariance matrices, initial P and code snapshots were verified.

| Setting | q | r | Position RMS difference (cm) | Maximum difference (cm) | Heading RMS difference (deg) |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.5 | 0.5 | 2.57 | 4.78 | 0.93 |
| Lower Q | 0.1 | 0.5 | 4.48 | 6.67 | 1.91 |
| Higher Q | 1 | 0.5 | 4.82 | 7.83 | 2.03 |
| Lower R | 0.5 | 0.1 | 2.60 | 4.04 | 0.92 |
| Higher R | 0.5 | 1 | 2.13 | 3.68 | 0.86 |

## Interpretation

Higher R has the lowest recorded pose disagreement, but separate recordings contain different IMU fluctuations. It is not proof of best absolute accuracy.
The controlled offline replay applies every setting to every sensor recording. Lower Q reduces steady velocity fluctuations most, while increasing pose disagreement with odometry. Higher R gives moderate smoothing with a smaller agreement penalty. Lower R follows measurements more strongly and yields more velocity jitter on the same inputs.
In the offline replay means, higher R reduces forward/angular velocity standard deviation by approximately 8%/9% relative to the baseline. Lower Q reduces them by approximately 20%/22%.
The common steady-motion IMU forward-acceleration means remain about −0.054 to −0.058 m/s² although odometry forward speed is nearly constant. Covariance changes do not remove this sensor/model discrepancy.

## Next experiment

Test point control using Q = 0.5 I6 and R = 1.0 I4 as a provisional smoothing compromise. Do not label it a final optimum. Reset Gazebo first:

```bash
python3 run_lab3_trial.py --sim --motion point --name lab3_point_r_high --q 0.5 --r 1.0 --duration 120
```

If time permits, run a point baseline (q = 0.5, r = 0.5) under the same reset conditions. Compare actual odometry distance to the goal as well as EKF distance at termination. Investigate the IMU forward-acceleration discrepancy before final physical-robot tuning.

The recording CSVs and runtime code were not modified. See comparison.json and both CSV tables for full metrics and provenance.
