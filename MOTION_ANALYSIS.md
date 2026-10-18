# Lab 1 motion analysis — 2026-10-09

The plots represent physically consistent sensor readings. The circle closely
matches its intended geometry; the spiral expands correctly but includes repeated
slowdowns; the line accelerates as intended but curves right and falls short of
the configured maximum speed. These are measured deviations to discuss in the
report, rather than evidence that the plotting is wrong.

This review covers all trajectory, odometry, IMU, and lidar panels for the three
current runs in `data/`, against Part 7, the rubric, and Tutorial 4. Comparisons
use the **current** parameters in [motions.py](motions.py). `/cmd_vel` was not
recorded, so we cannot prove that these exact commands were used during capture.

## Intended versus measured motion

| Motion | Current code | Measured result | Assessment |
|---|---|---|---|
| Circle | v = 0.30 m/s; ω = 0.60 rad/s; R = 0.50 m | Fitted R = **0.5042 m**; median v = **0.2983 m/s**, ω = **0.5888 rad/s** | Radius differs by **0.84%**; good circular geometry, with brief speed dips |
| Spiral | v = 0.30 m/s; R = min(0.50 + 0.010t, 1.50) m; ω = v/R | Median v = **0.2966 m/s**; moving-radius fit grows **0.01056 m/s** from approximately **0.498 m** | Expansion agrees approximately; execution has interruptions |
| Line | v = min(0.10t, 0.50) m/s; ω = 0 | Ramp slope **0.0954 m/s²**; late speed **0.4430 m/s**; net yaw **−12.05°** | Acceleration within about **4.6%** of target; late speed **11.4%** low; not perfectly straight |

Here, programmed `t` starts at the first generated motion command. Graph time
starts at the earliest sensor sample. These have different time origins within
the same run. No numerical acceptance tolerance is specified by the lab, so the
percentages describe agreement rather than an official pass/fail threshold.

## What each plot means

### Circle

- **x–y path:** The overlapping revolutions fit a 0.5042 m circle. Radial fit RMS
  is about **1.2 mm within the odometry estimate**, not measured physical accuracy.
  About **1.97 revolutions** were recorded, so start and end need not coincide.
- **x/y versus time:** Oscillations and their phase offset are expected from
  circular motion. Absolute coordinate offsets are the odometry origin, not an
  error. Brief slowdowns stretch the oscillations in time.
- **Yaw and gyro:** Unwrapped yaw increases almost linearly; its slope agrees
  with positive gyro z near 0.6 rad/s. Both speed and turning rate dip around
  10, 14, 18, and 21 s. Their proportional reduction preserves the circle's shape.
- **Acceleration:** Constant speed still needs inward acceleration:
  `a_y ≈ vω = 0.18 m/s²`, assuming IMU axes aligned with the robot. The moving
  IMU mean is **0.1785 m/s²**, while mean `a_x` is **−0.0119 m/s²**. Raw spikes
  prevent precise instantaneous acceleration verification; averaging supports
  the overall expected lateral acceleration.

See [circle engineering checks](plots/verification/circle_checks.png).

### Spiral

- **x–y path:** Successive loops spread outward. `R = v/ω` is the instantaneous
  turning radius, not necessarily distance from one fixed spiral centre.
- **x/y versus time:** Oscillations become broader and slower as turning radius
  grows and angular velocity decreases. Yaw increases with a decreasing slope.
- **Gyro and speed:** Turning rate falls from roughly 0.6 toward 0.24 rad/s.
  Gyro and odometry turning rates have correlation **0.990**, including dips;
  the dips coincide with translation slowing too. Speed is below 0.15 m/s for
  **8.65 s**, approximately **11.6%** of the analysed post-start window. Mean
  speed is therefore **0.2634 m/s**, despite a near-target median. This supports
  genuine execution interruptions in the sensor estimates; the cause is unknown.
- **Radius:** A fit using moving intervals gives growth **0.01056 m/s**, about
  **5.6%** above the intended slope, with radius residual RMS **0.053 m**.
  Near-zero angular rates are excluded from this ratio because division becomes
  unstable. Slowdowns remain visible in the speed and gyro plots.
- **Acceleration:** Expected lateral acceleration decreases as `vω` decreases;
  acceleration/deceleration during interruptions can also produce `a_x` spikes.
  Noisy acceleration is plausible but does not verify every spike's cause.

The run lasts about 78 s; the **1.50 m radius cap requires 100 s of command
execution**, so its final constant-radius behaviour was not tested.
See [spiral engineering checks](plots/verification/spiral_checks.png).

### Accelerating line

- **x–y path:** Travel is **2.527 m**, initially along a heading of approximately
  −45°. A diagonal line in odometry coordinates is valid; “forward” does not
  require increasing only global x. The endpoint is **0.214 m right** of the
  straight line along the initial heading.
- **x/y versus time:** Increasing slopes during the ramp, followed by nearly
  constant slopes, show acceleration then approximately constant speed.
- **Yaw and gyro:** Yaw drifts **−12.05°** and late turning rate is about
  **−0.053 rad/s**, instead of zero. IMU integration reproduces
  the logged heading change to within **0.042°** as a numerical crosscheck.
  The bend is supported by both signals, not just the trajectory's appearance.
- **Acceleration:** The ramp's mean IMU `a_x` is **0.1108 m/s²**, approximately
  consistent with the 0.10 m/s² command despite substantial sample noise.
  After reaching steady speed, ideal forward acceleration would approach zero.
  The turn means lateral acceleration need not stay zero either.

The line function's comment still says **0.02 m/s² and 0.15 m/s**, while its
executed formula uses **0.10 m/s² and 0.50 m/s**. The table uses the formula.
See [line engineering checks](plots/verification/line_checks.png).

### Lidar — all three runs

All scans contain 720 beams, separated by approximately **0.4993°**. For scan 0,
the circle/spiral/line plots retain **448/397/418** finite positive returns and
discard **272/323/302** invalid beams. Cartesian radii reproduce retained ranges
to floating-point precision (maximum error below **10⁻¹⁵ m**). Original beam
indices are preserved, so dropping invalid ranges does not rotate later points.

The scattered clusters and segments are plausible obstacle returns. Gaps are
consistent with invalid/out-of-range readings and should not be filled in as
walls. These are **single startup scans**, before clear movement, so they show
the surroundings rather than the circle, spiral, or line trajectory.

Tutorial 4 permits the first beam to be drawn along local +x. Absolute sensor
alignment would require `angle_min` and frame information; checking hardware
range limits would require `range_min`/`range_max`. These fields were not logged.
Thus the conversion is verified, but the exact room geometry and absolute
orientation cannot be established. See the official
[LaserScan definition](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/LaserScan.msg).

## Crosschecks, method, and limits

All nine CSVs have consistent rows, finite odometry/IMU values, and strictly
increasing timestamps. Median sampling rates are approximately **20 Hz odometry,
33 Hz IMU, and 7.4 Hz lidar**. The largest spiral lidar gap is **0.282 s**, about
two normal intervals; complete delivery cannot be claimed from these logs.

Speed is computed from `hypot(Δx, Δy)/Δt`; yaw rate from
`Δunwrap(theta)/Δt`, at interval midpoint times. Movement onset is inferred from
the first speed above 0.02 m/s. Statistics omit the following first second and
the last 0.2 s. Line steady statistics additionally start 5.5 s after inferred
onset; its acceleration fit uses onset +0.5 to +4.5 s. Spiral radius fitting uses
only v >0.27 m/s and ω >0.18 rad/s. These are analysis choices, not robot settings.

Trapezoidal integration of gyro z over the exact overlapping timestamp interval
agrees with odometry's net yaw change:

| Run | Odometry yaw change | Integrated gyro | Difference |
|---|---|---|---|
| Circle | 12.3942 rad | 12.3841 rad | −0.580° |
| Spiral | 23.5581 rad | 23.5533 rad | −0.277° |
| Line | −0.21029 rad | −0.20956 rad | +0.042° |

IMU units are m/s² and rad/s, per the
[Imu definition](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/Imu.msg).
The body-axis interpretation assumes forward x, left y, up z and positive
counterclockwise yaw, following
[ROS REP 103](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst).
Odometry pose belongs to its header frame, per the
[Odometry definition](https://github.com/ros2/common_interfaces/blob/humble/nav_msgs/msg/Odometry.msg).
Frame IDs, IMU mounting transforms, and fusion configuration were not recorded;
odometry may already use the IMU. Agreement therefore establishes consistency,
not independent ground truth or proof of a specific physical fault.

Diagnostic command curves use current-code formulas and estimate command start
as the first lidar timestamp +0.05 s (half the timer period). They are guides,
not captured commands. Circle acceleration diagnostics show 1 s sample means;
the original Part 7 plots retain raw measurements.

Recreate the checks with NumPy and Matplotlib installed:

```bash
MPLBACKEND=Agg python3 plots/verification/analyze.py
```

[metrics.json](plots/verification/metrics.json) stores the numerical results and
source CSV hashes. Update the analysis formulas if motion parameters change.

For another run, preserve these CSVs first and record `/cmd_vel` plus full sensor
messages alongside them. Check for another velocity publisher and capture robot
status during slowdowns. External distance/heading measurements would help
distinguish actual path error from odometry error. These recordings could resolve
the interruptions and line drift; current logs do not identify their cause.
