# Lab 3 EKF derivation and implementation

This follows the six-state LabThree starter and Tutorial 6, especially slides
23–34. Localization IV develops Gaussian prediction/correction; Localization V
(slides numbered 95–98) explains the Jacobians and their evaluation points.
The tutorial's earlier three-state GPS example is explanatory, not our sensor
model. The implementation uses odometry twist and IMU acceleration.

## State, measurements, and assumptions

Let the state be

$$\mathbf{x}=[p_x,p_y,\theta,\omega,v,a]^T,\qquad a=\dot v.$$

Positions are in the odometry/world frame. Heading is in radians, angular
velocity in rad/s, forward velocity in m/s, and acceleration in m/s². The
measurement order is

$$\mathbf{z}=[v_{odom},\omega_{odom},a_{x,IMU},a_{y,IMU}]^T.$$

Assume planar motion, negligible lateral slip, and a body-aligned IMU whose
horizontal accelerations have no gravity component. For body velocity
$[v,0]^T$, rotating the body frame contributes lateral acceleration $v\omega$:

$$a_x=a,\qquad a_y=v\omega.$$

Verify the actual IMU topic, frame orientation, and stationary/moving readings
in Gazebo and on the robot before interpreting experimental results. The
code does not rotate an arbitrarily mounted IMU or estimate sensor bias.

## Motion model and its Jacobian

The starter uses forward Euler integration:

$$f(\mathbf{x})=\begin{bmatrix}
p_x+v\cos\theta\,\Delta t\\
p_y+v\sin\theta\,\Delta t\\
\theta+\omega\Delta t\\
\omega\\
v+a\Delta t\\
a
\end{bmatrix}.$$

Angular velocity and forward acceleration persist between updates. Velocity
changes over the step, while position uses the previous velocity: no extra
$\tfrac12a\Delta t^2$ term is added to this specified discretization.

The nontrivial position derivatives are

$$\frac{\partial f_1}{\partial\theta}=-v\sin\theta\Delta t,
\quad\frac{\partial f_1}{\partial v}=\cos\theta\Delta t,$$
$$\frac{\partial f_2}{\partial\theta}=v\cos\theta\Delta t,
\quad\frac{\partial f_2}{\partial v}=\sin\theta\Delta t.$$

Thus, in the same state order,

$$A=\frac{\partial f}{\partial\mathbf{x}}=\begin{bmatrix}
1&0&-v\sin\theta\Delta t&0&\cos\theta\Delta t&0\\
0&1& v\cos\theta\Delta t&0&\sin\theta\Delta t&0\\
0&0&1&\Delta t&0&0\\
0&0&0&1&0&0\\
0&0&0&0&1&\Delta t\\
0&0&0&0&0&1
\end{bmatrix}.$$

Evaluate $A$ at the previous corrected state. Predict the mean using $f$,
not $A\mathbf{x}$:

$$\mathbf{x}^-=f(\mathbf{x}^+_{previous}),\qquad P^-=AP^+_{previous}A^T+Q.$$

## Measurement model and its Jacobian

$$h(\mathbf{x})=\begin{bmatrix}v\\\omega\\a\\v\omega\end{bmatrix}.$$

In particular, $\partial(v\omega)/\partial\omega=v$ and
$\partial(v\omega)/\partial v=\omega$. Therefore

$$H=\frac{\partial h}{\partial\mathbf{x}}=\begin{bmatrix}
0&0&0&0&1&0\\
0&0&0&1&0&0\\
0&0&0&0&0&1\\
0&0&0&v&\omega&0
\end{bmatrix}.$$

Evaluate $H$ at the **predicted** state. The starter calls this matrix `C`;
`jacobian_H()` supplies it. This evaluation is after motion prediction,
which matters when acceleration changes the predicted velocity.

$$S=HP^-H^T+R,\qquad K=P^-H^TS^{-1},$$
$$\mathbf{x}^+=\mathbf{x}^-+K\big(\mathbf{z}-h(\mathbf{x}^-)\big).$$

The implementation solves a linear system instead of computing an explicit
inverse. It uses the equivalent Joseph covariance update for numerical
stability:

$$P^+=(I-KH)P^-(I-KH)^T+KRK^T.$$

Heading is wrapped to $[-\pi,\pi)$. The measurement vector contains no angle,
so its innovation has no angular wrap operation.

$P,Q,A$ are 6×6, $R,S$ are 4×4, $H$ is 4×6, and $K$ is 6×4.
`P` represents uncertainty in the estimate, `Q` process noise, and `R`
measurement noise. `Q` is added per accepted prediction, matching the lab
starter; it is not a continuous-time spectral density. Keep sensor rates
consistent when comparing tuning runs.

## ROS integration and initialization

`localization.py` subscribes to `/odom` and `/imu` with sensor-data QoS and
the provided approximate synchronizer (queue 10, slop 0.1 s). This is the
starter's paired-update approach; a higher-rate asynchronous filter is not
implemented. Inspect pair timing before changing the synchronizer settings.

The first accepted pair anchors position/heading once to the odometry pose.
The velocity/angular velocity/acceleration prior is zero; that pair performs
measurement correction without integrating an artificial first interval.
Later predictions use differences between accepted odometry header stamps.
Duplicate or backward timestamps are ignored. Restart the localization node
after resetting the simulation clock or odometry frame.

There is no repeated absolute-pose correction from odometry. Odometry pose is
logged for comparison; only its forward and angular velocities enter `z`.
Without an absolute position/heading reference this is dead reckoning with
sensor fusion: drift and arbitrary displacement recovery remain limitations.

`decisions.py` spins the localizer and decision maker in one executor, then
feeds `[estimated_x, estimated_y, estimated_heading, odom_header_stamp]` to the
existing controller. PID tuning, point target and supplied spiral commands
are unchanged. No physical robot motion is needed to run the offline tests.

## Parameters and logs

Defaults: `Q = 0.5 I6`, `R = 0.5 I4`, `initial_P = 0.1 I6`. These are starting
settings for the prescribed experiments, not tuned final values.

After launching the lab's Gazebo/robot environment, examples from this folder:

```bash
python3 decisions.py --motion spiral --ros-args -p q_scale:=0.1 -p r_scale:=0.5 -p log_file:=test_results/lab3_spiral_q01/robot_pose.csv
python3 decisions.py --motion point --ros-args -p q_scale:=0.5 -p r_scale:=0.5 -p log_file:=test_results/lab3_point_baseline/robot_pose.csv
```

Use a fresh output folder for each run. The existing logger overwrites the
selected CSV at startup. Point-controller `linear.csv` and `angular.csv`
remain in the working directory and should also be archived if needed.
Override `odom_topic` or `imu_topic` with ROS parameters if necessary.

The CSV contains raw IMU ax/ay, estimated ax/ay, raw and estimated v/omega,
raw and estimated position/heading, and both sensor timestamps (integer ns).
`robot_pose.settings.json` records the actual Q/R/initial P, initialized state,
topics, synchronization settings, and timing/frame assumptions. Read CSV
columns by name when making report plots and convert elapsed ns to seconds.
The starter plotter is not a final report plotting workflow.

## Saved trial runner and plotter

The shell helper previously shown in chat was meant to be pasted into a
terminal. Its replacement is the saved `run_lab3_trial.py` utility. The
localizer still writes the actual sensor/estimate CSV; the runner organizes
and supervises the experiment and calls `plot_lab3.py` afterward.

Source the ROS/Gazebo environment, launch Gazebo in another terminal, and
restart Gazebo before each independent comparison run. From the repo root:

```bash
python3 run_lab3_trial.py --sim --name lab3_spiral_q_low --q 0.1 --r 0.5
python3 run_lab3_trial.py --sim --name lab3_spiral_q_high --q 1.0 --r 0.5
python3 run_lab3_trial.py --sim --name lab3_spiral_r_low --q 0.5 --r 0.1
python3 run_lab3_trial.py --sim --name lab3_spiral_r_high --q 0.5 --r 1.0
```

Run these individually, resetting Gazebo between them. Spiral is the default
motion and each recording has a 30-second wall-clock limit. Compare common
elapsed sensor-time windows afterward. A point-controller check can use:

```bash
python3 run_lab3_trial.py --sim --motion point --name lab3_point_baseline --q 0.5 --r 0.5 --duration 120
```

The runner creates `test_results/<name>_<timestamp>/` for every invocation,
so repetitions do not overwrite previous recordings. It runs a copy of the
current runtime code in that folder. Contents include:

- `robot_pose.csv` and `robot_pose.settings.json` from localization;
- `linear.csv` and `angular.csv` for controller motions;
- `code/` snapshots of runtime files and the run/plot tools;
- `trial.json` with arguments, timestamps, source hashes and exit/stop status;
- `console.log` and `stop_command.log`;
- `plots/measurements_vs_estimates.png` and `.pdf`;
- `plots/pose_comparison.png` and `.pdf`, and `plots/summary.json`.

At the duration limit or Ctrl+C, the runner shuts down its child process and
attempts an explicit zero-velocity publication before plotting. If that
publication fails, it reports the failure and returns a nonzero status. A
run interrupted early retains its partial logs. `--no-plot` defers plotting.
The utility does not launch or reset Gazebo automatically.

To plot any existing recording without ROS, Gazebo or robot motion:

```bash
python3 plot_lab3.py test_results/lab3_spiral_baseline
```

Pass one or more run directories or CSV paths. The plotter keeps all rows,
uses named columns, preserves integer timestamps, labels axes with units,
and exports raw-versus-estimated comparisons. Its plots and summary do not
claim odometry is independent ground truth.

## Verification

Run `python3 -m pytest -q tests` from the repository root. The tests check
Jacobians against numerical derivatives, nonlinear update linearization,
noise reduction and covariance validity, and synchronized callback timing and
CSV contents using real ROS message types without starting DDS or a robot.

Run `python3 tests/ros_smoke.py` for the separate ROS integration check. It
uses localhost-only DDS in domain 72, isolated odometry/IMU topics, temporary
output files, and no velocity publisher. This check exercises actual message
delivery, QoS compatibility, synchronization, parameters, and CSV logging.

Gazebo motion, actual robot topic/QoS settings, physical sensor frames, and
final covariance tuning still require experimental validation.
