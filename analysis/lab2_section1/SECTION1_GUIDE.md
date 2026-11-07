# Lab 2 Section 1 — figure guide and draft captions

Reproduce from the repository root: `python3 plot_section1.py`.

`section1_report_insert.pdf` contains a two-page Section 1 insert, leaving one body page for Section 2. Standalone PNG and vector PDF figures are in `figures/`; optional diagnostics are in `supplement/`. This is a figure insert, not the complete report or a Section 2 submission.

## Data and selection

Only this group's timestamped logs are used. The other group's PDF was inspected for coverage only; it is not read by this script and none of its data, captions, gains or numerical results are used.

| Role | Source folder | Selection basis |
|---|---|---|
| point_p | `part5P_3_20261030_164626` | Smallest angular overshoot among the Part 5 P trials |
| point_pid | `part5PID_220261030_162103` | Smallest angular overshoot among the Part 5 PID trials |
| parabola | `part6_2_20261030_170136` | Smallest time-weighted RMS geometric path error |
| sigmoid | `part6_sigmoid_2_20261030_172213` | Smallest time-weighted RMS geometric path error |

These are deliberate tracking/overshoot selections, not universally optimal gains. The Part 4 P baseline and every Part 5 in-lab trial remain visible in the tuning figure. The main point comparison does not mix Gazebo initial conditions with in-lab initial conditions.

Point-controller P/PID labels follow the folder names. In-lab Part 6 PID use was explicitly confirmed by the user. Exact in-lab per-trial gains are unavailable: the intermediate in-lab commits add CSVs without preserving corresponding controller settings. The final working controller settings are not assigned retrospectively to older trials.

## Draft figure captions

1. **Point errors.** Recorded linear distance error and wrapped bearing error, with their logged derivatives, for selected P and PID runs. Dotted references indicate zero error and the 5 cm position tolerance. Both logs retain all recorded samples; each run shares one pose-based time origin.

2. **Point pose.** Recorded x, y and heading versus elapsed time for the same selected runs. Position targets are x = −1.0 m and y = −0.8 m. The controller does not specify a final heading.

3. **Path and phase portraits.** Recorded x–y paths and linear/angular error–error-rate trajectories. Circles and squares mark first and last samples. The cross and circle indicate the point goal and its 5 cm acceptance region. Finite recorded trajectories do not establish global stability.

4. **Point tuning trials.** Every archived in-lab Part 5 P and PID trial is compared, with the Part 4 P run retained as a baseline. Bold curves indicate the selected trials. Selected P trial 3 has 1.80° peak angular-error overshoot and reaches the position tolerance in 17.04 s; PID trial 2 has 2.75° and 12.90 s. Trial-to-trial differences show tuning outcomes, but missing gain records prevent claims that an individual gain change caused a particular improvement.

5. **Trajectory tracking.** Recorded in-lab PID paths are compared with y = x² on [0, 1.5] m and y = 2/(1 + exp(−2x)) − 1 = tanh(x) on [0, 2.5] m. Both trials are retained, and the selected trial is emphasised. Squares identify last recorded poses; crosses identify the planned endpoints.

## Trajectory selection rationale (draft)

PID was retained for Part 6 because it completed both prescribed shapes within the 5 cm goal tolerance, with sub-centimetre RMS geometric tracking error in the selected runs. PID was used for all in-lab Part 6 runs (group confirmation). The second trial is selected for each trajectory because it reduces RMS and peak geometric tracking error, accepting a slower completion time. The available in-lab data support a choice between these PID trials; they do not establish that PID outperforms P for these trajectories.

| Shape | Trial | Observed arrival (s) | RMS path error (cm) | Peak path error (cm) |
|---|---|---:|---:|---:|
| parabola | 1 | 28.16 | 0.668 | 1.794 |
| parabola | 2 (selected) | 34.29 | 0.507 | 1.205 |
| sigmoid | 1 | 34.59 | 1.035 | 5.282 |
| sigmoid | 2 (selected) | 43.04 | 0.652 | 4.047 |

## Interpretation and verification

- Timestamp integers are subtracted before conversion to seconds; no first data row is skipped.
- Each run uses the first pose stamp for all three logs. Error logging starts slightly later because the PID implementation waits for three history samples; missing derivatives are not invented.
- Derivatives are the recorded controller estimates, not a newly differentiated or smoothed signal. Data are not smoothed, rescaled in time, extended after logging, or normalised to hide differences.
- Point linear error is unsigned distance to the goal. Angular overshoot is the largest error on the opposite side of zero from the initial angular error, expressed in degrees; it is not position overshoot or overshoot relative to a fixed commanded heading.
- Arrival is the first recorded pose within 5 cm. Its time resolution is limited by odometry sampling. Logs stop around that event, so final recorded error is not measured steady-state error.
- Trajectory linear error is distance to the final goal; angular error is relative to the current look-ahead waypoint. Neither is the perpendicular path-tracking error.
- Path error is the shortest Euclidean distance to the finite continuous analytic curve, found by dense-grid localisation and scalar refinement. RMS is time weighted: sqrt(integral(distance² dt)/duration). Endpoint distances are included.
- Desired mathematical paths and zero/goal references are shown. No unverified theoretical time-response model is fitted. Gazebo comparisons are optional and explicitly labelled; their gains/robot model differ from the in-lab conditions.
- Every processed error timestamp is matched to its pose. Logged distance is checked against the stated goal, and point angular error is checked against the pose-derived goal bearing. Source CSV SHA-256 hashes, sample counts and settings provenance are in `manifest.json`.
