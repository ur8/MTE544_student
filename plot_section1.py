"""Reproduce Lab 2 Section 1 figures from this group's timestamped CSV logs.

Run: python3 plot_section1.py
Requires numpy, scipy and matplotlib; does not import ROS or change robot logs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mte544_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parent
GOAL = np.array([-1.0, -0.8])
TOLERANCE = 0.05
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#68737D"
COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#946700"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
P_NAMES = ["part5P_120261030_164305", "part5P_2_20261030_164427", "part5P_3_20261030_164626"]
PID_NAMES = ["part5PID_20261030_154438", "part5PID_220261030_162103", "part5PID_320261030_163514", "part5PID_420261030_163921"]
BASELINE = "part4P20261030_152923"
TRAJECTORIES = {
    "parabola": ["part6PIDP_1_20261030_165312", "part6_2_20261030_170136"],
    "sigmoid": ["part6_sigmoid_1_20261030_170855", "part6_sigmoid_2_20261030_172213"],
}
SIMULATION = {
    "parabola": "part6_parabola_P_20261029_052843",
    "sigmoid": "part6_sigmoid_P_20261029_053247",
}


@dataclass
class Log:
    values: np.ndarray
    stamps: np.ndarray

    def seconds(self, origin: int) -> np.ndarray:
        # Subtract integers before conversion: epoch nanoseconds exceed float precision.
        return (self.stamps - origin).astype(float) / 1e9


@dataclass
class Run:
    name: str
    label: str
    pose: Log
    linear: Log
    angular: Log

    @property
    def origin(self) -> int:
        return int(self.pose.stamps[0])

    @property
    def duration(self) -> float:
        return float(self.pose.seconds(self.origin)[-1])

    @property
    def overshoot_deg(self) -> float:
        # All point runs begin with negative bearing error. Generalise to either sign.
        direction = np.sign(self.angular.values[0, 0])
        return float(np.rad2deg(max(0.0, np.max(-direction * self.angular.values[:, 0]))))


def read_log(path: Path, expected: list[str]) -> Log:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = [v.strip() for v in next(reader) if v.strip()]
        if header != expected:
            raise ValueError(f"Unexpected columns in {path}: {header}")
        rows = [[v.strip() for v in row if v.strip()] for row in reader]
    rows = [row for row in rows if row]
    if len(rows) < 2 or any(len(row) != len(expected) for row in rows):
        raise ValueError(f"Empty, short or malformed log: {path}")
    stamps = np.array([int(row[-1]) for row in rows], dtype=np.int64)
    values = np.array([[float(v) for v in row[:-1]] for row in rows])
    if not np.all(np.isfinite(values)) or not np.all(np.diff(stamps) > 0):
        raise ValueError(f"Nonfinite data or nonincreasing timestamps: {path}")
    return Log(values, stamps)


def load_run(folder: Path, label: str) -> Run:
    run = Run(folder.name, label,
              read_log(folder / "robot_pose.csv", ["x", "y", "theta", "stamp"]),
              read_log(folder / "linear.csv", ["e", "e_dot", "e_int", "stamp"]),
              read_log(folder / "angular.csv", ["e", "e_dot", "e_int", "stamp"]))
    if not np.array_equal(run.linear.stamps, run.angular.stamps):
        raise ValueError(f"Error logs are misaligned: {folder}")
    if not np.all(np.isin(run.linear.stamps, run.pose.stamps)):
        raise ValueError(f"An error timestamp has no matching pose: {folder}")
    return run


def curve(kind: str, x: np.ndarray | float) -> np.ndarray | float:
    return np.square(x) if kind == "parabola" else np.tanh(x)


def curve_end(kind: str) -> float:
    return 1.5 if kind == "parabola" else 2.5


def curve_distances(points: np.ndarray, kind: str) -> np.ndarray:
    """Shortest Euclidean distance to the finite continuous prescribed curve.

    Find the best neighbourhood on a dense grid, then refine on the analytic
    curve. Explicit endpoints include positions just outside the planned domain.
    This is geometric distance, not the logged distance to the final goal.
    """
    end = curve_end(kind)
    grid = np.linspace(0.0, end, 4097)
    grid_y = curve(kind, grid)
    distances = []
    for x, y in points:
        nearest = int(np.argmin((grid - x) ** 2 + (grid_y - y) ** 2))
        result = minimize_scalar(
            lambda u: (u - x) ** 2 + (curve(kind, u) - y) ** 2,
            bounds=(grid[max(0, nearest - 1)], grid[min(len(grid) - 1, nearest + 1)]),
            method="bounded", options={"xatol": 1e-12})
        distance2 = min(result.fun, x * x + y * y,
                        (end - x) ** 2 + (curve(kind, end) - y) ** 2)
        distances.append(np.sqrt(distance2))
    return np.array(distances)


def audit_run(run: Run, kind: str | None) -> dict:
    goal = GOAL if kind is None else np.array([curve_end(kind), curve(kind, curve_end(kind))])
    distance_to_goal = np.linalg.norm(run.pose.values[:, :2] - goal, axis=1)
    arrived = np.flatnonzero(distance_to_goal <= TOLERANCE)
    indices = np.searchsorted(run.pose.stamps, run.linear.stamps)
    linear_residual = np.max(np.abs(distance_to_goal[indices] - run.linear.values[:, 0]))
    if linear_residual > 1e-8:
        raise ValueError(f"Goal assumption inconsistent with logged distance: {run.name}")
    if kind is None:
        xy = run.pose.values[indices, :2]
        bearing = np.arctan2(goal[1] - xy[:, 1], goal[0] - xy[:, 0])
        angular_expected = (bearing - run.pose.values[indices, 2] + np.pi) % (2 * np.pi) - np.pi
        residual = (angular_expected - run.angular.values[:, 0] + np.pi) % (2 * np.pi) - np.pi
        if np.max(np.abs(residual)) > 1e-8:
            raise ValueError(f"Logged bearing error inconsistent with pose: {run.name}")
    row = {"folder": run.name, "label": run.label,
           "environment": "Gazebo" if "20261029" in run.name else "in-lab",
           "kind": kind or "point", "pose_samples": len(run.pose.stamps),
           "error_samples": len(run.linear.stamps), "duration_s": run.duration,
           "arrival_5cm_s": float(run.pose.seconds(run.origin)[arrived[0]]) if len(arrived) else None,
           "final_position_error_cm": float(distance_to_goal[-1] * 100),
           "angular_overshoot_deg": run.overshoot_deg if kind is None else None,
           "max_logged_distance_residual_m": float(linear_residual),
           "path_rms_cm": None, "path_max_cm": None}
    if kind:
        distances = curve_distances(run.pose.values[:, :2], kind)
        row["path_rms_cm"] = float(100 * np.sqrt(trapezoid(distances ** 2, run.pose.seconds(run.origin)) / run.duration))
        row["path_max_cm"] = float(100 * np.max(distances))
    return row


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "lines.linewidth": 1.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "grid.alpha": 0.22, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.facecolor": "white"})


def decorate(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True)
    ax.tick_params(labelsize=8)


def zero(ax, tolerance: bool = False) -> None:
    ax.axhline(0, color=GREY, lw=0.8, ls=":", zorder=0)
    if tolerance:
        ax.axhline(TOLERANCE, color=GREY, lw=1, ls=":", label="5 cm tolerance")


def point_errors(axes, selected: list[Run]) -> None:
    for j, (field, title, unit) in enumerate([
        ("linear", "Distance error", "m"), ("angular", "Bearing error", "rad")]):
        for i in range(2):
            ax = axes[i, j]
            for run, colour, ls in zip(selected, [BLUE, ORANGE], ["-", "--"]):
                log = getattr(run, field)
                ax.plot(log.seconds(run.origin), log.values[:, i], color=colour, ls=ls, label=run.label)
            zero(ax, tolerance=(i == j == 0))
            decorate(ax, title if i == 0 else f"Logged {title.lower()} derivative", "Time (s)",
                     f"Error ({unit})" if i == 0 else f"Error rate ({unit}/s)")
            ax.legend(loc="best")


def point_pose(axes, selected: list[Run]) -> None:
    for i, ax in enumerate(axes):
        for run, colour, ls in zip(selected, [BLUE, ORANGE], ["-", "--"]):
            ax.plot(run.pose.seconds(run.origin), run.pose.values[:, i], color=colour, ls=ls, label=run.label)
        if i < 2:
            ax.axhline(GOAL[i], color=GREY, ls=":", lw=1, label=f"Goal: {GOAL[i]:g} m")
        decorate(ax, ["x position", "y position", "Heading"][i], "Time (s)",
                 ["x (m)", "y (m)", r"$\theta$ (rad)"][i])
        ax.legend(loc="best")


def point_geometry(axes, selected: list[Run]) -> None:
    ax = axes[0]
    for run, colour, ls in zip(selected, [BLUE, ORANGE], ["-", "--"]):
        xy = run.pose.values[:, :2]
        ax.plot(xy[:, 0], xy[:, 1], color=colour, ls=ls, label=run.label)
        ax.scatter(*xy[0], color=colour, marker="o", s=18, zorder=4)
        ax.scatter(*xy[-1], color=colour, marker="s", s=18, zorder=4)
    ax.scatter(*GOAL, color="black", marker="x", s=35, label="Goal", zorder=5)
    ax.add_patch(Circle(GOAL, TOLERANCE, fill=False, color=GREY, ls=":", label="5 cm region"))
    ax.set_aspect("equal", adjustable="datalim")
    decorate(ax, "Recorded approach paths", "x (m)", "y (m)")
    ax.legend(loc="best")
    for ax, field, title, unit in zip(axes[1:], ["linear", "angular"], ["Distance phase portrait", "Bearing phase portrait"], ["m", "rad"]):
        for run, colour, ls in zip(selected, [BLUE, ORANGE], ["-", "--"]):
            log = getattr(run, field)
            ax.plot(log.values[:, 0], log.values[:, 1], color=colour, ls=ls, label=run.label)
            ax.scatter(log.values[0, 0], log.values[0, 1], color=colour, marker="o", s=18, zorder=4)
            ax.scatter(log.values[-1, 0], log.values[-1, 1], color=colour, marker="s", s=18, zorder=4)
        zero(ax)
        ax.axvline(0, color=GREY, ls=":", lw=0.8)
        decorate(ax, title, f"Error ({unit})", f"Error rate ({unit}/s)")
        ax.legend(loc="best")


def tuning(axes, runs: dict[str, Run], selection: dict[str, str]) -> None:
    for row, (group, names, key) in enumerate([
        ("P", [BASELINE] + P_NAMES, "point_p"), ("PID", PID_NAMES, "point_pid")]):
        for column, field in enumerate(["linear", "angular"]):
            ax = axes[row, column]
            for index, name in enumerate(names):
                run = runs[name]
                selected = name == selection[key]
                log = getattr(run, field)
                ax.plot(log.seconds(run.origin), log.values[:, 0], color=COLOURS[index],
                        ls=STYLES[index], lw=2 if selected else 1.15,
                        label=run.label + (" (selected)" if selected else ""))
            zero(ax, tolerance=column == 0)
            decorate(ax, f"{group} trials: {'distance' if column == 0 else 'bearing'} error", "Time (s)",
                     "Error (m)" if column == 0 else "Error (rad)")
            ax.legend(loc="best", fontsize=7)


def trajectory_paths(axes, runs: dict[str, Run], selection: dict[str, str], simulation: bool = False) -> None:
    for ax, kind in zip(axes, TRAJECTORIES):
        end = curve_end(kind)
        x = np.linspace(0, end, 1201)
        ax.plot(x, curve(kind, x), color="black", ls=":", lw=1.8, label="Prescribed curve")
        if simulation:
            trials = [(runs[SIMULATION[kind]], "Gazebo P", GREY, "-."),
                      (runs[selection[kind]], "Selected in-lab PID", BLUE, "-")]
        else:
            trials = [(runs[name], f"PID trial {i + 1}" + (" (selected)" if name == selection[kind] else ""),
                       BLUE if name == selection[kind] else ORANGE,
                       "-" if name == selection[kind] else "--")
                      for i, name in enumerate(TRAJECTORIES[kind])]
        for run, label, colour, ls in trials:
            xy = run.pose.values[:, :2]
            ax.plot(xy[:, 0], xy[:, 1], color=colour, ls=ls, label=label)
            ax.scatter(*xy[-1], color=colour, marker="s", s=20, zorder=4)
        ax.scatter(0, 0, color="black", marker="o", s=20, label="Plan start", zorder=4)
        ax.scatter(end, curve(kind, end), color="black", marker="x", s=35, label="Goal", zorder=5)
        ax.add_patch(Circle((end, curve(kind, end)), TOLERANCE, fill=False, color=GREY, ls=":"))
        ax.set_aspect("equal", adjustable="box")
        ax.margins(0.1)
        decorate(ax, "Parabola: y = x²" if kind == "parabola" else "Sigmoid: y = tanh(x)", "x (m)", "y (m)")
        ax.legend(loc="best", fontsize=7)


def trajectory_errors(axes, runs: dict[str, Run], selection: dict[str, str]) -> None:
    for col, kind in enumerate(TRAJECTORIES):
        for row, field in enumerate(["linear", "angular"]):
            ax = axes[row, col]
            for i, name in enumerate(TRAJECTORIES[kind]):
                run = runs[name]
                log = getattr(run, field)
                selected = name == selection[kind]
                ax.plot(log.seconds(run.origin), log.values[:, 0],
                        color=BLUE if selected else ORANGE, ls="-" if selected else "--",
                        label=f"PID trial {i + 1}" + (" (selected)" if selected else ""))
            zero(ax, tolerance=row == 0)
            decorate(ax, f"{kind.title()}: {'distance to final goal' if row == 0 else 'look-ahead bearing error'}",
                     "Time (s)", "Error (m)" if row == 0 else "Error (rad)")
            ax.legend(loc="best")


def export(fig, path: Path, title: str, note: str) -> None:
    fig.suptitle(title, fontsize=13, weight="bold")
    fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.savefig(path.with_suffix(".png"), dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def report_insert(path: Path, runs: dict[str, Run], selection: dict[str, str], metrics: dict[str, dict]) -> None:
    """Two pages of original figures/captions; leave page three for Section 2."""
    selected = [runs[selection["point_p"]], runs[selection["point_pid"]]]
    with plt.rc_context({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "legend.fontsize": 6}):
        with PdfPages(path) as pdf:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle("1. Experimental plots", x=0.075, y=0.975, ha="left", fontsize=15, weight="bold")
            fig.text(0.075, 0.94, "Point control: selected P and PID runs", fontsize=11, weight="bold")
            fig.text(0.075, 0.916, "Goal (−1.0, −0.8) m; 5 cm stopping tolerance. Curves use the recorded run labels.", fontsize=8)
            grid = fig.add_gridspec(3, 1, left=0.09, right=0.97, top=0.89, bottom=0.12,
                                   height_ratios=[2.1, 1, 1.25], hspace=0.5)
            point_errors(np.array(grid[0].subgridspec(2, 2, hspace=0.85, wspace=0.4).subplots()), selected)
            point_pose(np.array(grid[1].subgridspec(1, 3, wspace=0.55).subplots()), selected)
            point_geometry(np.array(grid[2].subgridspec(1, 3, wspace=0.6).subplots()), selected)
            note = (f"Selected {selected[0].label} and {selected[1].label} have the smallest angular-error overshoot "
                    "within their respective Part 5 trial groups. Dotted lines mark zero error, position targets or "
                    "the 5 cm region. Circles and squares mark first and last samples in path/phase plots. "
                    "The derivative is the logged controller estimate. Each run uses its first pose timestamp as t = 0. "
                    "Logs end at the position stopping threshold; final heading is not independently controlled.")
            fig.text(0.075, 0.075, textwrap.fill(note, 122), fontsize=7, va="top", linespacing=1.4)
            fig.text(0.075, 0.026, "Group 22 • Section 1 figure insert • 1 / 2", fontsize=7, color=GREY)
            pdf.savefig(fig)
            plt.close(fig)

            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle("1. Experimental plots — tuning and paths", x=0.075, y=0.975, ha="left", fontsize=14, weight="bold")
            grid = fig.add_gridspec(2, 1, left=0.09, right=0.97, top=0.92, bottom=0.23,
                                   height_ratios=[1.15, 1], hspace=0.5)
            tuning(np.array(grid[0].subgridspec(2, 2, hspace=0.85, wspace=0.4).subplots()), runs, selection)
            trajectory_paths(np.array(grid[1].subgridspec(1, 2, wspace=0.3).subplots()), runs, selection)
            p, pid = [metrics[run.name] for run in selected]
            para, sig = [metrics[selection[k]] for k in TRAJECTORIES]
            note1 = (f"Point tuning: {selected[0].label} records {p['angular_overshoot_deg']:.2f}° peak angular-error "
                     f"overshoot and arrives in {p['arrival_5cm_s']:.2f} s; {selected[1].label} records "
                     f"{pid['angular_overshoot_deg']:.2f}° and {pid['arrival_5cm_s']:.2f} s. "
                     "Other trials expose the speed/overshoot tradeoff. Per-trial gains were not archived, "
                     "so the plots support response comparisons rather than attribution to a specific gain change.")
            note2 = ("Trajectory choice: retain PID, which completed both shapes within the 5 cm goal tolerance. "
                     "Select trial 2 for each shape, prioritising path tracking. "
                     f"Time-weighted RMS distance to the prescribed curve is {para['path_rms_cm']:.2f} cm "
                     f"for the parabola and {sig['path_rms_cm']:.2f} cm for the sigmoid. "
                     "Both improve tracking relative to trial 1 but take longer. All in-lab trajectory trials used "
                     "PID; no in-lab P trajectory comparison is available.")
            fig.text(0.075, 0.188, textwrap.fill(note1, 122), fontsize=7.5, va="top", linespacing=1.45)
            fig.text(0.075, 0.106, textwrap.fill(note2, 122), fontsize=7.5, va="top", linespacing=1.45)
            fig.text(0.075, 0.026, "Group 22 • Section 1 figure insert • 2 / 2", fontsize=7, color=GREY)
            pdf.savefig(fig)
            plt.close(fig)


def write_notes(out: Path, selection: dict[str, str], metrics: dict[str, dict]) -> None:
    p, pid = [metrics[selection[key]] for key in ["point_p", "point_pid"]]
    lines = ["# Lab 2 Section 1 — figure guide and draft captions", "",
             "Reproduce from the repository root: `python3 plot_section1.py`.", "",
             "`section1_report_insert.pdf` contains a two-page Section 1 insert, leaving one body page for Section 2. "
             "Standalone PNG and vector PDF figures are in `figures/`; optional diagnostics are in `supplement/`. "
             "This is a figure insert, not the complete report or a Section 2 submission.", "",
             "## Data and selection", "",
             "Only this group's timestamped logs are used. The other group's PDF was inspected for coverage only; "
             "it is not read by this script and none of its data, captions, gains or numerical results are used.", "",
             "| Role | Source folder | Selection basis |", "|---|---|---|"]
    for key, reason in [("point_p", "Smallest angular overshoot among the Part 5 P trials"),
                        ("point_pid", "Smallest angular overshoot among the Part 5 PID trials"),
                        ("parabola", "Smallest time-weighted RMS geometric path error"),
                        ("sigmoid", "Smallest time-weighted RMS geometric path error")]:
        lines.append(f"| {key} | `{selection[key]}` | {reason} |")
    lines += ["", "These are deliberate tracking/overshoot selections, not universally optimal gains. "
              "The Part 4 P baseline and every Part 5 in-lab trial remain visible in the tuning figure. "
              "The main point comparison does not mix Gazebo initial conditions with in-lab initial conditions.", "",
              "Point-controller P/PID labels follow the folder names. In-lab Part 6 PID use was explicitly "
              "confirmed by the user. Exact in-lab per-trial gains are unavailable: the intermediate in-lab "
              "commits add CSVs without preserving corresponding controller settings. The final working "
              "controller settings are not assigned retrospectively to older trials.", "",
              "## Draft figure captions", "",
              "1. **Point errors.** Recorded linear distance error and wrapped bearing error, with their logged "
              "derivatives, for selected P and PID runs. Dotted references indicate zero error and the 5 cm "
              "position tolerance. Both logs retain all recorded samples; each run shares one pose-based time origin.", "",
              "2. **Point pose.** Recorded x, y and heading versus elapsed time for the same selected runs. "
              "Position targets are x = −1.0 m and y = −0.8 m. The controller does not specify a final heading.", "",
              "3. **Path and phase portraits.** Recorded x–y paths and linear/angular error–error-rate trajectories. "
              "Circles and squares mark first and last samples. The cross and circle indicate the point goal and "
              "its 5 cm acceptance region. Finite recorded trajectories do not establish global stability.", "",
              "4. **Point tuning trials.** Every archived in-lab Part 5 P and PID trial is compared, with the Part 4 "
              "P run retained as a baseline. Bold curves indicate the selected trials. "
              f"Selected {p['label']} has {p['angular_overshoot_deg']:.2f}° peak angular-error overshoot "
              f"and reaches the position tolerance in {p['arrival_5cm_s']:.2f} s; "
              f"{pid['label']} has {pid['angular_overshoot_deg']:.2f}° and {pid['arrival_5cm_s']:.2f} s. "
              "Trial-to-trial differences show tuning outcomes, but missing gain records prevent claims "
              "that an individual gain change caused a particular improvement.", "",
              "5. **Trajectory tracking.** Recorded in-lab PID paths are compared with y = x² on [0, 1.5] m "
              "and y = 2/(1 + exp(−2x)) − 1 = tanh(x) on [0, 2.5] m. Both trials are retained, and the "
              "selected trial is emphasised. Squares identify last recorded poses; crosses identify the planned endpoints.", "",
              "## Trajectory selection rationale (draft)", "",
              "PID was retained for Part 6 because it completed both prescribed shapes within the 5 cm goal "
              "tolerance, with sub-centimetre RMS geometric tracking error in the selected runs. "
              "PID was used for all in-lab Part 6 runs (group confirmation). The second trial is selected for "
              "each trajectory because it reduces RMS and peak geometric tracking error, accepting a slower "
              "completion time. The available in-lab data support a choice between these PID trials; they "
              "do not establish that PID outperforms P for these trajectories.", "",
              "| Shape | Trial | Observed arrival (s) | RMS path error (cm) | Peak path error (cm) |",
              "|---|---|---:|---:|---:|"]
    for kind, names in TRAJECTORIES.items():
        for i, name in enumerate(names):
            m = metrics[name]
            lines.append(f"| {kind} | {i + 1}{' (selected)' if name == selection[kind] else ''} | "
                         f"{m['arrival_5cm_s']:.2f} | {m['path_rms_cm']:.3f} | {m['path_max_cm']:.3f} |")
    lines += ["", "## Interpretation and verification", "",
              "- Timestamp integers are subtracted before conversion to seconds; no first data row is skipped.",
              "- Each run uses the first pose stamp for all three logs. Error logging starts slightly later because "
              "the PID implementation waits for three history samples; missing derivatives are not invented.",
              "- Derivatives are the recorded controller estimates, not a newly differentiated or smoothed signal. "
              "Data are not smoothed, rescaled in time, extended after logging, or normalised to hide differences.",
              "- Point linear error is unsigned distance to the goal. Angular overshoot is the largest error "
              "on the opposite side of zero from the initial angular error, expressed in degrees; it is not "
              "position overshoot or overshoot relative to a fixed commanded heading.",
              "- Arrival is the first recorded pose within 5 cm. Its time resolution is limited by odometry sampling. "
              "Logs stop around that event, so final recorded error is not measured steady-state error.",
              "- Trajectory linear error is distance to the final goal; angular error is relative to the current "
              "look-ahead waypoint. Neither is the perpendicular path-tracking error.",
              "- Path error is the shortest Euclidean distance to the finite continuous analytic curve, "
              "found by dense-grid localisation and scalar refinement. RMS is time weighted: "
              "sqrt(integral(distance² dt)/duration). Endpoint distances are included.",
              "- Desired mathematical paths and zero/goal references are shown. No unverified theoretical "
              "time-response model is fitted. Gazebo comparisons are optional and explicitly labelled; "
              "their gains/robot model differ from the in-lab conditions.",
              "- Every processed error timestamp is matched to its pose. Logged distance is checked against "
              "the stated goal, and point angular error is checked against the pose-derived goal bearing. "
              "Source CSV SHA-256 hashes, sample counts and settings provenance are in `manifest.json`.", ""]
    (out / "SECTION1_GUIDE.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "test_results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis" / "lab2_section1")
    args = parser.parse_args()
    runs: dict[str, Run] = {}
    labels = {BASELINE: "P baseline"}
    labels.update({name: f"P trial {i + 1}" for i, name in enumerate(P_NAMES)})
    labels.update({name: f"PID trial {i + 1}" for i, name in enumerate(PID_NAMES)})
    for kind, names in TRAJECTORIES.items():
        labels.update({name: f"{kind.title()} PID trial {i + 1}" for i, name in enumerate(names)})
    labels.update({name: f"Gazebo {kind} P" for kind, name in SIMULATION.items()})
    for folder in sorted(args.data_dir.iterdir()):
        if folder.is_dir() and (folder / "robot_pose.csv").exists():
            runs[folder.name] = load_run(folder, labels.get(folder.name, folder.name))
    metrics = {}
    for name, run in runs.items():
        kind = next((k for k, names in TRAJECTORIES.items() if name in names or name == SIMULATION[k]), None)
        metrics[name] = audit_run(run, kind)
    selection = {"point_p": min(P_NAMES, key=lambda n: runs[n].overshoot_deg),
                 "point_pid": min(PID_NAMES, key=lambda n: runs[n].overshoot_deg)}
    selection.update({kind: min(names, key=lambda n: metrics[n]["path_rms_cm"]) for kind, names in TRAJECTORIES.items()})
    out = args.output_dir
    figures, supplement = out / "figures", out / "supplement"
    figures.mkdir(parents=True, exist_ok=True)
    supplement.mkdir(parents=True, exist_ok=True)
    style()
    selected = [runs[selection["point_p"]], runs[selection["point_pid"]]]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    point_errors(axes, selected)
    export(fig, figures / "01_point_errors", "Point control | error responses", "Selected runs minimise recorded angular overshoot within each Part 5 group.")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    point_pose(axes, selected)
    export(fig, figures / "02_point_pose", "Point control | measured pose", "Position targets are specified; a final heading target is not specified.")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
    point_geometry(axes, selected)
    export(fig, figures / "03_point_path_phase", "Point control | path and phase portraits", "Circle = first recorded sample; square = last recorded sample. Cross = goal.")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    tuning(axes, runs, selection)
    export(fig, figures / "04_point_tuning", "Point control | archived tuning trials", "Controller names follow run-folder labels; exact per-trial gains were not archived.")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.7))
    trajectory_paths(axes, runs, selection)
    export(fig, figures / "05_trajectory_paths", "Trajectory control | recorded PID paths and prescribed curves", "All in-lab trajectory trials use PID. Selected trials minimise RMS geometric path error.")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    trajectory_errors(axes, runs, selection)
    export(fig, supplement / "06_trajectory_errors", "Optional | trajectory error logs", "Distance to final goal and look-ahead bearing error are not perpendicular path error.")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.7))
    trajectory_paths(axes, runs, selection, simulation=True)
    export(fig, supplement / "07_gazebo_comparison", "Optional | Gazebo and in-lab trajectory paths", "Gazebo P and in-lab PID use different settings; simulation is a comparison, not the ideal curve.")
    report_insert(out / "section1_report_insert.pdf", runs, selection, metrics)
    with (out / "run_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(metrics.values()))), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics.values())
    sources = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in args.data_dir.iterdir() if folder.name in runs for p in sorted(folder.glob("*.csv"))}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    manifest = {"source_commit": commit.stdout.strip() if commit.returncode == 0 else None,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "selected_runs": selection, "source_csv_sha256": sources,
                "point_controller_provenance": "Controller labels follow archived folder names; exact per-trial gains unverified.",
                "part6_controller_provenance": "User explicitly confirmed: part 6 was only PID controller (in-lab runs).",
                "selection_rule": {"point": "Minimum peak angular-error overshoot within each Part 5 controller group.",
                                   "trajectory": "Minimum time-weighted RMS shortest distance to the prescribed finite curve."},
                "preprocessing": "Keep every CSV row. Preserve integer ns timestamps. One first-pose origin per run. No smoothing.",
                "run_audit": metrics}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_notes(out, selection, metrics)
    print(json.dumps({"output": str(out), "selected": selection, "audited_runs": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
