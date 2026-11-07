"""Verify the archived production sweep and regenerate Figure 1."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from exact_mfpt import exact_mfpt, kappa_commit_asymptotic
from kramers_validation import Y_PLUS, mle_rate_censored
from safeguards import ci95_lognormal, hazard_stabilization_scan


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SUMMARY = DATA / "final_sweep_data_CORRECTED_CI.csv"
SEEDS = DATA / "final_sweep_seeds.json"
ARCHIVED_RAW = DATA / "raw"
BASE_FIELDS = {
    "r", "T_tilde", "escape_times", "escaped", "censoring_time", "n_traj", "seed_base",
}
ARCHIVE_FIELDS = BASE_FIELDS | {"t_star", "events_total", "events_post_truncation"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(label, actual, reference):
    if not np.isclose(actual, float(reference), rtol=1e-10, atol=1e-13):
        raise ValueError(f"{label}: calculated {actual!r}, archived {reference!r}")


def verify(raw_dir):
    with SUMMARY.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    registry = json.loads(SEEDS.read_text(encoding="utf-8"))
    keys = [f"r_{i:02d}" for i in range(16)]
    expected_files = {f"{key}_raw.npz" for key in keys}
    require(len(rows) == 16 and set(registry) == set(keys), "Expected 16 rows and seed records")
    require({path.name for path in raw_dir.glob("*.npz")} == expected_files,
            f"{raw_dir}: unexpected raw file set")
    payloads_seen = set()

    for i, (r, key, row) in enumerate(zip(np.geomspace(0.15, 7.0, 16), keys, rows)):
        seed = 77000 + 10000 * i
        theta = 0.75 / r
        metadata = registry[key]
        close(f"{key} registry r", r, metadata["r"])
        close(f"{key} CSV r", r, row["r"])
        close(f"{key} CSV noise", theta, row["T_tilde"])
        require(seed == metadata["seed_base"] == int(row["seed_base"]), f"{key}: seed mismatch")
        require(6000 == metadata["n_traj"] == int(row["n_traj_total"]), f"{key}: count mismatch")

        with np.load(raw_dir / f"{key}_raw.npz", allow_pickle=False) as raw:
            fields = ARCHIVE_FIELDS if raw_dir == ARCHIVED_RAW else BASE_FIELDS
            require(set(raw.files) == fields, f"{key}: unexpected NPZ fields")
            require(all(raw[field].shape == () for field in fields - {"escape_times", "escaped"}),
                    f"{key}: expected scalar metadata")
            times, escaped = raw["escape_times"], raw["escaped"]
            censoring = raw["censoring_time"].item()
            require(raw["seed_base"].item() == seed and raw["n_traj"].item() == 6000,
                    f"{key}: raw seed or count mismatch")
            close(f"{key} raw r", r, raw["r"].item())
            close(f"{key} raw noise", theta, raw["T_tilde"].item())
            require(censoring == 150.0 and times.shape == escaped.shape == (6000,),
                    f"{key}: unexpected trajectory dimensions or censoring horizon")
            require(times.dtype.kind == "f" and escaped.dtype == np.dtype(bool),
                    f"{key}: unexpected trajectory data types")
            require(np.all(np.isfinite(times[escaped]) & (times[escaped] > 0)
                           & (times[escaped] <= censoring)) and np.all(np.isnan(times[~escaped])),
                    f"{key}: invalid event times or censoring flags")
            digest = hashlib.sha256(times.tobytes() + escaped.tobytes()).digest()
            require(digest not in payloads_seen, f"{key}: duplicate trajectory payload")
            payloads_seen.add(digest)

            naive, _, events = mle_rate_censored(times, escaped, censoring)
            t_star, sim, scan = hazard_stabilization_scan(times, escaped, censoring)
            events_post = next(item["n_events"] for item in scan if item["t_star"] == t_star)
            observed = np.where(escaped, times, censoring)
            at_risk = observed >= t_star
            exposure = (observed[at_risk] - t_star).sum()
            low, high = ci95_lognormal(sim, events_post)
            exact = 1.0 / exact_mfpt(-Y_PLUS, b=Y_PLUS, T_tilde=theta)
            kramers = kappa_commit_asymptotic(r)
            if raw_dir == ARCHIVED_RAW:
                require(raw["events_total"].item() == events and
                        raw["events_post_truncation"].item() == events_post and
                        raw["t_star"].item() == t_star, f"{key}: raw event or truncation mismatch")

        for field, expected in {
            "n_events": events, "events_total": events, "events_post_truncation": events_post,
            "n_at_risk_post_truncation": at_risk.sum(), "t_star": t_star, "t_star_stabilized": t_star,
        }.items():
            require(expected == float(row[field]), f"{key}: {field} mismatch")
        for field, expected in {
            "total_time_at_risk_post": exposure,
            "kappa_sim_naive": naive, "kappa_sim_stabilized": sim, "kappa_stabilized": sim,
            "ci_low_corrected": low, "ci_high_corrected": high,
            "kappa_sim_ci_lo": low, "kappa_sim_ci_hi": high,
            "kappa_exact_mfpt": exact, "kappa_asymptotic": kramers,
            "rel_err_sim_vs_exact": abs(sim - exact) / exact,
            "rel_err_sim_vs_asymptotic": abs(sim - kramers) / kramers,
            "log_ratio_sim_vs_exact": np.log(sim / exact),
            "log_ratio_sim_vs_asymptotic": np.log(sim / kramers),
            "log_ratio_exact_vs_asymptotic": np.log(exact / kramers),
            "transient_bias_pct": 100 * abs(sim - naive) / naive,
        }.items():
            close(f"{key} {field}", expected, row[field])
        if i == 15:
            print(f"r=7: events={events}, post={events_post}, t*={t_star:g}, "
                  f"kappa_sim={sim:.12g}, CI=[{low:.12g}, {high:.12g}], "
                  f"kappa_exact={exact:.12g}, kappa_K={kramers:.12g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ARCHIVED_RAW,
                        help="Check fresh raw files from the optional full simulation")
    args = parser.parse_args()
    try:
        verify(args.raw_dir.resolve())
        subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_final_validation.py")],
                       check=True)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("PASS: 16 raw datasets, 96,000 trajectories, corrected intervals, references, and Figure 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
