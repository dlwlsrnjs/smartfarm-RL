import argparse
import pandas as pd
import numpy as np
import os


def load_csv(path):
	return pd.read_csv(path)


def summary_stats(df):
	return {
		"rows": len(df),
		"time_start": float(df["time"].min()) if "time" in df else None,
		"time_end": float(df["time"].max()) if "time" in df else None,
		"rad_mean": float(df["global radiation"].mean()),
		"temp_mean": float(df["air temperature"].mean()),
		"wind_mean": float(df["wind speed"].mean()),
		"rh_mean": float(df["RH"].mean()),
	}


def main(local_path, ref_path):
	local = load_csv(local_path)
	ref = load_csv(ref_path)

	print("Local summary:", summary_stats(local))
	print("Amsterdam summary:", summary_stats(ref))

	common_cols = [
		"global radiation",
		"air temperature",
		"wind speed",
		"RH",
	]
	print("\nMeans (local - ref):")
	for c in common_cols:
		print(c, float(local[c].mean() - ref[c].mean()))

	print("\nStd dev ratio (local/ref):")
	for c in common_cols:
		ref_std = ref[c].std()
		print(c, float(local[c].std() / ref_std) if ref_std != 0 else np.nan)


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--local", required=True)
	ap.add_argument("--ref", required=True)
	args = ap.parse_args()
	main(args.local, args.ref)
