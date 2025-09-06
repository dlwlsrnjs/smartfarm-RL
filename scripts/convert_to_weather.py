import os
import argparse
import pandas as pd
import numpy as np


def compute_sky_temp(air_temp_c, cloud_frac=0.5):
	"""
	Compute sky temperature from air temperature and assumed cloud cover fraction.
	"""
	sigma = 5.67e-8
	C2K = 273.15
	ld_clear = 213 + 5.5 * air_temp_c
	eps_clear = ld_clear / (sigma * (air_temp_c + C2K) ** 4)
	eps_cloud = (1 - 0.84 * cloud_frac) * eps_clear + 0.84 * cloud_frac
	ld_cloud = eps_cloud * sigma * (air_temp_c + C2K) ** 4
	return (ld_cloud / sigma) ** 0.25 - C2K


def main(in_csv, location, source_prefix, out_root, cloud_frac=0.5, year=None):
	# Input column names in the user's dataset
	ts_col = "저장시간"
	out_temp_col = "외부온도"
	wind_col = "풍속"
	rad_col = "외부일사"
	out_rh_col = "외부습도"
	in_rh_fallback_col = "내부습도(1)"

	df = pd.read_csv(in_csv)
	df["ts"] = pd.to_datetime(df[ts_col])
	if year is None:
		year = int(df["ts"].dt.year.mode()[0])

	# Parse and sanitize columns
	df["air temperature"] = pd.to_numeric(df[out_temp_col], errors="coerce")
	df["wind speed"] = pd.to_numeric(df[wind_col], errors="coerce")
	df["global radiation"] = pd.to_numeric(df[rad_col], errors="coerce").clip(lower=0)
	if out_rh_col in df.columns:
		df["RH"] = pd.to_numeric(df[out_rh_col], errors="coerce")
	else:
		df["RH"] = pd.to_numeric(df[in_rh_fallback_col], errors="coerce")

	df = df.set_index("ts").sort_index()
	df5 = df[["global radiation", "wind speed", "air temperature", "RH"]].resample("5min").mean().interpolate()

	# Derived columns
	df5["sky temperature"] = compute_sky_temp(df5["air temperature"].values, cloud_frac=cloud_frac)
	df5["CO2 concentration"] = 400.0
	df5["??"] = 0.0

	# Time columns
	start_of_year = pd.Timestamp(year=year, month=1, day=1, tz=df5.index.tz)
	df5["time"] = (df5.index - start_of_year).total_seconds()
	df5["day number"] = np.floor(df5["time"] / 86400)

	# Reorder and save
	out_df = df5[[
		"time",
		"global radiation",
		"wind speed",
		"air temperature",
		"sky temperature",
		"??",
		"CO2 concentration",
		"day number",
		"RH",
	]].reset_index(drop=True)

	out_dir = os.path.join(out_root, location)
	os.makedirs(out_dir, exist_ok=True)
	out_path = os.path.join(out_dir, f"{source_prefix}{year}.csv")
	out_df.to_csv(out_path, index=False)
	print(f"Saved: {out_path}")


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--in_csv", required=True)
	ap.add_argument("--location", required=True, help="e.g., MyGreenhouse")
	ap.add_argument("--source_prefix", default="LOCAL", help="filename prefix")
	ap.add_argument("--out_root", default="greenlight_gym/envs/data/")
	ap.add_argument("--cloud_frac", type=float, default=0.5)
	ap.add_argument("--year", type=int, default=None)
	args = ap.parse_args()
	main(**vars(args))
