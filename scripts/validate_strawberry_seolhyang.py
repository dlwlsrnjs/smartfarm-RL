import numpy as np
import math

from greenlight_gym.envs.cython.greenlight_cy import GreenLight as GL


# --- Simulator base settings (match env) ---
H = 1.0                    # [s] ODE solver step
TIME_INTERVAL = 300        # [s] observation/control interval
SOLVER_STEPS = int(TIME_INTERVAL / H)
NX = 28
NU = 11
ND = 10
NO_LAMPS, LED, HPS, INT = 0, 1, 0, 0


# --- Constants ---
ETA_MG_PER_M3_PER_PPM = 0.554   # p.etaMgPpm
MOLAR_MASS_CH2O_MG_PER_MMOL = 30_000.0  # 30 g/mol = 30,000 mg/mmol


def ppm_to_mg_per_m3(ppm: float) -> float:
    return ppm * ETA_MG_PER_M3_PER_PPM


def esat_pa(temp_C: float) -> float:
    # Tetens [Pa]
    return 610.78 * math.exp((17.2694 * temp_C) / (temp_C + 237.29))


def make_weather_array(n_steps: int,
                       glob_rad_Wm2: float,
                       out_temp_C: float,
                       out_rh_pct: float,
                       out_co2_ppm: float,
                       wind_ms: float = 1.0,
                       sky_temp_C: float = None,
                       soil_out_C: float = None) -> np.ndarray:
    d = np.zeros((n_steps * SOLVER_STEPS, ND), dtype=np.float64)
    vp_out = esat_pa(out_temp_C) * (out_rh_pct / 100.0)   # Pa
    co2_out_mg_m3 = ppm_to_mg_per_m3(out_co2_ppm)
    if sky_temp_C is None:
        sky_temp_C = out_temp_C - 10.0
    if soil_out_C is None:
        soil_out_C = out_temp_C - 2.0
    d[:, 0] = glob_rad_Wm2   # Global radiation [W m^-2]
    d[:, 1] = out_temp_C     # Outdoor Temperature [C]
    d[:, 2] = vp_out         # Outdoor vapor pressure [Pa]
    d[:, 3] = co2_out_mg_m3  # Outdoor CO2 [mg m^-3]
    d[:, 4] = wind_ms        # Wind [m s^-1]
    d[:, 5] = sky_temp_C     # Sky temp [C]
    d[:, 6] = soil_out_C     # Outdoor soil temp [C]
    return d


def run_block(gl: GL,
              glob_rad_Wm2: float,
              out_temp_C: float,
              out_co2_ppm: float,
              out_rh_pct: float = 60.0,
              duration_steps: int = 48,   # 48*300s = 4h
              learned_all_zero: bool = True):
    weather = make_weather_array(duration_steps, glob_rad_Wm2, out_temp_C, out_rh_pct, out_co2_ppm)
    gl.reset(weather, 0)

    # Seolhyang initial crop state
    gl.setCropState(cLeaf=1.2e5, cStem=0.6e5, cFruit=0.4e5, tCanSum=800.0)

    if learned_all_zero:
        learned_idx = np.arange(NU, dtype=np.uint8)
        controls = np.zeros(NU, dtype=np.float32)
    else:
        learned_idx = np.array([], dtype=np.uint8)
        controls = np.array([], dtype=np.float32)

    x_prev = gl.getStatesArray()
    total_ch2o_prev_mg = (x_prev[22] + x_prev[23] + x_prev[24] + x_prev[25])
    fruit_prev_mg = x_prev[25]

    harvest_kg_sum = 0.0
    for _ in range(duration_steps):
        gl.step(controls, learned_idx)
        harvest_kg_sum += gl.fruit_harvest  # [kg CH2O m^-2 per ts]

    x_now = gl.getStatesArray()
    total_ch2o_now_mg = (x_now[22] + x_now[23] + x_now[24] + x_now[25])
    fruit_now_mg = x_now[25]

    delta_total_mg = total_ch2o_now_mg - total_ch2o_prev_mg
    harvest_mg = harvest_kg_sum * 1e6

    # Apparent assimilation (approx.)
    delta_mg_per_ts = (delta_total_mg + harvest_mg)
    duration_s = duration_steps * TIME_INTERVAL
    mg_per_s = delta_mg_per_ts / duration_s
    umol_per_s = (mg_per_s / MOLAR_MASS_CH2O_MG_PER_MMOL) * 1e3

    fruit_gain_mg = fruit_now_mg - fruit_prev_mg
    fruit_gain_mg_per_h = (fruit_gain_mg / duration_s) * 3600.0

    return {
        "PAR_Wm2": GL.PAR.__get__(gl),
        "A_app_umol_m2_s": umol_per_s,
        "fruit_gain_mg_m2_per_h": fruit_gain_mg_per_h,
        "fruit_weight_kg_m2": GL.fruit_weight.__get__(gl),
    }


def build_model():
    gl = GL(H, NX, NU, ND, NO_LAMPS, LED, HPS, INT, SOLVER_STEPS)
    return gl


def sweep_and_print():
    gl = build_model()

    print("\n=== 1) Irradiance sweep @ T=22C, CO2=900 ppm ===")
    for I in [0, 100, 200, 400, 600, 800, 1000]:
        res = run_block(gl, glob_rad_Wm2=I, out_temp_C=22.0, out_co2_ppm=900.0)
        print(f"I={I:4.0f} | A_app={res['A_app_umol_m2_s']:6.1f} umol/m2/s | fruit+{res['fruit_gain_mg_m2_per_h']:7.1f} mg/m2/h | PAR~{res['PAR_Wm2']:.1f}")

    print("\n=== 2) Temperature sweep @ I=800 W/m2, CO2=900 ppm ===")
    for T in [12, 16, 20, 22, 24, 26, 28, 30, 32]:
        res = run_block(gl, glob_rad_Wm2=800.0, out_temp_C=float(T), out_co2_ppm=900.0)
        print(f"T={T:2.0f}C | A_app={res['A_app_umol_m2_s']:6.1f} umol/m2/s | fruit+{res['fruit_gain_mg_m2_per_h']:7.1f} mg/m2/h")

    print("\n=== 3) CO2 sweep @ I=800 W/m2, T=22C ===")
    for co2 in [400, 600, 800, 900, 1000, 1200]:
        res = run_block(gl, glob_rad_Wm2=800.0, out_temp_C=22.0, out_co2_ppm=float(co2))
        print(f"CO2={co2:4.0f} | A_app={res['A_app_umol_m2_s']:6.1f} umol/m2/s | fruit+{res['fruit_gain_mg_m2_per_h']:7.1f} mg/m2/h")


if __name__ == "__main__":
    sweep_and_print()


