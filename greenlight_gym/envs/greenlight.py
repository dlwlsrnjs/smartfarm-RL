from typing import Optional, Tuple, Any, List, Dict
import os
import yaml

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Box

from greenlight_gym.envs.cython.greenlight_cy import GreenLight as GL
from greenlight_gym.common.utils import loadWeatherData
from greenlight_gym.envs.observations import (
    ModelObservations,
    WeatherObservations,
    AggregatedObservations,
    StateObservations,
)
from greenlight_gym.envs.rewards import (
    AdditiveReward,
    HarvestHeatCO2Reward,
    ArcTanPenaltyReward,
    MultiplicativeReward,
)

from datetime import date

REWARDS = {
    "AdditiveReward": AdditiveReward,
    "MultiplicativeReward": MultiplicativeReward,
    "HarvestHeatCO2Reward": HarvestHeatCO2Reward,
    "ArcTanPenaltyReward": ArcTanPenaltyReward,
}


class GreenLightEnv(gym.Env):
    """
    This class represents the Gymnasium Env wrapper for the GreenLight model.
    It can be used by RL algorithms to train agents to control the greenhouse climate.
    It is a subclass of the gym.Env class.
    This Base class is used to define the environment settings and the interface for the GreenLight model.

    Args:
        weather_data_dir: path to the weather data
        location: location of the recorded weather data
        data_source: source of the weather data
        h: [s] time step for the RK4 solver
        nx: number of states
        nu: number of control inputs
        nd: number of disturbances
        no_lamps: whether lamps are used
        led_lamps: whether led lamps are used
        hps_lamps: whether hps lamps are used
        int_lamps: whether interlighting lamps are used
        dmfm: dry matter fruit mass
        season_length: [days] length of the growing season
        pred_horizon: [days] number of future weather predictions
        time_interval: [s] time interval in between observations
        start_train_year: start year for training
        end_train_year: end year for training
        start_train_day: start day for training
        end_train_day: end day for training
        reward_function: reward function to use
        training: whether we are training or testing
        train_days: days to train on
    """

    def __init__(
        self,
        weather_data_dir: str,  # path to weather data
        location: str,  # location of the recorded weather data
        data_source: str,  # source of the weather data
        h: float,  # [s] time step for the RK4 solver
        nx: int,  # number of states
        nu: int,  # number of control inputs
        nd: int,  # number of disturbances
        no_lamps: int,  # whether lamps are used
        led_lamps: int,  # whether led lamps are used
        hps_lamps: int,  # whether hps lamps are used
        int_lamps: int,  # whether interlighting lamps are used
        dmfm: float,  # dry matter to fresh matter ratio for the fruit
        season_length: int,  # [days] length of the growing season
        pred_horizon: int,  # [days] number of future weather predictions
        time_interval: int,  # [s] time interval in between observations
        start_train_year: int = 2011,  # start year for training
        end_train_year: int = 2020,  # end year for training
        start_train_day: int = 59,  # end year for training
        end_train_day: int = 244,  # end year for training
        reward_function: str = "None",  # reward function to use
        training: bool = True,  # whether we are training or testing
        train_days: Optional[List[int]] = None,  # days to train on
    ) -> None:
        super(GreenLightEnv, self).__init__()

        # number of seconds in the day
        self.c = 86400

        # arguments that are kept the same over various simulations
        self.nx = nx
        self.nu = nu
        self.nd = nd
        self.no_lamps = no_lamps
        self.led_lamps = led_lamps
        self.hps_lamps = hps_lamps
        self.int_lamps = int_lamps
        self.weather_data_dir = weather_data_dir
        self.location = location
        self.data_source = data_source
        self.h = h
        self.dmfm = dmfm
        self.season_length = season_length
        self.pred_horizon = pred_horizon
        self.time_interval = time_interval
        self.N = int(
            season_length * self.c / time_interval
        )  # number of timesteps to take for python wrapper
        self.solver_steps = int(
            time_interval / self.h
        )  # number of steps the solver takes between time_interval
        self.reward_function = reward_function

        self.train_years = list(range(start_train_year, end_train_year + 1))

        if train_days is None:
            self.train_days = list(range(start_train_day, end_train_day + 1))
        else:
            self.train_days = train_days

        self.training = training
        self.eval_idx = 0

        self.observations = None
        self.rewards = None

        self.observation_space = None
        self.action_space = None

        # which actuators the GreenLight model can control (3동 정책 노출도 기준)
        # 주의: 내부 GLModel u-인덱스는 변경되지 않습니다.
        # - uFCU -> u[0] (모델 난방에 매핑; FCU on/off 프록시)
        # - uCO2 -> u[1]
        # - uThScr -> u[2]
        # - uVent -> u[3]
        # - uShade -> u[4] (모델상 램프 채널에 매핑; 물리 효과는 별도 처리 필요)
        # - uCircFans -> u[5] (모델상 인터라이트 채널에 매핑; 물리 효과는 별도 처리 필요)
        # 레일파이프(u[6]), 블랙아웃(u[7])은 외부 제어에서 제외
        self.control_indices = {
            "uBoil": 0,
            "uCO2": 1,
            "uThScr": 2,
            "uVent": 3,      # Env-aggregated uVent_eff injected here
            "uShade": 4,
            "uCircFans": 5,
            "uGroPipe": 6,
            "uBlScr": 7,
            "uMist": 8,
            "uFcuFan": 9,
            "uFcuPump": 10,
        }

        # lower and upper bounds for air temperature, co2 concentration, humidity
        self.obs_low = None
        self.obs_high = None

        # # initialize the model in cython
        self.GLModel = GL(
            self.h,
            nx,
            nu,
            self.nd,
            no_lamps,
            led_lamps,
            hps_lamps,
            int_lamps,
            self.solver_steps,
        )

        # Optional: load metering config from YAML if available
        self.config = {}
        try:
            cfg_path = os.path.join("config", "greenhouse.yaml")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)
        except Exception:
            self.config = {}

        # Pre-extract metering parameters with safe defaults
        dtwin = self.config.get("digital_twin", {}) if isinstance(self.config, dict) else {}
        hvac = dtwin.get("hvac", {})
        fcu = hvac.get("fcu", {})
        airflow = dtwin.get("airflow", {})
        circ = airflow.get("circulation_fans", {})
        screens = dtwin.get("screens", {})
        top_shade = screens.get("top_shade", {})
        top_energy = screens.get("top_energy", {})

        self._meter = {
            "fcu_fan_W": float(fcu.get("fan_power_W", 0.0)) if fcu else 0.0,
            "fcu_pump_W": float(fcu.get("pump_power_W", 0.0)) if fcu else 0.0,
            "circ_count": int(circ.get("count", 0)) if circ else 0,
            "circ_each_W": float(circ.get("power_each_W", 0.0)) if circ else 0.0,
            "shade_W": float(top_shade.get("power_W", 0.0)) if top_shade else 0.0,
            "energy_W": float(top_energy.get("power_W", 0.0)) if top_energy else 0.0,
        }

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step function that simulates a single timestep using actions given actions.
        The next state is numerically approximated the GreenLight model implemented in Cython.
        The action is scaled to the range of the control inputs.
        Args:
            action (np.ndarray): action that provides the control inputs for the GreenLight model.
        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: observation, reward, truncated, done, info
        """
        # scale the action to the range of the control inputs, which is between [0, 1]
        action = self._scale(action, self.action_space.low, self.action_space.high)

        # compute area-weighted roof vent effective opening and inject into control vector (index uVent)
        try:
            vents_cfg = self.config.get("digital_twin", {}).get("vents", {})
            AL = float(vents_cfg.get("roof_left", {}).get("effective_area_m2", 0.0))
            AR = float(vents_cfg.get("roof_right", {}).get("effective_area_m2", 0.0))
            den = max(AL + AR, 1e-6)
            # decode desired L/R from action scaled to [0,1]
            uVentL = action[self.control_indices["uVent"]] if "uVentL" not in self.control_indices else action[self.control_indices["uVentL"]]
            uVentR = action[self.control_indices["uVent"]] if "uVentR" not in self.control_indices else action[self.control_indices["uVentR"]]
            uVent_eff = (uVentL * AL + uVentR * AR) / den
            # place effective vent into uVent slot
            action[self.control_indices["uVent"]] = np.clip(uVent_eff, 0.0, 1.0)
            # cache for logging
            self._last_uVent_eff = float(uVent_eff)
        except Exception:
            pass

        # simulate single timestep using the GreenLight model and get the observation
        self.GLModel.step(action, self.control_idx)
        obs = self._get_obs()

        # ===== Power & Energy metering =====
        try:
            dt = float(self.config.get("digital_twin", {}).get("meta", {}).get("timestep_s", self.time_interval))
        except Exception:
            dt = float(self.time_interval)

        # derive control outputs used
        try:
            u = self.GLModel.getControlsArray()
        except Exception:
            u = None

        P_FCU = P_CIRC = P_SHADE = P_ENERGY = 0.0
        fcu_on = circ_fans_on = 0.0
        screen_pos = vent_pos = 0.0

        if u is not None:
            # map controls
            uThScr    = float(u[self.control_indices.get("uThScr", 2)])
            uShade    = float(u[self.control_indices.get("uShade", 4)])
            uCircFans = float(u[self.control_indices.get("uCircFans", 5)])
            uMist     = float(u[self.control_indices.get("uMist", 8)])
            uFcuFan   = float(u[self.control_indices.get("uFcuFan", 9)])
            uFcuPump  = float(u[self.control_indices.get("uFcuPump", 10)])

            # FCU electric power (fan/pump separated)
            P_FCU = self._meter["fcu_fan_W"] * (1.0 if uFcuFan > 0.5 else 0.0) + \
                    self._meter["fcu_pump_W"] * (1.0 if uFcuPump > 0.5 else 0.0)

            # Circulation fans power (allow duty)
            P_CIRC = self._meter["circ_count"] * self._meter["circ_each_W"] * max(0.0, min(1.0, uCircFans))

            # Screen drive power (approx: only when moving; fallback to 0)
            # You can refine by detecting movement via previous positions
            P_SHADE = 0.0  # treat motors as impulse; average ignored here
            P_ENERGY = 0.0

            fcu_on = 1.0 if (uFcuFan > 0.5 and uFcuPump > 0.5) else 0.0
            circ_fans_on = 1.0 if uCircFans > 0.5 else (uCircFans if 0.0 < uCircFans < 0.5 else 0.0)
            screen_pos = uThScr
            # compute effective vent position by area weighting if config available
            try:
                vents_cfg = self.config.get("digital_twin", {}).get("vents", {})
                AL = float(vents_cfg.get("roof_left", {}).get("effective_area_m2", 0.0))
                AR = float(vents_cfg.get("roof_right", {}).get("effective_area_m2", 0.0))
                den = max(AL + AR, 1e-6)
                vent_pos = (uVentL*AL + uVentR*AR) / den
            except Exception:
                vent_pos = 0.5*(uVentL + uVentR)

        P_total_W = P_FCU + P_CIRC + P_SHADE + P_ENERGY

        dE_total_kWh = P_total_W * dt / 3_600_000.0
        dE_fcu_kWh   = P_FCU     * dt / 3_600_000.0
        dE_circ_kWh  = P_CIRC    * dt / 3_600_000.0
        dE_scr_kWh   = (P_SHADE + P_ENERGY) * dt / 3_600_000.0

        self.energy_kWh_total = getattr(self, "energy_kWh_total", 0.0) + dE_total_kWh
        self.energy_kWh_fcu   = getattr(self, "energy_kWh_fcu",   0.0) + dE_fcu_kWh
        self.energy_kWh_circ  = getattr(self, "energy_kWh_circ",  0.0) + dE_circ_kWh
        self.energy_kWh_scr   = getattr(self, "energy_kWh_scr",   0.0) + dE_scr_kWh

        # check if the simulation has reached a terminal state and get the reward
        if self._terminalState(obs):
            self.terminated = True
            reward = 0
        else:
            reward = self._reward()

        # additional information to return
        info = self._get_info()
        try:
            info.update({
                "P_FCU_W": P_FCU,
                "P_CIRC_W": P_CIRC,
                "P_SCR_W": P_SHADE + P_ENERGY,
                "P_TOTAL_W": P_total_W,
                "E_TOTAL_kWh": self.energy_kWh_total,
                "E_FCU_kWh": self.energy_kWh_fcu,
                "E_CIRC_kWh": self.energy_kWh_circ,
                "E_SCR_kWh": self.energy_kWh_scr,
                "fcu_on": fcu_on,
                "circ_fans_on": circ_fans_on,
                "screen_pos": screen_pos,
                "vent_pos": vent_pos,
            })
        except Exception:
            pass

        return (obs, reward, self.terminated, False, info)

    def _get_info(self):
        """
        Placeholder function that returns additional information about the simulation step.
        """
        pass

    def _init_rewards(
        self,
        co2_price: Optional[float] = None,
        gas_price: Optional[float] = None,
        tom_price: Optional[float] = None,
        k: Optional[List[float]] = None,
        obs_low: Optional[List[float]] = None,
        obs_high: Optional[List[float]] = None,
    ) -> None:
        """
        Placeholder function that initialises the reward function.

        Args:
            co2_price (Optional[float], optional): CO2 price. Defaults to None.
            gas_price (Optional[float], optional): Gas price. Defaults to None.
            tom_price (Optional[float], optional): Tomato price kg/€. Defaults to None.
            k (Optional[List[float]], optional): Penalty weights. Defaults to None.
            obs_low (Optional[List[float]], optional): Lower bound of observation space. Defaults to None.
            obs_high (Optional[List[float]], optional): Upper bound observation space. Defaults to None.
        """
        pass

    def _init_observations(
        self,
        model_obs_vars: Optional[List[str]] = None,
        weather_obs_vars: Optional[List[str]] = None,
        Np: Optional[int] = None,
    ) -> None:
        """Function that initialises the observation object using observation modules.

        Args:
            model_obs_vars (Optional[List[str]], optional): List with observations from the greenlight model. Defaults to None.
            weather_obs_vars (Optional[List[str]], optional): List with observations from the weather data. Defaults to None.
            Np (Optional[int], optional): Number of future weather data points to include in the observation space. Defaults to None.
        """
        obs_list = []
        if model_obs_vars is not None:
            obs_list.append(ModelObservations(model_obs_vars))
        if weather_obs_vars is not None:
            obs_list.append(WeatherObservations(weather_obs_vars, Np))
        self.observations = AggregatedObservations(obs_list, model_obs_idx=0)

    def _generate_observation_space(self) -> None:
        """
        Creates the observations space for the environment.
        """
        self.observation_space = Box(
            low=self.observations.low,
            high=self.observations.high,
            shape=(self.observations.Nobs,),
            dtype=np.float32,
        )

    def _get_obs(self):
        """
        Placeholder function that returns the observation.
        """
        pass

    def _terminalState(self, obs: np.ndarray) -> bool:
        """
        Function that checks whether the simulation has reached a terminal state.
        This is the case if we have reached the end of the set growing season.
        Which is determined by the length of the season.
        Also in terminal state for nan or inf in the state values.

        Args:
            obs (np.ndarray): observation

        Returns:
            bool: True if the simulation has reached a terminal state; False otherwise.
        """
        # Function that checks whether the simulation has reached a terminal state.
        # Terminal obs are reached when the simulation has reached the end of the growing season.
        # Or when there are nan or inf in the state values.

        if self.GLModel.timestep >= self.N:
            return True
        # check for nan and inf in observation values
        elif np.isnan(obs).any() or np.isinf(obs).any():
            print("Nan or inf in states")
            return True
        return False

    def _get_time(self) -> float:
        """Returns the time in days since 01-01-0001 of the simulation.

        Returns:
            float: time in days since 01-01-0001.
        """
        return self.GLModel.time

    def _get_time_in_days(self) -> float:
        """
        Get time in days since 01-01-0001 upto the starting day of the simulation.
        """
        d0 = date(1, 1, 1)
        d1 = date(self.growth_year, 1, 1)
        delta = d1 - d0
        return delta.days + self.start_day

    def _scale(self, action: np.ndarray, amin: np.ndarray, amax: np.ndarray) -> float:
        """
        Scale the action between [0,1].
        Based on input action, and its min and max values.

        Args:
            action (_type_): action to scale
            amin (_type_): minimal value of the action
            amax (_type_): maximal value of the action

        Returns:
            _type_: scaled action [0,1]
        """
        return (action - amin) / (amax - amin)

    def _reset_eval_idx(self):
        """
        Reset the evaluation index for picking the start day to 0.
        """
        self.eval_idx = 0

    def increase_eval_idx(self):
        """
        Increase the evaluation index by 1.
        """
        self.eval_idx += 1

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Container function that resets the environment to its initial state.
        """
        super().reset(seed=seed)

        # pick a random growth year and start day if we are training
        # otherwise use one of the start days for evaluation
        if self.training:
            self.growth_year = self.np_random.choice(self.train_years)
            self.start_day = self.np_random.choice(self.train_days)
        else:
            self.start_day = self.start_days[self.eval_idx]
            self.increase_eval_idx()

        # load in weather data for specific simulation
        self.weatherData = loadWeatherData(
            self.weather_data_dir,
            self.location,
            self.data_source,
            self.growth_year,
            self.start_day,
            self.season_length,
            self.pred_horizon,
            self.h,
            self.nd,
        )

        # compute days since 01-01-0001
        # as time indicator by the model
        timeInDays = self._get_time_in_days()

        # reset the GreenLight model starting settings
        self.GLModel.reset(self.weatherData, timeInDays)
        self.terminated = False
        # init energy accumulators
        self.energy_kWh_total = 0.0
        self.energy_kWh_fcu = 0.0
        self.energy_kWh_circ = 0.0
        self.energy_kWh_scr = 0.0
        return self._get_obs(), {}


class GreenLightHeatCO2(GreenLightEnv):
    """
    Child class of GreenLightEnv.
    Starts with a fully mature crop that is ready for harvest.

    Controls the greenhouse climate through four actuators:
    1) heating (uBoil)
    2) carbon dioxide supply (uCo2)
    3) ventialation (uVent)
    4) thermal screen (uThScr)

    Uses a reward that reflects the revenue made from harvesting tomatoes, and the costs for
    heating the greenhouse and injecting CO2 given their resource price.
    Also penalises the reward function violating indoor climate boundaries.

    Args:
        cLeaf (float): initial DW for leaves mg/m2
        cStem (float): initial DW for stems mg/m2
        cFruit (float): initial DW for fruit mg/m2
        tCanSum (float): initial sum of canopy temperature (development stage)
        co2_price (float): price of CO2
        gas_price (float): price of gas
        tom_price (float): price of tomatoes
        k (List[float]): penalty weights
        obs_low (List[float]): lower bounds of the observation space
        obs_high (List[float]): upper bounds of the observation space
        control_signals (Optional[List[str]]): list of control signals
        model_obs_vars (Optional[List[str]]): list of model observation variables
        weather_obs_vars (Optional[List[str]]): list of weather observation variables
        omega (float): weight for the multiplicative reward function
        **kwargs: additional keyword arguments for GreenLightEnv
    """

    def __init__(
        self,
        cLeaf: float = 0.9e5,  # [DW] mg/m2
        cStem: float = 2.5e5,  # [DW] mg/m2
        cFruit: float = 2.8e5,  # [DW] mg/m2
        tCanSum: float = 3e3,
        co2_price: float = 0.1,
        gas_price: float = 0.26,
        tom_price: float = 1.6,
        k: List[float] = [1, 1, 1],
        obs_low: List[float] = [0, 0, 0],
        obs_high: List[float] = [np.inf, np.inf, np.inf],
        control_signals: Optional[List[str]] = None,
        model_obs_vars: Optional[List[str]] = None,
        weather_obs_vars: Optional[List[str]] = None,
        omega: float = 1.0,
        **kwargs,
    ) -> None:
        super(GreenLightHeatCO2, self).__init__(**kwargs)
        self.cLeaf = cLeaf
        self.cStem = cStem
        self.cFruit = cFruit
        self.tCanSum = tCanSum
        self.co2Price = co2_price
        self.gasPrice = gas_price
        self.tomatoPrice = tom_price
        self.k = np.array(k)
        self.control_signals = control_signals
        self.model_obs_vars = model_obs_vars
        self.weather_obs_vars = weather_obs_vars

        # the prediction horizon of weather variables for our observations
        Np = int(self.pred_horizon * self.c / self.time_interval)

        # intialise observation and reward functions
        self._init_observations(model_obs_vars, weather_obs_vars, Np)
        self._init_rewards(co2_price, gas_price, tom_price, k, obs_low, obs_high, omega)

        # initialise the observation and action spaces
        self._generate_observation_space()

        # action space is a Box with low and high values for each control signal
        self.action_space = Box(
            low=-1, high=1, shape=(len(control_signals),), dtype=np.float32
        )
        self.control_idx = np.array(
            [self.control_indices[control_input] for control_input in control_signals],
            dtype=np.uint8,
        )

    def _get_info(self):
        """
        Information to return after each timestep.

        Returns:
            _type_: information to return
        """
        return {
            "controls": self.GLModel.getControlsArray(),
            "Time": self.GLModel.time,
            "profit": self.rewards.rewards_list[0].profit,
            "violations": self.rewards.rewards_list[1].abs_pen,
            "timestep": self.GLModel.timestep,
        }

    def _get_obs(self) -> np.ndarray:
        """
        Function that returns the observation.
        Given the observation space modules initialised in _init_observations.

        Returns:
            np.ndarray: observation
        """
        return self.observations.compute_obs(
            self.GLModel, self.solver_steps, self.weatherData
        )

    def _init_rewards(
        self,
        co2_price: float,
        gas_price: float,
        tom_price: float,
        k: List[float],
        obs_low: List[float],
        obs_high: List[float],
        omega: float = 0.3,
    ) -> None:
        harvest_reward = HarvestHeatCO2Reward(
            co2_price,
            gas_price,
            tom_price,
            self.dmfm,
            self.time_interval,
            self.GLModel.maxco2rate,
            self.GLModel.maxHeatCap,
            self.GLModel.maxHarvest,
            self.GLModel.energyContentGas,
        )
        penalty_reward = ArcTanPenaltyReward(k, obs_low, obs_high)
        self.rewards = REWARDS[self.reward_function](
            rewards_list=[harvest_reward, penalty_reward], omega=omega
        )
        # Optional: set electricity price (€/kWh). Typical smart greenhouse tariff ~0.15–0.2 €/kWh.
        try:
            elec_price = float(self.config.get("digital_twin", {}).get("tariffs", {}).get("electricity_eur_per_kwh", 0.18))
            # rewards_list[0] is HarvestHeatCO2Reward
            self.rewards.rewards_list[0].set_electricity_price(elec_price)
        except Exception:
            pass

    def _reward(self) -> float:
        """
        Get the reward from the reward module.

        Returns:
            float: reward
        """
        return self.rewards._compute_reward(self.GLModel)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        # set crop state to start of the season
        self.GLModel.setCropState(self.cLeaf, self.cStem, self.cFruit, self.tCanSum)
        return self._get_obs(), {}


class GreenLightRuleBased(GreenLightEnv):
    """
    Child class of GreenLightEnv.
    Starts with a fully mature crop that is ready for harvest.

    Controls the greenhouse climate through four actuators:
    - carbon dioxide supply
    - heating
    - ventialation
    - thermal screen

    Uses a reward that reflects the revenue made from harvesting tomatoes, and the costs for
    heating the greenhouse and injecting CO2 given their resource price.
    Also penalises violating indoor climate boundaries.
    """

    def __init__(
        self,
        cLeaf: float = 0.9e5,
        cFruit: float = 2.8e5,
        cStem: float = 2.5e5,
        tCanSum: float = 1035,
        co2_price: float = 0.1,
        gas_price: float = 0.35,
        tom_price: float = 1.6,
        k: List[float] = [1, 1, 1],
        obs_low: List[float] = [0, 0, 0],
        obs_high: List[float] = [np.inf, np.inf, np.inf],
        control_signals: List[str] = [],
        model_obs_vars: Optional[List[str]] = None,
        weather_obs_vars: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super(GreenLightRuleBased, self).__init__(**kwargs)
        self.cLeaf = cLeaf
        self.cStem = cStem
        self.cFruit = cFruit
        self.tCanSum = tCanSum

        self.control_signals = control_signals
        self.model_obs_vars = model_obs_vars
        self.weather_obs_vars = weather_obs_vars

        Np = int(
            self.pred_horizon * self.c / self.time_interval
        )  # the prediction horizon in timesteps for our weather predictions

        # intialise observation and reward functions
        self._init_observations(model_obs_vars, weather_obs_vars, Np)
        self._init_rewards(co2_price, gas_price, tom_price, k, obs_low, obs_high)

        # initialise the observation and action spaces
        self._generate_observation_space()
        self.action_space = Box(
            low=-1, high=1, shape=(len(control_signals),), dtype=np.float32
        )
        self.control_idx = np.array(
            [self.control_indices[control_input] for control_input in control_signals],
            dtype=np.uint8,
        )

    def _get_obs(self) -> np.ndarray:
        return self.observations.compute_obs(
            self.GLModel, self.solver_steps, self.weatherData
        )

    def _init_rewards(
        self,
        co2_price: float,
        gas_price: float,
        tom_price: float,
        k: List[float],
        obs_low: List[float],
        obs_high: List[float],
    ) -> None:
        self.rewards = AdditiveReward(
            [
                HarvestHeatCO2Reward(
                    co2_price,
                    gas_price,
                    tom_price,
                    self.dmfm,
                    self.time_interval,
                    self.GLModel.maxco2rate,
                    self.GLModel.maxHeatCap,
                    self.GLModel.maxHarvest,
                    self.GLModel.energyContentGas,
                ),
                ArcTanPenaltyReward(k, obs_low, obs_high),
            ]
        )
        # Optional electricity price
        try:
            elec_price = float(self.config.get("digital_twin", {}).get("tariffs", {}).get("electricity_eur_per_kwh", 0.18))
            self.rewards.rewards_list[0].set_electricity_price(elec_price)
        except Exception:
            pass

    def _reward(self) -> float:
        return self.rewards._compute_reward(self.GLModel)

    def _get_info(self):
        return {
            "controls": self.GLModel.getControlsArray(),
            "Time": self.GLModel.time,
            "profit": self.rewards.rewards_list[0].profit,
            "violations": self.rewards.rewards_list[1].abs_pen,
            "timestep": self.GLModel.timestep,
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.GLModel.setCropState(self.cLeaf, self.cStem, self.cFruit, self.tCanSum)
        return self._get_obs(), {}


class GreenLightStatesTest(GreenLightEnv):
    """
    Child class of GreenLightEnv.
    Starts with a fully mature crop that is ready for harvest.
    The start dates are early year, (January and Februari), which reflects the start of the harvest season.

    Controls the greenhouse climate through four actuators:
    - carbon dioxide supply
    - heating
    - ventialation
    - thermal screen

    Uses a reward that reflects the revenue made from harvesting tomatoes, and the costs for
    heating the greenhouse and injecting CO2 given their resource price.
    Also penalises violating indoor climate boundaries.
    """

    def __init__(
        self,
        cLeaf: float = 2.5e5,
        cStem: float = 0.9e5,
        cFruit: float = 2.8e5,
        tCanSum: float = 1035,
        obs_low: List[float] = [0, 0, 0],
        obs_high: List[float] = [np.inf, np.inf, np.inf],
        control_signals: Optional[List[str]] = None,
        model_obs_vars: Optional[List[str]] = None,
        weather_obs_vars: Optional[List[str]] = None,
        weather: np.ndarray = None,
        **kwargs,
    ) -> None:
        super(GreenLightStatesTest, self).__init__(**kwargs)
        self.cLeaf = cLeaf
        self.cStem = cStem
        self.cFruit = cFruit
        self.tCanSum = tCanSum

        self.control_signals = control_signals
        self.model_obs_vars = model_obs_vars
        self.weather_obs_vars = weather_obs_vars

        Np = int(
            self.pred_horizon * self.c / self.time_interval
        )  # the prediction horizon in timesteps for our weather predictions

        # intialise observation and reward functions
        self._init_observations(model_obs_vars, weather_obs_vars, Np)

        # initialise the observation and action spaces
        self._generate_observation_space()
        self.action_space = Box(
            low=0, high=1, shape=(len(control_signals),), dtype=np.float32
        )
        self.control_idx = np.array(
            [self.control_indices[control_input] for control_input in control_signals],
            dtype=np.uint8,
        )

        self.weatherData = weather

    def _get_obs(self) -> np.ndarray:
        return self.GLModel.getStatesArray()

    def _reward(self):
        return 1

    def _get_info(self):
        return {
            "controls": self.GLModel.getControlsArray(),
            "Time": self.GLModel.time,
        }

    def update_h(self, h: float):
        self.GLModel.update_h(h)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Container function that resets the environment.
        """
        self.start_day = self.start_days[self.eval_idx]
        self.increase_eval_idx()

        # compute days since 01-01-0001
        # as time indicator by the model
        timeInDays = self._get_time_in_days()

        # reset the GreenLight model starting settings
        self.GLModel.reset(self.weatherData, timeInDays)
        self.GLModel.setCropState(self.cLeaf, self.cStem, self.cFruit, self.tCanSum)

        self.terminated = False

        return self._get_obs(), {}


if __name__ == "__main__":
    pass
