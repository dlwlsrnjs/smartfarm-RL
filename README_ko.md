# README_ko.md

## 1) 개요

GreenLight-Gym은 Cython으로 고성능 구현된 하우스(온실) 물리 시뮬레이터를 Gymnasium 환경으로 감싼 RL 학습 환경입니다. RL 에이전트가 난방·CO2·환기·보온커튼 등 구동기(액추에이터)를 제어하여 수확 이익과 제약 위반을 동시에 고려한 정책을 학습합니다.

본 문서는 다음을 한국어로 상세 설명합니다.
- 설정(YAML) → 관측/액션 → 보상 → 학습 파이프라인
- 구동기(액추에이터) 프로파일과 내부 제어 로직
- 물리 함수와 Cython 내부 구조(상태/파라미터/자원/최대용량 등)
- 평가/로깅 결과 컬럼
- 우리 온실에 맞춘 커스터마이징(물리함수/구동기/보상/관측/날씨/학습) 방법

---

## 2) 실행과 파이프라인 개요

학습 실행 예시:

```bash
# 가상환경 활성화 (Git Bash)
source gr/Scripts/activate

# 온라인 로깅 (사전 wandb login 필요)
python -m greenlight_gym.experiments.train_eval \
  --env_id GreenLightHeatCO2 \
  --project local_test --group ppo \
  --env_config_name my_local_5min \
  --algorithm ppo-99 \
  --total_timesteps 100000 \
  --n_eval_episodes 1 \
  --num_cpus 1 \
  --n_evals 2
```

전체 흐름:
- 설정 로드: `greenlight_gym/configs/envs/my_local_5min.yml` 및 `greenlight_gym/configs/algorithms/ppo-99.yml`
- 환경 생성: `GreenLightHeatCO2` → Cython 모델 초기화 → 관측/보상 초기화
- 학습 루프: PPO로 스텝 전개, 주기적 평가/로깅, 최적 모델 갱신
- 결과 저장: 평가 결과 CSV와 VecNormalize 저장, W&B 로깅

쉽게 말해: 설정을 읽고 시뮬레이터를 만들고, 에이전트가 5분마다 조작하며 성과(보상)를 기록하고, 중간중간 평가해서 결과를 저장합니다.

---

## 3) 환경 설정과 관측/액션

### 3.1 환경 설정(YAML)

학습에 사용된 환경 설정:

```yaml
GreenLightEnv:
  weather_data_dir: greenlight_gym/envs/data/
  location: MyGreenhouse
  data_source: LOCAL
  nx: 28
  nu: 8
  nd: 10
  no_lamps: 0
  led_lamps: 1
  hps_lamps: 0
  int_lamps: 0
  dmfm: 0.0627
  h: 1
  season_length: 10
  pred_horizon: 0.0
  time_interval: 300
  start_train_year: 2024
  end_train_year: 2024
  start_train_day: 149
  end_train_day: 330
  training: True
  reward_function: MultiplicativeReward

GreenLightHeatCO2:
  cLeaf: 0.9e5
  cStem: 2.5e5
  cFruit: 2.8e5
  tCanSum: 3e3
  co2_price: 0.1
  gas_price: 0.26
  tom_price: 1.6
  k: [1.2, 8.e-3, 1.0]
  obs_low:  [10, 0, 60]
  obs_high: [35, 1000, 90]
  omega: 1.0
  control_signals: [uBoil, uCO2, uThScr, uVent]
  model_obs_vars: [air_temp, co2_conc, in_rh, fruit_weight, fruit_harvest, PAR, daily_avg_temp, co2_resource, gas_resource, hour_of_day_sin, hour_of_day_cos, day_of_year_sin, day_of_year_cos]
  weather_obs_vars: [glob_rad, out_temp, out_rh, out_co2, wind_speed]

options:
  start_days: [149, 160, 180]
  growth_years: [2024]

results_columns: [Time, Air Temperature, CO2 concentration, Humidity, Fruit weight,
  Fruit harvest, PAR, Daily mean crop temperature, CO2 resource, Gas resource, Hour of day sin,
  Hour of day cos, Day of year sin, Day of year cos, Global radiation, Outdoor temperature, 
  Outdoor Humidity, Outdoor CO2, Wind speed, uBoil, uCO2, uThScr, uVent, uLamp, uIntLamp, uGroPipe, 
  uBlScr, Profits, Temperature violation, CO2 violation, Humidity violation, Return, episode]
```

- 시간 해상도: 적분 `h=1 s`, 제어/관측 간격 `time_interval=300 s(=5분)`
- 에피소드 길이: `season_length=10 days` → 한 에피소드 스텝 수 `N = season_length * 86400 / time_interval`
- 평가/로깅 컬럼: 위 `results_columns`에 정의

쉽게 말해: 5분 간격으로 10일을 시뮬레이션하고, 어떤 값을 저장/그래프로 볼지 미리 정해둡니다.

### 3.2 관측(Observations)

- 모델 관측 14개: 실내 온도/CO2/RH, 작물중량/수확, PAR, 일평균온도, 자원(가스/CO2), 시간 특성(sin/cos)
- 날씨 관측 5개: `glob_rad, out_temp, out_rh, out_co2, wind_speed`
- 미래예측(`pred_horizon=0`): 미래 시퀀스 없이 현재 값만 포함

쉽게 말해: 지금 상태(실내+날씨 요약값)만 에이전트에 보여줍니다.

### 3.3 액션(Action)과 스케일링

- 사용 제어신호: `uBoil(난방), uCO2, uThScr(보온커튼), uVent(환기)`
- 액션 범위: 에이전트 출력 `[-1,1]` → 내부에서 `[0,1]`로 스케일
- 스케일 식: \( a_{scaled} = \frac{a - a_{min}}{a_{max}-a_{min}} \)
  - 여기서 `a_min=-1`, `a_max=1` 이므로 \( a_{scaled}=\frac{a+1}{2} \)

쉽게 말해: 에이전트가 내는 -1~1 값을 0~1 비율로 바꿔서 장치를 약~강으로 움직입니다.

---

## 4) 보상 함수(이익+제약)와 수식

현재 선택된 보상: `MultiplicativeReward` (이익 × 페널티 결합)

쉽게 말해: 돈을 많이 벌되, 실내환경(온도/습도/CO2) 조건을 깨면 감점입니다.

### 4.1 수익 기반 보상 HarvestHeatCO2Reward
- 기본 이익(한 스텝): \( \Pi = \text{harvest}/\mathrm{dmfm} \cdot \text{tom\_price} - \text{CO2\_resource} \cdot \text{co2\_price} - \text{gas\_resource}\cdot \text{gas\_price} \)
- 자원량(한 스텝):
  - \( \text{CO2\_resource} = \text{mcExtAir} \cdot \text{time\_interval}\cdot 10^{-6}\;[kg\,m^{-2}] \)
  - \( \text{gas\_resource} = \frac{\text{hBoilPipe}\cdot \text{time\_interval}\cdot 10^{-6}}{\text{energyContentGas}}\;[m^3\,m^{-2}] \)
- 스케일링:
  - \( r_{min} = -(\text{max\_co2\_rate}\cdot \text{time\_interval}\cdot\text{co2\_price}) - \frac{10^{-6}\cdot \text{max\_heat\_cap}\cdot \text{time\_interval}}{\text{energyContentGas}}\cdot \text{gas\_price} \)
  - \( r_{max} = \frac{\text{max\_harvest}}{\mathrm{dmfm}} \cdot \text{time\_interval}\cdot \text{tom\_price} \)
  - \( r_\mathrm{profit} = \frac{\Pi - r_{min}}{r_{max}-r_{min}} \in [0,1] \)

쉽게 말해: 한 스텝에 번 돈을 0~1 사이 점수로 환산합니다(자원비용 포함).

### 4.2 제약 페널티 ArcTanPenaltyReward
- 실내 제약: 온도[10,35], CO2[0,1000], RH[60,90]
- 절대편차: 
  - 하한 위반: \( \max(0, \mathrm{low}-x) \)
  - 상한 위반: \( \max(0, x-\mathrm{high}) \)
- 변환:
  - \( \mathrm{pen}_i = \frac{2}{\pi}\arctan(-k_i\cdot |\mathrm{penalty}_i|) \) (위반 클수록 음수 큰 값)
  - 최종 페널티 평균: \( \overline{\mathrm{pen}} = \mathrm{mean}(\mathrm{pen}_i) \le 0 \)

쉽게 말해: 설정 범위를 벗어나면 많이 벗어날수록 더 큰 감점을 주되, 너무 큰 값은 완만하게 제한합니다.

### 4.3 곱셈형 결합 MultiplicativeReward
- 구현: \( R = r_\mathrm{profit}\cdot \bigl(1 - \omega\cdot(-\overline{\mathrm{pen}})\bigr) = r_\mathrm{profit}\cdot (1 + \omega\overline{\mathrm{pen}}) \)
  - \(\overline{\mathrm{pen}}\le 0\) 이므로 페널티가 클수록 \(R\) 감소

쉽게 말해: "돈 점수"에 "환경점수 보정"을 곱해 최종 점수를 만듭니다.

---

## 5) Cython 내부 구조(물리 모델)

### 5.1 상태 벡터 정의(28차원)

주요 상태:
- CO2: `co2Air(x0), co2Top(x1)`
- 온도: `tAir(x2), tTop(x3), tCan(x4), tCovIn(x5), tCovE(x6), tThScr(x7), tFlr(x8), tPipe(x9), tSoil1..5(x10..x14)`
- 수증기압: `vpAir(x15), vpTop(x16)`
- 설비: `tLamp(x17), tIntLamp(x18), tGroPipe(x19), tBlScr(x20)`
- 일평균온도: `tCan24(x21)`
- 작물: `cBuf(x22), cLeaf(x23), cStem(x24), cFruit(x25), tCanSum(x26)`
- 시간: `time(x27)` (절대 일수)

초기화는 야간 설정 및 외기/포화수증기압 기준으로 설정됩니다.

쉽게 말해: 시작할 때 실내/작물/장치 상태를 무난한 초기값으로 맞춰둡니다.

### 5.2 파라미터 집합(Parameters)

대표값(일부):
- 구조치수/열용량/복사특성: `aFlr, aCov, hAir, cP*, lambda*`
- 통풍/누설: `aRoof, aSide, cLeakage, cWgh`
- 난방/배관: `pBoil, phiPipeE/I, lPipe, epsPipe, capPipe`
- CO2 설비: `phiExtCo2`
- 기체/에너지 상수: `rhoAir0, R, energyContentGas(=31.65 MJ/m3)`
- 작물 광합성/성장/호흡/완충 버퍼: `rg*, c*G, c*M, cBufMax/Min, laiMax, sla`
- 제어 세트포인트/밴드: `tSpDay/Night, rhMax, co2SpDay/Night, *Pband, DeadZone 등`
- 램프/인터라이트/블랙아웃 스크린 물성

이 값들은 온실 스펙/운영정책에 맞게 교정할 핵심 지점입니다.

쉽게 말해: 이 숫자들이 우리 하드웨어/운영정책을 반영합니다. 우리 시설에 맞게 바꾸면 시뮬레이터가 현실에 가까워집니다.

### 5.3 구동기(액추에이터)와 제어 로직

내부 제어는 `controlSignal()`에서 매 스텝 계산됩니다.
- 비례형 스무딩 제어 함수:
  \[
    \mathrm{PC}(x, \mathrm{SP}; \mathrm{Pband}, \mathrm{min}, \mathrm{max}) = \mathrm{min} + \frac{\mathrm{max}-\mathrm{min}}{1+\exp\!\left(-\frac{2}{\mathrm{Pband}}\ln 100\cdot (x - \mathrm{SP} - \frac{\mathrm{Pband}}{2})\right)}
  \]
- 핵심 세트포인트:
  - 난방: `heatSetPoint = isDayInside*tSpDay + (1-isDayInside)*tSpNight + heatCorrection*lampNoCons`
  - 과열 환기 기준: `heatMax = heatSetPoint + heatDeadZone`
  - CO2: `co2SetPoint = isDayInside*co2SpDay`
  - RH: `rhIn = 100*vpAir/satVp(tAir)`
  - 주/야 판정: `isDayInside = max(smoothLamp, d[8])` (태양/램프)
- 제어 출력:
  - `uBoil = PC(tAir, heatSetPoint, tHeatBand, 0, 1)`
  - `uCO2 = PC(co2InPpm, co2SetPoint, co2Band, 0, 1)`
  - `uThScr = min(thScrCold, max(thScrHeat, thScrRh))`
  - `uVent = min(ventCold, max(ventHeat, ventRh))`
  - (램프/인터라이트/그로파이프/블랙아웃은 설정에 따라 0 또는 내부 로직)

학습 액션은 위 내장 제어 후, 선택된 인덱스에 덮어써 적용됩니다(예: RL이 4개 제어를 직접 결정).

쉽게 말해: 기본 자동제어가 먼저 계산되고, 우리가 학습하려는 장치는 RL이 최종 결정을 덮어씁니다.

---

## 6) 평가/로깅과 결과

- 평가 주기: `total_timesteps / n_evals / num_cpus`
- 저장: 최적 정책, VecNormalize, 평가 CSV(`data/<project>/<group>/<run>.csv`)
- 결과 컬럼: `results_columns`에 정의(시간/실내 상태/바깥 날씨/제어/자원/제약/이익/리턴/에피소드)
 
쉽게 말해: 일정 스텝마다 에이전트를 시험해 점수를 보고, 결과를 표/그래프로 남깁니다.

---

## 7) 커스터마이징 가이드(우리 온실 맞춤)

쉽게 말해: Cython 쪽 물리 파라미터/제어 로직을 우리 시설에 맞게 바꾸고, Python/YAML에서 이름과 목록을 맞춰주면 됩니다.

### 7.1 물리 파라미터/함수 교체

- 파일:
  - 파라미터: `greenlight_gym/envs/cython/define_parameters.pxd`
  - 제어: `greenlight_gym/envs/cython/compute_controls.pxd`
  - 상태미분/적분(RK4): `difference_function.pxd`, `ODE.pxd`
- 수정 예:
  - 설비 스펙(난방용량, CO2 공급능력, 배관 치수): `p.pBoil, p.phiExtCo2, p.lPipe, p.phiPipeE/I`
  - 통풍/누설 계수: `p.aRoof, p.aSide, p.cLeakage`
  - 제어 정책 기준: `tSpDay/Night, rhMax, co2SpDay/Night, DeadZone/Pband`
- 빌드:
  ```bash
  python setup.py build_cython_only --inplace
  ```
- 주의:
  - 상태/입출력 차원 변경 시: `nx, nu, nd`와 관측/제어 매핑, YAML/파이썬 래퍼도 함께 조정

### 7.2 구동기(액추에이터) 추가/변경

- 새 구동기 추가:
  1) Cython: `controlSignal()`에서 `u[...]` 로직 추가, `nu` 증가
  2) Python 래퍼: `greenlight_gym/envs/greenlight.py`의 `control_indices`에 이름/인덱스 등록
  3) YAML: `control_signals`에 새 구동기 이름 추가
  4) 관측/로깅: 필요 시 결과 컬럼/관측 변수 업데이트
- 학습 액션에 포함하려면: `control_signals`와 `action_space`가 자동 반영됨

### 7.3 보상 함수 교체/추가

- 구현:
  - `greenlight_gym/envs/rewards.py`에 새 클래스 추가(예: `MyReward(BaseReward)`), `_compute_reward(self, GLModel)` 구현
  - 보상 조합 시 `AdditiveReward/MultiplicativeReward` 활용 또는 유사 클래스를 새로 정의
- 레지스트리 등록:
  - `greenlight_gym/envs/greenlight.py`의 `REWARDS` 딕셔너리에 이름→클래스 매핑 추가
- 사용:
  - YAML `reward_function: <키이름>`으로 선택
  - 보상에 필요한 상수/가격/경계 등은 YAML의 `GreenLightHeatCO2` 섹션 파라미터로 넘겨 초기화

### 7.4 관측 변수 확장

- Cython에서 새 관측 노출:
  - `greenlight_cy.pyx`에 `@property`로 계산/반환 추가(예: 내부 이슬점 등)
- 파이썬 관측 모듈:
  - `greenlight_gym/envs/observations.py`의 이름↔Cython 속성 매핑 확인/추가
  - YAML `model_obs_vars`/`weather_obs_vars`에 새 변수 명시

### 7.5 날씨 데이터 교체

- 로더: `greenlight_gym/common/utils.py`의 `loadWeatherData`
- 형식:
  - `nd=10` 열의 날씨 배열을 Cython에 복사(d[0..9]): 전역복사, 외기온/습, CO2, 풍속, 일사, 일중/일년 진폭 등
  - 우리 센서 포맷에 맞추려면 로더에서 컬럼 매핑 조정
- YAML:
  - `weather_data_dir, location, data_source`로 데이터 소스 스위칭
  - `weather_obs_vars`는 관측에 노출할 날씨 항목 지정

### 7.6 학습/알고리즘 조정

- PPO 하이퍼파라미터: `greenlight_gym/configs/algorithms/ppo-99.yml`의 `my_local_5min` 블록에서 수정
  - 예: `batch_size`를 `n_steps * n_envs`의 약수로(경고 제거용)

---

## 8) 재현/운영 팁

- W&B:
  - 온라인: `wandb login` 후 실행
  - 오프라인: `export WANDB_MODE=offline` 또는 `disabled`
- 체크포인트/정규화: `greenlight_gym/train_data/<project>/...`에 저장
- 디버그:
  - KeyError 보상: YAML의 `reward_function`을 `REWARDS` 키 중 하나로 지정
  - 결과 컬럼 불일치: `results_columns` 길이 ↔ 평가 데이터 마지막에 `Return, episode` 포함

쉽게 말해: 온라인이면 로그인, 오프라인이면 끄기 설정으로 돌리고, 에러 메시지에 맞춰 설정값을 맞추면 됩니다.

---

## 9) 부록: 상태/자원/최대용량 정의(요약)

- 수확량(건물질) 증가량: `a.mcFruitHarSum` 누산 → `fruit_harvest = a.mcFruitHarSum * 1e-6`
- 자원:
  - CO2: `co2_resource = a.mcExtAir * time_interval * 1e-6`
  - 가스: `gas_resource = (a.hBoilPipe * time_interval * 1e-6) / energyContentGas`
- 최대치:
  - `maxHeatCap = p.pBoil / p.aFlr`
  - `maxco2rate = p.phiExtCo2 / p.aFlr * 1e-6`
  - `maxHarvest = p.rgFruit * 1e-6`

---

## 10) 우리 온실 적용 체크리스트

- 물리 파라미터:
  - 바닥/지붕 면적, 높이, 통풍 면적, 누설계수
  - 난방 시스템 용량, 배관 규격, 가스 열량
  - CO2 공급능력과 가격
  - 작물 생리 파라미터(광합성/성장/호흡/버퍼)
  - 제어 정책 세트포인트/밴드/데드존
- 구동기:
  - 실제 제어 가능한 항목만 `control_signals`에 포함
  - 추가 구동기는 Cython/Python/YAML 세 곳 동시 반영
- 보상:
  - 가격/제약경계/가중치(k, ω)를 현장 요구에 맞게 교정
  - 필요 시 새 보상 클래스 구현
- 관측/날씨:
  - 센서 가용성에 맞춰 관측 변수 선정
  - 날씨 파일 포맷/컬럼 매핑 확인
- 검증:
  - 단일 에피소드 짧게(시간축/제어/자원) 시각화로 sanity check
  - 배치 크기/스텝 수 정합성 점검

---

## 코드 인용

```12:31:greenlight_gym/envs/greenlight.py
REWARDS = {
    "AdditiveReward": AdditiveReward,
    "MultiplicativeReward": MultiplicativeReward,
    "HarvestHeatCO2Reward": HarvestHeatCO2Reward,
    "ArcTanPenaltyReward": ArcTanPenaltyReward,
}
```

```487:501:greenlight_gym/envs/greenlight.py
harvest_reward = HarvestHeatCO2Reward(
    co2_price, gas_price, tom_price, self.dmfm, self.time_interval,
    self.GLModel.maxco2rate, self.GLModel.maxHeatCap, self.GLModel.maxHarvest,
    self.GLModel.energyContentGas,
)
penalty_reward = ArcTanPenaltyReward(k, obs_low, obs_high)
self.rewards = REWARDS[self.reward_function](
    rewards_list=[harvest_reward, penalty_reward], omega=omega
)
```

```169:178:greenlight_gym/envs/cython/compute_controls.pxd
u[0] = proportionalControl(x[2], heatSetPoint, p.tHeatBand, 0, 1)
u[1] = proportionalControl(co2InPpm, co2SetPoint, p.co2Band, 0, 1)
u[2] = fmin(thScrCold, fmax(thScrHeat, thScrRh))
u[3] = fmin(ventCold, fmax(ventHeat, ventRh))
u[4] = lampOn
u[5] = intLampOn * p.intLamps
u[6] = proportionalControl(x[2], heatSetPoint, p.tHeatBand, 0, 1) * p.pBoilGro
u[7] = p.useBlScr * (1-d[9]) * fmax(lampOn, intLampOn)
return u
```

```439:471:greenlight_gym/envs/cython/greenlight_cy.pyx
@property
def co2_resource(self):
    return self.a.mcExtAir*self.time_interval*1e-6
@property
def gas_resource(self):
    return (self.a.hBoilPipe*self.time_interval*1e-6)/self.p.energyContentGas
@property
def maxHeatCap(self):
    return self.p.pBoil / self.p.aFlr
@property
def maxco2rate(self):
    return self.p.phiExtCo2/self.p.aFlr * 1e-6
@property
def maxHarvest(self):
    return self.p.rgFruit * 1e-6
```

```187:220:greenlight_gym/envs/cython/greenlight_cy.pyx
# 상태 인덱스 개요(요약)
# x[0]: co2Air, x[1]: co2Top, x[2]: tAir, x[3]: tTop, x[4]: tCan, x[5]: tCovIn, x[6]: tCovE,
# x[7]: tThScr, x[8]: tFlr, x[9]: tPipe, x[10..14]: tSoil1..5,
# x[15]: vpAir, x[16]: vpTop, x[17]: tLamp, x[18]: tIntLamp, x[19]: tGroPipe, x[20]: tBlScr,
# x[21]: tCan24, x[22]: cBuf, x[23]: cLeaf, x[24]: cStem, x[25]: cFruit, x[26]: tCanSum,
# x[27]: time (days since 01-01-0001)
```




---

## 11) 산출물(모델)과 실시간 적용 방법

### 11.1 어떤 모델이 저장되나요?

- 학습이 끝나면 Stable-Baselines3의 PPO 모델이 `last_model.zip`으로 저장됩니다.
  - 경로 예시: `train_data/<project>/models/<runname>/last_model.zip`
- 관측/보상 정규화 통계(VecNormalize)가 별도로 저장됩니다.
  - 경로 예시: `greenlight_gym/train_data/<project>/envs/<runname>/vecnormalize.pkl`

쉽게 말해: “정책 신경망 + 하이퍼파라미터”가 zip으로, “관측/보상 스케일링 정보”가 pkl로 저장됩니다.

### 11.2 모델이 입력으로 무엇을 받고, 무엇을 내놓나요?

- 입력(Observation): YAML에 정의된 `model_obs_vars + weather_obs_vars` 순서로 구성된 벡터(부동소수점 배열)
  - 예: `air_temp, co2_conc, in_rh, ... , glob_rad, out_temp, ...`
  - 학습 시 `VecNormalize(norm_obs=True)`가 적용되어 “정규화된 관측”을 보았습니다.
- 출력(Action): `control_signals` 길이의 연속 제어값, 범위는 `[-1, 1]`
  - 예: `[uBoil, uCO2, uThScr, uVent]`
  - 환경 내부에서 `[0,1]`로 다시 스케일되어 Cython 모델에 전달됩니다.

쉽게 말해: “정해진 순서의 숫자 벡터”를 넣으면 “장치를 얼마나 켤지(−1~1)”가 나옵니다.

### 11.3 오프라인 추론(시뮬레이터 환경에서)

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 1) 학습된 정책 로드
model_path = "train_data/local_test/models/<runname>/last_model.zip"
model = PPO.load(model_path)

# 2) 정규화 통계 로드(관측 정규화를 동일하게 적용)
vec_path = "greenlight_gym/train_data/local_test/envs/<runname>/vecnormalize.pkl"
vec_norm = VecNormalize.load(vec_path)

# 3) 환경을 준비(필요 시 더미 환경)
#   - 실제 예측시에는 obs를 직접 정규화해서 넣거나,
#     DummyVecEnv로 래핑된 env를 구성해 model.predict에 전달합니다.

def normalize_obs(obs):
    # vec_norm.obs_rms.mean/var 를 사용해 학습과 동일한 방식으로 정규화
    # SB3는 predict에 대해 자동 정규화를 해주지 않으므로 사용자가 적용해야 합니다.
    return vec_norm.normalize_obs(obs)

# 예시 관측(obs)을 받아 정책 추론
raw_obs = ...  # YAML 순서대로 만든 관측 벡터 (shape: [obs_dim])
obs = normalize_obs(raw_obs.copy())
action, _ = model.predict(obs, deterministic=True)
# action: [-1, 1] 범위. 장치별 비율로 해석(환경 내부에서 [0,1]로 재스케일)
```

### 11.4 실시간 적용(현장 센서/구동기 연동)

관측 파이프라인(권장):
1) 센서 값 수집(온도/습도/CO2/일사/풍속/시각 등)
2) YAML의 `model_obs_vars + weather_obs_vars` 순서대로 벡터 조립
3) `VecNormalize`로 관측 정규화 적용(학습과 동일한 통계)
4) `model.predict(obs, deterministic=True)`로 액션 계산
5) 액션(−1~1)을 설비 제어 비율로 변환 → 필요 시 물리 단위(밸브 개도율, 환기창 %)로 매핑
6) PLC/장치 제어 명령 전송

샘플 코드 골격:

```python
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

model = PPO.load("train_data/local_test/models/<runname>/last_model.zip")
vec_norm = VecNormalize.load("greenlight_gym/train_data/local_test/envs/<runname>/vecnormalize.pkl")

def build_obs_from_sensors():
    # 1) 센서/SCADA/PLC로부터 읽기
    # 2) YAML 정의 순서에 맞춰 배열 구성
    # 3) dtype=float32 권장
    return np.array([...], dtype=np.float32)

def to_physical_actuators(action):
    # RL 출력 [-1,1] → [0,1] 스케일
    scaled = (action + 1.0) / 2.0
    # 설비별 실제 단위로 변환(예: 개도율 %, 유량, setpoint 등)
    # 시설 사양에 맞는 선형/비선형 맵핑 적용
    return scaled

while True:
    raw_obs = build_obs_from_sensors()
    obs = vec_norm.normalize_obs(raw_obs.copy())
    action, _ = model.predict(obs, deterministic=True)
    actuators = to_physical_actuators(action)
    # 여기에서 PLC/장치에 actuators 전송
    time.sleep(300)  # 5분 간격(환경 설정과 맞춤)
```

주의사항:
- 관측 순서/범위가 학습과 동일해야 합니다(순서 중요).
- VecNormalize 통계를 반드시 동일하게 적용해야 정책이 의도대로 동작합니다.
- 물리 장치 매핑은 시설별 사양과 안전 로직(상한/하한, 램프업/다운 등)을 반드시 반영하세요.

쉽게 말해: 파일 두 개(`last_model.zip`, `vecnormalize.pkl`)만 있으면, 센서를 벡터로 만들고 정규화한 뒤 `predict`로 제어값을 얻어 장치에 보내면 됩니다.
