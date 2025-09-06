# Seolhyang (Strawberry) Digital Twin & RL Pipeline - Repro Guide

## 1) Requirements
- OS: Windows 10/11 (Linux/macOS도 가능)
- Python: 3.11.x 권장
- Build tools: C/C++ 컴파일러 (Windows: Build Tools for Visual Studio), `pip`, `wheel`
- Git

## 2) Clone & Setup
```bash
# 1) Clone
git clone <YOUR_REMOTE_URL>.git
cd GreenLightGym-master

# 2) Python venv
python -m venv gr
source gr/Scripts/activate  # Windows Git Bash / PowerShell: .\gr\Scripts\activate

# 3) Upgrade pip & wheel
python -m pip install --upgrade pip wheel

# 4) Install deps (필요 시 requirements.txt 사용)
pip install -r requirements.txt  # 없으면 아래 최소 세트 참고

# 최소 설치 가이드(예)
# pip install numpy cython gymnasium pyyaml stable-baselines3 pandas

# 5) Build Cython extensions (필수)
python setup.py build_ext --inplace
```

## 3) Key Configs
- 설향 파라미터: `greenlight_gym/envs/cython/define_parameters.pxd` (이미 반영됨)
- 전기요금: `config/greenhouse.yaml` → `digital_twin.tariffs.electricity_eur_per_kwh`
- 전기만 비용: `greenlight_gym/configs/envs/my_local_5min.yml` 와 `benchmark-rule-based.yml`에서 `co2_price: 0.0`, `gas_price: 0.0`
- 관측 벡터: `my_local_5min.yml`의 `model_obs_vars`에서 `co2_resource, gas_resource` 제거됨

## 4) Quick Validation
```bash
# 가상환경 활성화 후
python scripts/validate_strawberry_seolhyang.py
```
- 출력: 일사/온도/CO2 스윕에 따른 겉보기 동화(A_app) 및 과실 건물 증가율

## 5) Training (예시)
- 병렬 환경/하이퍼파라미터는 프로젝트 스크립트에 맞춰 조정
- 기본 주기: 300 s, 권장 스텝: 0.5–2M(베이스라인) → 5–10M(튜닝)

## 6) Notes
- 외부 센서 입력(필수): GHI, 외기온도, 외기RH, 외기CO2, 풍속
- 내부 센서(필수): 실내 온도, RH, CO2
- CO2/가스 비용은 0으로 처리, 전력 비용만 보상에서 차감
- Cython 변경 후에는 반드시 재빌드 필요: `python setup.py build_ext --inplace`
