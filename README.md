<!--
  ⚠️ 제출 주의: 본 과제(EL5001 Proj 02) 제출 규정상 최종 README·코드·주석·PPT는 **영어**여야 합니다.
  이 한국어 문서는 작업/이해용 초안이며, 제출 직전에 영어판(README.md)으로 옮길 것.
-->

# F1 Driver — Model-free Deep RL (2D 연속 주행)

F1 레이스를 **2차원 연속 주행 MDP**로 모델링하고, **Model-free Deep RL**(DQN · PPO · SAC)로 "차량 내부 드라이버" 정책을 학습합니다.
Project 01의 1차원/이산형 *피트 월 전략가*를, 차량을 직접 조종하는 **2차원/연속형 드라이버**로 전면 재설계했습니다.

> **Proj 01 대비 핵심 변화**
> - 상태: 6,840개 이산 상태 → **14차원 연속 벡터**(차량·센서·환경)
> - 행동: 6개 이산 행동 → **연속 Box(4)** (조향·페달·ERS·피트)
> - 해법: Value Iteration / Q-table → **함수 근사 기반 Deep RL** (DQN, PPO, SAC)
> - 트랙: 19개 섹션 인덱스 → **실제 서킷 이미지에서 추출한 연속 2D 트랙**

---

## 목차

1. [동기 & 문제 정의](#1-동기--문제-정의)
2. [트랙 재구성 (build_track.py)](#2-트랙-재구성-build_trackpy)
3. [MDP 정식화](#3-mdp-정식화)
4. [알고리즘 구현](#4-알고리즘-구현)
5. [하이퍼파라미터 & 학습 설정](#5-하이퍼파라미터--학습-설정)
6. [리워드 셰이핑 & Ablation](#6-리워드-셰이핑--ablation)
7. [실행 방법](#7-실행-방법)
8. [파일 구조](#8-파일-구조)
9. [결과](#9-결과)

---

## 1. 동기 & 문제 정의

F1 드라이버는 매 순간 **순차적 의사결정**을 수행합니다 — 코너에서 얼마나 브레이크할지, 직선에서 ERS를 언제 방출할지, 비가 오면 언제 피트인해 타이어를 교체할지. 이 모든 결정은

- **현재 상태에 의존**(Markovian)하고,
- **지금의 선택이 수십 스텝 뒤 결과(충돌/완주/랩타임)에 영향**을 미치며,
- **장기 누적 보상(안전하고 빠른 완주)** 을 최대화해야 합니다.

행동 공간이 **연속(조향·가감속)이면서 이산 전략(피트)이 혼합**되어 있어, 단순 이산 Q-table로는 표현이 어렵습니다. 따라서 **함수 근사 기반 Deep RL**이 적합합니다.

- **에이전트**: 차량(드라이버)
- **목표**: 충돌 없이 목표 랩(기본 3랩)을 가능한 빠르게 완주
- **창의성**: 실제 서킷 이미지에서 추출한 트랙 위에서, raycast 센서·타이어 온도·연속 날씨(노면 젖음)를 포함한 멀티모달 상태로 주행

---

## 2. 트랙 재구성 (`build_track.py`)

실제 서킷 이미지(`1/track.webp`)에서 주행 가능한 2D 중앙선을 자동 추출합니다.

```
track.webp ──build_track.py──▶ assets/track.npz        (centerline / left / right / half_width)
                            └─▶ assets/track_preview.png (검증용 4-패널 이미지)
```

**추출 파이프라인**
1. 검은 트랙 밴드 마스킹 → 가장 큰 연결요소만 추출(라벨·범례 제거)
2. 트랙은 고리(annulus) 형태 → **바깥/안쪽 윤곽(contour)의 중점**을 중앙선으로 계산
3. 윤곽 사이 거리로 **트랙 폭(half_width)** 추정 → 백분위 클램프 + 스무딩으로 코너 이상치 제거
4. 픽셀 → 월드 좌표 변환, 한 랩 길이를 5,000 m로 스케일링, 400개 waypoint로 리샘플

**env는 `track.npz`만 로드**하므로, 이미지를 수정하면 `python build_track.py`만 다시 실행하면 됩니다(코드 수정 불필요).

---

## 3. MDP 정식화

### 3-1. 상태 공간 (Observation) — 14차원 연속 벡터

| # | 변수 | 정규화 | 설명 |
|---|------|--------|------|
| 1 | Speed | `/MAX_SPEED` | 현재 속도 |
| 2–3 | Heading error | sin, cos | 차량 진행 방향과 트랙 접선의 각도 차 |
| 4 | Lateral offset | `/half_width` | 중앙선 기준 횡방향 이탈(부호 포함) |
| 5 | Compound | 0/1 | Dry(0) / Inter(1) |
| 6 | Tire temp | 0–1 | Cold→Optimal→Overheat |
| 7 | ERS | 0–1 | 배터리 잔량 |
| 8–12 | Raycast ×5 | 0–1 | 전방·좌우 부채꼴(−70°,−35°,0°,35°,70°) 벽 거리 |
| 13 | Wetness | 0–1 | 노면 젖음 정도(연속) |
| 14 | Rain prob | 0–1 | 다음 랩 강우 관련 확률 |

> **Markov 성립(근사)**: 차량 운동·타이어·날씨·센서 정보가 모두 상태에 포함되어 다음 전이를 결정합니다. 날씨의 잠재 상태(raining)는 명시되지 않지만 wetness 추세로 관측 가능 → **근사적으로 Markovian**(PDF가 허용).

### 3-2. 행동 공간 (Action) — 혼합형을 단일 `Box(4)`로 인코딩

| # | 행동 | 범위 | 의미 |
|---|------|------|------|
| 0 | Steering | −1 … 1 | 좌 … 우 |
| 1 | Pedal | −1 … 1 | 풀 브레이크 … 풀 스로틀 |
| 2 | ERS_Deploy | −1 … 1 | (내부에서 0…1로 매핑) 배터리 방출량 |
| 3 | Pit_signal | −1 … 1 | `> 0.5`면 피트인 요청 → **다음 start/finish 통과 시 타이어 교체** |

> **설계 근거**: SB3의 SAC/DDPG/TD3는 `Box`만, 어떤 SB3 알고리즘도 Dict/Tuple 혼합 액션을 지원하지 않습니다. 따라서 **이산 피트 결정을 연속 Box의 한 차원으로 임베딩**(임계값 0.5)하여 모든 알고리즘과 호환되게 했습니다. 액션은 전부 `[-1,1]` 대칭(SAC 권장).

### 3-3. 전이 (Transition)

- **차량 운동(2D 키네마틱, bicycle 근사)**: 페달→가감속, 조향→요(yaw)율(속도 비례 권한), 드래그, 위치 적분.
- **타이어 온도**: 하드 드라이빙(페달·속도·조향)으로 가열, 주변온도로 냉각. 마이크로 레이스를 위해 **10배 가속**(`TIRE_SCALE`).
- **노면 젖음(Wetness)**: 현재 날씨(raining) 목표값으로 드리프트, 역시 **10배 가속**.
- **날씨 전이**: **랩 완료 시** `P(Dry→Rain)=0.30`, `P(Rain→Dry)=0.40`로 토글.
- **피트**: 요청 시 다음 랩 진입에서 타이어를 노면 상태에 맞춰 교체(Wet→Inter, Dry→Dry), 온도 리셋, 페널티 부과.
- **슬립**: 노면이 젖었는데(`wetness>0.45`) Dry 타이어면 그립 0.6배(감속·조향 약화) + 페널티.

### 3-4. 보상 (Reward Shaping)

| 항목 | 값 | 설명 |
|------|----|------|
| Progress (dense) | `+0.05 × 전진거리(m)` | 중앙선 따라 전진한 거리 비례(세그먼트 투영으로 연속) |
| Time penalty | `−0.03 / step` | 빠른 완주 유도 |
| Speed reward | `+speed_reward × (v/Vmax)` | (기본 0, ablation용) 속도 장려 |
| Overheat | `−0.5 / step` | 타이어 과열 구간 |
| Slip | `−0.3 / step` | 젖은 노면 + Dry 타이어 |
| Pit | `−30` | 피트 1회 |
| **Crash** | **`−500` + 종료** | raycast 기준 벽 충돌(트랙 이탈) |
| **Complete** | **`+200` + 종료** | 목표 랩 완주 |

> (선택) **Terminal time-bonus** `finish_time_bonus`(기본 0, Phase-2 §6-1에서만 사용): 완주 시 빠를수록 추가 보너스 → 진짜 목표(제한시간 내 최단 완주)를 보상에 직접 인코딩.

### 3-5. 종료 조건 (Termination)

- **충돌**: 횡방향 이탈이 트랙 폭 초과 → `−500`, `terminated=True`
- **완주**: `laps ≥ n_laps` → `+200`, `terminated=True`
- **시간 초과**: `steps ≥ max_steps`(기본 `n_laps×500`) → `truncated=True`

---

## 4. 알고리즘 구현

PDF 요구(① value-based ② policy-based ③ your solution)에 맞춘 라인업:

| 역할 | 알고리즘 | 액션 공간 | 비고 |
|------|----------|-----------|------|
| **Baseline 1 (value-based)** | **DQN** | Discrete(21) | `DiscretizedF1Driver` 래퍼로 연속 Box를 21개 이산 액션으로 매핑 |
| **Baseline 2 (policy-based)** | **PPO** | Box(4) | on-policy actor-critic |
| **Your solution** | **SAC** | Box(4) | off-policy, 엔트로피 기반 탐색 — 연속 제어·확률적 동역학에 적합 |
| **추가 비교군** | **TD3** | Box(4) | off-policy 결정론적 — SAC 대비 탐색 방식 비교용 |

- **DiscretizedF1Driver** (`wrappers.py`): 조향 5단계 × {브레이크/코스트/스로틀} + {스로틀+ERS} 5개 + 피트 1개 = **21개 이산 액션**. DQN이 동일 환경에서 baseline으로 동작하도록 함.
- **SAC를 your solution으로 선택한 근거**:
  - **off-policy → 샘플 효율** 높음(짧은 마이크로 레이스에 유리)
  - **엔트로피 최대화 탐색** → 젖음/타이어 같은 확률적 동역학에서 강건
  - 연속 제어를 native하게 처리, 학습 안정성 우수

---

## 5. 하이퍼파라미터 & 학습 설정

공통: `γ = 0.99`, MLP 정책 `[256, 256]`(PPO/SAC), device 자동 감지(GPU 우선).

| 알고리즘 | 주요 하이퍼파라미터 |
|----------|--------------------|
| **DQN** | lr 1e-3, buffer 200k, learning_starts 5k, batch 128, train_freq 4, target_update 2k, exploration_fraction 0.3, final_eps 0.05 |
| **PPO** | n_steps 1024, batch 256, n_epochs 10, GAE λ 0.95, ent_coef 0.0, lr 3e-4, **n_envs 병렬**(SubprocVecEnv) |
| **SAC** | lr 3e-4, buffer 300k, learning_starts 10k, batch 256, τ 0.005, train_freq 1, ent_coef auto |

- **병렬 env**: env 스텝이 병목 → PPO는 `--n-envs`로 병렬화(GPU보다 속도 이득 큼).
- **로깅**: `EvalCallback`(주기적 평가), `CheckpointCallback`(체크포인트), `EpisodeMetrics`(에피소드별 성공률·크래시율·랩길이·평균속도·과열비율 → CSV), tensorboard.

---

## 6. 리워드 셰이핑 & Ablation

보상 가중치는 모두 `F1DriverEnv(...)` 생성자 kwargs로 조정 가능합니다(기본값=설계값, 비파괴). `train.py --reward-preset`으로 변형 학습:

| Preset | 설정 | 검증 목적 |
|--------|------|----------|
| `baseline` | 설계 그대로 | 기준 |
| `no_shaping` | overheat/slip/time 페널티 제거 | **보상 셰이핑이 실제로 도움이 되는가** |
| `aggressive` | crash_pen 200 + speed_reward 0.02 | **충돌 회피 과다("소심한 정책")** 완화 효과 |
| `racing` | crash_pen 100 + speed_reward 0.05 | brake-and-park 국소최적 탈출 (메인 학습에 사용한 프리셋) |

→ 이 변형들이 곧 **your solution(SAC)의 ablation study**를 구성합니다(결과는 §9-4). raycast 센서 on/off(`--no-raycast`) ablation은 §9-5.

### 6-1. Phase-2: 타임어택 reward 설계 (진짜 목표 직접 최적화)

본 과제의 *진짜* 목표는 **제한시간(3랩 = 375 s) 안에 가능한 빠르게 완주**하는 것입니다. 분석 결과 이 환경은
**시간제한이 binding**(완주 랩타임이 제한선에 근접)이라, "빠르게"와 "완주"가 같은 방향입니다 — 즉 페이스를 끌어올리면
성공률↑·랩타임↓이 동시에 개선되며, 한계는 충돌입니다. 그런데 기존 완주 보너스는 **고정 +200**이라 *얼마나 빨리*
끝냈는지를 반영하지 못합니다. 이를 보완하는 **terminal time-bonus**를 추가했습니다(env kwarg `finish_time_bonus`, 기본 0 → 비파괴):

```
완주 시  reward += complete_bonus + finish_time_bonus × max(1 − steps/max_steps, 0)
```

완주가 빠를수록(steps 작을수록) 보너스가 커집니다. 한계효과는 `−finish_time_bonus/max_steps`(예: 300/1500 = 0.2/step)로
dense `time_pen`(0.03)보다 강하지만 **완주해야만 받는 terminal 신호**라 정지(park) 정책은 영향받지 않고 완주 정책만 가속하도록 유도합니다.

| Preset | 설정 (vs `racing`) | 격리하는 레버 |
|--------|--------------------|----------------|
| `timeattack_dense` | speed_reward 0.10 + time_pen 0.06 | dense 페이스 압력 강화 |
| `timeattack_finish` | + finish_time_bonus 300 | terminal 빠른-완주 보너스 |
| `timeattack` | dense + terminal 둘 다 | 풀 제안 |

학습: `bash run_phase2.sh` (SAC 3프리셋 × 3시드 from-scratch + best `racing` 모델 warm-start fine-tune).
**평가는 불변 지표(랩타임/성공률/크래시율)로만** — 프리셋마다 보상 스케일이 달라 보상끼리 비교 불가. 결과는 §9-7.

---

## 7. 실행 방법

```bash
# 0) 의존성
pip install -r requirements.txt

# 1) 트랙 추출 (이미지 수정 시에만 재실행)
python build_track.py

# 2) 파이프라인 점검 (CPU에서 빠르게)
python train.py --algo all --smoke

# 3) 학습 (GPU 서버 권장)
python train.py --algo dqn --timesteps 500000 --seed 0
python train.py --algo ppo --timesteps 1000000 --n-envs 8 --seed 0
python train.py --algo sac --timesteps 500000 --seed 0
#  Ablation:
python train.py --algo sac --reward-preset no_shaping --seed 0
python train.py --algo sac --reward-preset aggressive --seed 0

# 3b) Phase-2 타임어택 reward 설계 (§6-1) — from-scratch sweep + fine-tune
python train.py --algo sac --reward-preset timeattack --seed 0
#  best 모델에서 저LR warm-start fine-tune:
python train.py --algo sac --reward-preset timeattack --seed 0 \
    --init-from results/sac_racing_seed0_best/best_model.zip --learning-rate 1e-4 --timesteps 200000

# 4) 시각화 (results/의 모델·로그를 자동 탐색)
python visualize.py
```

서버에서 전체를 한 번에 돌리려면: **메인+ablation은 `bash run_all.sh`**, **Phase-2는 `bash run_phase2.sh`**(run_all 이후 실행).

**출력물** (`results/`): `<algo>_seed<seed>.zip`(모델), `<algo>_seed<seed>_best/`(best), `<algo>_seed<seed>_metrics.csv`, `eval/evaluations.npz`, `tb/`(tensorboard), `viz/*.png|gif`.

---

## 8. 파일 구조

```
2/f1/
├── build_track.py    # 트랙 이미지 → 2D 중앙선/경계 추출 (assets/track.npz)
├── env.py            # F1DriverEnv (gymnasium 연속 환경: 상태·행동·전이·보상)
├── wrappers.py       # DiscretizedF1Driver (DQN용 21개 이산 액션)
├── train.py          # SB3 학습 파이프라인 (DQN/PPO/SAC/TD3, 프리셋·fine-tune·eval·ckpt·metrics·tb)
├── visualize.py      # 트랙맵·레이싱라인·raycast GIF·학습곡선·알고리즘 비교·ablation
├── run_all.sh        # 메인 매트릭스(4 algo×5 seed) + 보상/raycast ablation + viz (GPU 서버)
├── run_phase2.sh     # Phase-2 타임어택 reward sweep + warm-start fine-tune + viz
├── requirements.txt
├── assets/           # track.npz, track_preview.png
└── results/          # 학습 결과물 (CSV·figure·best 모델 일부만 커밋, 나머지 .gitignore)
```

---

## 9. 결과

GPU 서버에서 **메인 4개 알고리즘 × 5 시드** (`racing` 보상 프리셋, 날씨 토글은 랩 경계에서 자연 발생) +
**SAC ablation**(보상 프리셋 4종 / raycast on·off)을 학습했습니다. 학습량: DQN·SAC·TD3 각 500k,
PPO 1M timestep. 모든 수치는 **마지막 300 에피소드 평균 (± 시드 표준편차)** 입니다.

> **평가 지표 주의**: 보상은 **대리(surrogate) 지표**이고, 본 과제의 *진짜* 목표는 **충돌 없이 빠르게 완주**하는 것입니다.
> 따라서 핵심 지표는 **랩타임(초, 성공 에피소드에 한해 `ep_len·DT/N_LAPS`)·성공률·크래시율**이며, 보상은 보조로만 봅니다.
> 또한 **보상 프리셋이 다르면 보상 스케일이 달라 서로 비교 불가** — ablation은 랩타임·성공률·진행거리로 판단합니다.

### 9-1. 학습 곡선 — `results/viz/learning_curves.png`

x축을 **timestep**으로 두어(에피소드 인덱스가 아님), 즉시 충돌해 짧은 에피소드를 수천 개 만드는 DQN과
긴 에피소드의 연속제어 알고리즘을 공정하게 비교합니다.

- **SAC**: 샘플 효율이 가장 좋음 — ~350k에서 보상 ~165로 가장 빠르게 수렴.
- **PPO**: 더 느리게 오르지만 1M까지 꾸준히 상승해 최종 ~145로 SAC와 동급.
- **TD3**: 초반 −210까지 떨어졌다가 회복(~−45)하나 끝까지 양(+)으로 못 올라옴.
- **DQN**: ~−90에서 평탄 — 이산화된 액션으로는 주행 자체를 학습하지 못함.

### 9-2. 알고리즘 비교 — `results/viz/metric_comparison.png`

| 방법 | 랩타임(s) ↓ | 성공률 ↑ | 크래시율 ↓ | 평균 진행거리(m) | 평균 보상(대리) |
|------|------------:|---------:|-----------:|-----------------:|----------------:|
| DQN  | — (완주 없음) | 0.0 % | 98.9 % |   767 |  −86 |
| **PPO** | 121.3 | **6.9 %** | 30.8 % | **5099** | **142** |
| **SAC** | **118.3** | 2.8 % | **18.5 %** | 4834 | 106 |
| TD3  | 120.5 | 1.2 % | 18.9 % | 2520 |  −86 |

- **이산화 value-based(DQN)의 붕괴**: 21개 이산 액션으로는 거의 즉시(평균 767 m) 충돌(크래시 98.9 %)하며 완주 0건.
  연속 제어 문제를 이산화로 푸는 접근의 한계를 명확히 보여줍니다.
- **PPO vs SAC (핵심 트레이드오프)**: PPO가 **완주를 가장 자주(6.9 %)** 성공하고 가장 멀리 가지만,
  SAC는 **랩타임이 가장 짧고(118.3 s) 가장 안전(크래시 18.5 %)** 합니다 → "자주 끝내는 PPO" vs "빠르고 안전한 SAC".
- 전반적으로 성공률이 낮은(≤ 7 %) 이유: 3랩 내내 코너·타이어 과열·날씨/피트 이벤트를 모두 통과해야 하는
  난도 높은 마이크로 레이스이기 때문입니다(결정론적 롤아웃은 더 멀리 감 — §9-3).

### 9-3. 레이싱 라인 / 주행 영상 — `results/viz/traj_<algo>.png`, `drive_sac.gif`

속도 색상 궤적(`traj_*`)과 raycast·텔레메트리 애니메이션(`drive_sac.gif`). 결정론적 롤아웃(seed 7) 기준
**SAC는 2랩 이상을 안정적으로 주행**(직선 가속 → 코너 감속, reward ~454)하며, 충돌 시 빨간 X로 종료 지점을 표시합니다.

### 9-4. 보상 셰이핑 Ablation (SAC) — `results/viz/reward_ablation.png`

SAC를 4개 보상 프리셋으로 학습(`baseline`/`no_shaping`/`aggressive`/`racing`, seed 0; `racing`은 5 시드):

| 프리셋 | 성공률 | 크래시율 | 진행거리(m) | 거동 |
|--------|-------:|---------:|------------:|------|
| baseline   | 0 % | 4.7 % |  648 | 코너 앞에서 정지 후 주차("brake-and-park") |
| no_shaping | 0 % | 6.3 % |  747 | 동일하게 주차 |
| aggressive | 0 % | 8.3 % |  775 | 거의 주차(방향만 약간 개선) |
| **racing** | **2.8 %** | 18.5 % | **4834** | **유일하게 실제 주행** |

→ **핵심 발견**: 충돌 페널티(−500)가 시간 페널티를 압도하면, 약한 정책은 코너에서 **완전 정지**하는 안전한 국소최적("brake-and-park")에 갇힙니다.
속도 보상을 강하게 준 `racing` 프리셋(crash_pen 100 + speed_reward 0.05)만 이 국소최적을 탈출해 진행거리가 7× 이상 증가합니다.
즉 **보상 셰이핑은 단순 미세조정이 아니라 학습 성패를 가르는 요소**입니다. (크래시율이 함께 오르는 것은 정지 대신 실제로 코너에 진입하기 때문 — 의도된 trade-off)

### 9-5. Raycast 센서 Ablation (SAC) — `results/viz/raycast_ablation.png`

`racing` SAC에서 5개 raycast 거리 센서를 끄고(`--no-raycast`) 학습 (seed 0 기준):

| 설정 | 크래시율 | 진행거리(m) | 평균 보상 |
|------|---------:|------------:|----------:|
| raycast **ON**  (seed 0) | 14.0 % | 2901 |  32 |
| raycast **OFF** (seed 0) | 31.3 % | 7653 | 220 |

→ 센서를 끄면 진행거리·보상은 오히려 커지지만 **크래시율이 약 2배**로 뜁니다. raycast는 벽 거리 정보를 제공해
**충돌을 회피(안전성↑)** 하는 대신 다소 보수적으로 만듭니다 — 센서가 안전한 주행에 기여함을 확인.
(메인 비교의 raycast-ON SAC는 5 시드 평균 크래시 18.5 %; OFF는 단일 시드라 절대수치보다 **경향**으로 해석)

### 9-7. Phase-2 타임어택 reward 설계 — `results/viz/timeattack_ablation.png`, `finetune_compare.png`

> `bash run_phase2.sh` 완료 후 채워집니다. (§6-1 설계, 불변 지표 = 랩타임/성공률/크래시율)

| 프리셋 | 랩타임(s) ↓ | 성공률 ↑ | 크래시율 ↓ | 비고 |
|--------|------------:|---------:|-----------:|------|
| racing (base) | 118.3 | 2.8 % | 18.5 % | Phase-1 기준 |
| timeattack_dense | TBD | TBD | TBD | dense 페이스만 |
| timeattack_finish | TBD | TBD | TBD | terminal 보너스만 |
| timeattack | TBD | TBD | TBD | 둘 다 |
| **timeattack (fine-tune)** | TBD | TBD | TBD | best racing → warm-start |

*(빈칸은 학습 후 채움 — 어느 reward가 제한시간 내 최단 랩타임을 내는지, 속도↔안전 trade-off와 함께 보고)*

### 9-8. 요약

- **연속 제어(PPO/SAC/TD3) ≫ 이산화 value-based(DQN)** — 이 문제에서 액션 이산화는 치명적.
- **SAC = 가장 빠르고 안전**(your solution으로서 타당), **PPO = 가장 자주 완주** — 둘이 상보적.
- 보상 셰이핑(§9-4)이 brake-and-park 국소최적 탈출에 결정적이고, raycast 센서(§9-5)는 안전성에 기여.
- Phase-2(§9-7): 진짜 목표(제한시간 내 최단 완주)를 terminal reward에 직접 인코딩해 최적 reward를 탐색.
