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
9. [결과](#9-결과-학습-후-업데이트)

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

→ 이 변형들이 곧 **your solution(SAC)의 ablation study**를 구성합니다.

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

# 4) 시각화 (results/의 모델·로그를 자동 탐색)
python visualize.py
```

**출력물** (`results/`): `<algo>_seed<seed>.zip`(모델), `<algo>_seed<seed>_best/`(best), `<algo>_seed<seed>_metrics.csv`, `eval/evaluations.npz`, `tb/`(tensorboard), `viz/*.png|gif`.

---

## 8. 파일 구조

```
2/f1/
├── build_track.py    # 트랙 이미지 → 2D 중앙선/경계 추출 (assets/track.npz)
├── env.py            # F1DriverEnv (gymnasium 연속 환경: 상태·행동·전이·보상)
├── wrappers.py       # DiscretizedF1Driver (DQN용 21개 이산 액션)
├── train.py          # SB3 학습 파이프라인 (DQN/PPO/SAC, eval·ckpt·metrics·tb)
├── visualize.py      # 트랙맵·레이싱라인·raycast GIF·학습곡선·알고리즘 비교
├── requirements.txt
├── assets/           # track.npz, track_preview.png
└── results/          # 학습 결과물 (.gitignore)
```

---

## 9. 결과 (학습 후 업데이트)

> 본 섹션은 GPU 서버 본학습 완료 후 채워집니다. (현재 PPO 300k 검증: ep_rew −498 → −78 로 학습 확인)

### 9-1. 학습 곡선 — `results/viz/learning_curves.png`
*(DQN vs PPO vs SAC 에피소드 보상)*

### 9-2. 성공률 / 크래시율 — `results/viz/success_curves.png`

### 9-3. 알고리즘 비교 — `results/viz/metric_comparison.png`
| 방법 | 성공률 | 크래시율 | 평균 보상 | 완주 스텝(↓) |
|------|--------|----------|-----------|--------------|
| DQN | TBD | TBD | TBD | TBD |
| PPO | TBD | TBD | TBD | TBD |
| **SAC** | TBD | TBD | TBD | TBD |

### 9-4. 레이싱 라인 / 주행 영상 — `results/viz/traj_<algo>.png`, `drive_<algo>.gif`
*(속도 색상 궤적 + raycast·텔레메트리 애니메이션)*

### 9-5. Ablation (SAC) — baseline vs no_shaping vs aggressive
*(리워드 설계 타당성 근거)*
