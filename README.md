# Predicting Student Health Risk (Kaggle Playground Series S6E7)

대학생의 생활 습관·생체 지표 데이터로부터 건강 상태(`health_condition`)를 3개 클래스(`at-risk` / `fit` / `unhealthy`)로 분류하는 Kaggle Playground Series 대회 프로젝트입니다. 평가지표는 **Balanced Accuracy**이며, 클래스 불균형이 매우 심합니다(at-risk 85.9% / unhealthy 8.4% / fit 5.8%).

## 파일 구성

| 파일 | 설명 |
|---|---|
| `predicting_student_health_risk.ipynb` | EDA부터 모델링, 16개 절에 걸친 실험 기록까지 담긴 메인 노트북 |
| `submission.csv` | 최종 제출 파일 (최종 채택 모델 기준) |
| `student_health_risk_summary.docx` | 1~11절(baseline~하이퍼파라미터 튜닝) 요약 보고서 |
| `today_summary.docx` | 12~16절(FT-Transformer~stress 실험) 요약 보고서 |

## 데이터

- 학습 데이터 690,088행 / 테스트 데이터 295,753행, 원본 피처 13개(수치형 7 + 범주형 6)
- 거의 모든 피처에 1~12% 수준의 결측치가 train/test에 동일한 패턴으로 존재(무작위성에 가까움)
- 범주형 피처는 전부 3~4개 수준의 저-카디널리티(`stress_level`, `sleep_quality`, `physical_activity_level`, `smoking_alcohol`, `diet_type`, `gender`)

## 방법론

- **검증**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`를 모든 실험에서 동일하게 재사용해 공정하게 비교
- **불균형 처리**: 트리 모델은 `sample_weight=compute_sample_weight('balanced', ...)`, 신경망은 `class_weight`를 `CrossEntropyLoss`에 반영
- **결측치·범주형 처리**: 별도 전처리 없이 대응 가능한 트리 기반 모델(XGBoost `enable_categorical=True` 등)을 기본 방향으로 채택
- **신규 피처/모델 후보는 항상 mutual information 등 사전 진단 → 실제 OOF 비교 → 정직한 결과 보고 순서로 검증**

## 최종 모델

**XGBoost + FEATURES_V2(15개 피처) + 튜닝된 하이퍼파라미터**

```python
TUNED_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05, min_child_weight=10,
    subsample=0.7, colsample_bytree=0.7, reg_lambda=2,
    tree_method='hist', enable_categorical=True, eval_metric='mlogloss',
    early_stopping_rounds=30,
)
```

- **FEATURES_V2** = 수치형 7개(`sleep_duration`, `heart_rate`, `bmi`, `calorie_expenditure`, `step_count`, `exercise_duration`, `water_intake`) + 파생 4개(`stress_ord`, `sleep_quality_ord`, `physical_activity_level_ord`, `sleep_x_stress`) + 범주형 4개(`stress_level`, `sleep_quality`, `physical_activity_level`, `smoking_alcohol`)
- **최종 OOF Balanced Accuracy: 0.9499**

## 실험 이력

총 16개 절에 걸쳐 모델·피처·앙상블 방향을 폭넓게 실험했습니다. 대부분의 실험이 baseline을 넘지 못했지만, 그 실패의 원인을 매번 데이터로 진단하고 기록했습니다.

| 절 | 실험 | OOF Balanced Acc. | 채택 |
|---|---|---|---|
| 1. Baseline | LightGBM vs XGBoost, 13개 원본 피처 | 0.9493 | XGBoost 채택 |
| 2. 피처 엔지니어링 | sleep×stress 상호작용, 서수 인코딩 3종, diet_type·gender 제거 → FEATURES_V2(15개) | 0.9497 (+0.0004) | 채택 |
| 3. 추가 피처 실험 | step_count×stress_ord 상호작용 | 0.9497 (±0) | 미채택 |
| 4. CatBoost | 네이티브 범주형 처리 비교 + XGBoost 블렌딩 | 0.9488 / 블렌드 개선 없음 | 미채택 |
| 5. RandomForest | 배깅 계열 모델 비교 + XGBoost 블렌딩 | 0.9480 / 블렌드 개선 없음 | 미채택 |
| 6. 하이퍼파라미터 튜닝 | min_child_weight·subsample·colsample·reg_lambda 등 9개 조합 탐색 | 0.9499 (+0.0002) | 채택 (최종) |
| 7. FT-Transformer | PyTorch로 직접 구현(CPU 전용, fold-0만 검증) | 0.9492 (vs XGBoost fold-0 0.9503) | 미채택 |
| 8. 클래스별 타겟 인코딩(범주형) | 4개 범주형 피처 → one-vs-rest 타겟 인코딩 12개 피처 | 0.949767 (-0.000118) | 미채택 |
| 9. 클래스별 타겟 인코딩(전체 확장) | 11개 수치형 피처까지 확장(분위수 구간화, 총 60개 피처) | 0.946994 (-0.002891) | 미채택 |
| 10. 스태킹 앙상블 | XGBoost·CatBoost·RandomForest OOF 확률 → LR/XGBoost 메타모델(중첩 CV) | 0.94976~0.94979 | 미채택 |
| 11. stress 신호 극대화 | 단조성 제약 검토(비단조 관계 발견) → stress_level별 전문가 모델 | 0.918615 (-0.031270) | 미채택 |

*OOF: Out-of-Fold 예측 기준 5-fold 평균 Balanced Accuracy*

## 주요 인사이트

- **저-카디널리티 범주형 + 강한 트리 모델 조합에서는 구조를 바꾸는 접근(모델 다양화, 피처 재인코딩, 세그먼트 분리)보다 기존 모델의 정규화·하이퍼파라미터 미세조정이 더 안정적으로 통했습니다.** CatBoost·RandomForest·FT-Transformer 모두 XGBoost보다 약했고, 블렌딩·스태킹 모두 최적 가중치가 XGBoost 100%로 수렴했습니다.
- **타겟 인코딩류 기법은 이 데이터셋에서 일관되게 도움이 되지 않았습니다.** 범주형이 전부 3~4개 수준이라 트리 네이티브 분할이 이미 거의 최적의 정보를 뽑아내고 있었고, 수치형까지 확장하면 오히려 원본 정보 손실과 노이즈만 늘었습니다.
- **`stress_level`은 가장 중요한 피처였지만, 건강 상태와의 관계는 단조적이지 않았습니다.** at-risk 비율이 low(79.7%)→medium(99.4%)→high(71.8%)로 medium에서 정점을 찍고 다시 떨어지는 비단조 패턴이며, 오히려 "서로 다른 성격의 하위집단을 규정하는 범주형 변수"에 가까웠습니다. 이 인사이트에 기반해 시도한 stress 기반 전문가 모델도 실패했는데, 이는 이미 XGBoost가 이 정보를 자유롭게 최적으로 활용하고 있었기 때문으로 해석됩니다.
- **스태킹 메타 모델에는 베이스 모델과 동일한 클래스 불균형 보정을 반드시 함께 적용해야 합니다.** 이를 빠뜨리면 베이스 모델이 교정해둔 균형이 무너지며 Balanced Accuracy가 0.87~0.93대로 크게 하락할 수 있습니다.

## 실행 방법

```bash
pip install pandas numpy scikit-learn xgboost catboost torch matplotlib seaborn
jupyter notebook predicting_student_health_risk.ipynb
```

## 향후 고려사항

- Logistic Regression 등 선형 계열 모델 추가 실험
- 실제 Kaggle 리더보드 점수를 통한 OOF 대비 검증
- 피처 선택(feature selection)을 통한 모델 단순화 시도
