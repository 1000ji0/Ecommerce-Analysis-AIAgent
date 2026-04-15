# DAISY Ecommerce Analysis Agent — 테스트 시나리오

> 파일: `data/sample/ecommerce_sample.csv` (500행 × 14컬럼)  
> 실행: `python main.py --file data/sample/ecommerce_sample.csv`

---

## 데이터 컬럼 참고

| 컬럼 | 설명 |
|------|------|
| date | 날짜 |
| channel | 광고 채널 (검색광고/SNS광고/디스플레이/이메일) |
| ad_spend | 광고비 |
| impressions | 노출수 |
| clicks | 클릭수 |
| sessions | 세션수 |
| new_visitors | 신규 방문자 |
| add_to_cart | 장바구니 추가 |
| purchases | 구매 건수 |
| avg_order_value | 평균 주문 금액 |
| return_rate | 반품률 |
| satisfaction | 고객 만족도 |
| repeat_customers | 재구매 고객 수 |
| TARGET | 매출 (분석 타겟) |

---

## 시나리오 1 — 단일 명령 테스트 (AG-04)

AG-04의 각 task를 개별로 테스트한다.  
HITL 없이 바로 실행 → 결과 반환되어야 함.

```
👤  데이터 기본 정보 보여줘
```
**기대 결과:** shape, 결측값, 이상치 비율, 타겟 상관관계 top5 출력

```
👤  이상치 있는 컬럼 알려줘
```
**기대 결과:** 이상치 비율이 높은 컬럼 목록 출력

```
👤  TARGET이랑 상관관계 높은 변수 뭐야?
```
**기대 결과:** 상관계수 상위 5개 변수 출력

```
👤  피처 중요도 상위 5개 뽑아줘
```
**기대 결과:** Borda Count 기반 변수 중요도 순위 출력  
**사전 조건:** `brew install libomp` 완료 필요 (LightGBM)

```
👤  채널별 매출 분포 시각화해줘
```
**기대 결과:** 차트 PNG 파일 생성 경로 출력  
확인: `sessions/{session_id}/charts/` 폴더

```
👤  데이터에서 인사이트 뽑아줘
```
**기대 결과:** 핵심 인사이트 + 액션 아이템 목록 출력

---

## 시나리오 2 — 연속 대화 테스트 (컨텍스트 누적)

이전 분석 결과를 이어받아 대화가 이어지는지 확인한다.

```
👤  ad_spend 분포 보여줘
👤  이상치 비율 높은 컬럼만 따로 알려줘
👤  purchases랑 TARGET 상관관계 어때?
👤  위 결과로 인사이트 만들어줘
👤  보고서 만들어줘
```

**기대 결과:**  
- 각 명령마다 해당 분석만 실행  
- 마지막 보고서에 이전 분석 결과 반영  
- `sessions/{session_id}/reports/report_*.pdf` 생성

---

## 시나리오 3 — 전체 파이프라인 + HITL 4포인트 테스트

HITL 두 단계 구조 (Phase A 정보 수집 → Phase B 결과 검토) 전체 확인.

```
👤  전체 분석해줘
```

### HITL ① 계획 승인

**Phase A (자유 텍스트)**
```
🤖  이번 분석에서 특별히 집중하고 싶은 변수나 요구사항이 있으신가요?
> purchases와 sessions 중심으로 분석해줘
```

**Phase B (선택)**
```
선택: 1.승인 / 2.수정 / 3.재실행
> 1
```

### HITL ② 전처리 결과 확인

**Phase A (자유 텍스트)**
```
🤖  전처리 방식이나 제외하고 싶은 변수가 있으신가요?
> 이상치 제거 방법을 IQR로 바꿔줘
```

**Phase B (선택)**
```
선택: 1.승인 / 2.수정 / 3.재실행
> 2  (수정 테스트)
수정 내용: top_n을 10으로 늘려줘
```

### HITL ③ Feature 선정 확인

**Phase A (자유 텍스트)**
```
🤖  변수 선정 기준이나 제외하고 싶은 변수가 있으신가요?
> return_rate는 제외해줘
```

**Phase B (선택)**
```
선택: 1.승인 / 2.수정 / 3.재실행
> 3  (재실행 테스트)
```

↑ 재실행하면 HITL ① 부터 다시 시작되어야 함

```
→ 다시 HITL ① Phase A 부터 반복
> (이번엔 전부 승인으로 진행)
```

### HITL ④ 최종 보고서 승인

**Phase A (자유 텍스트)**
```
🤖  보고서에 추가하고 싶은 내용이나 형식 요청이 있으신가요?
> PDF로 깔끔하게 만들어줘
```

**Phase B (선택)**
```
선택: 1.승인 / 2.수정 / 3.재실행
> 1  (최종 승인)
```

**기대 결과:**  
- `sessions/{session_id}/reports/report_*.pdf` 생성  
- SQLite DB에 세션 완료 기록  
- LTM 저장 확인

---

## 시나리오 4 — 에지 케이스 테스트

의도 파악 및 fallback 동작 확인.

```
👤  안녕
👤  오늘 날씨 어때?
👤  데이터 몇 행이야?
👤  뭘 분석할 수 있어?
👤  ㅎㅇ
```

**기대 결과:**  
- 분석 요청 아님 → NONE 처리 → 에이전트가 자연스럽게 안내

```
👤  아이큐알로 이상치 제거해줘
👤  가우시안 방식으로 이상치 분석해줘
👤  상관계수 높은거 10개만 뽑아줘
```

**기대 결과:**  
- 동의어/구어체 표현도 AG-04로 올바르게 라우팅

---

## 시나리오 5 — HITL 중 종료 테스트

HITL 진행 중 exit 입력 시 정상 종료 확인.

```
👤  전체 분석해줘
(HITL ① Phase A 질문 나타남)
> exit
```

**기대 결과:**  
- 즉시 세션 종료  
- 에러 없이 프로세스 정상 종료

---

## 확인 체크리스트

```bash
# 로그 확인
cat logs/sessions/$(ls -t logs/sessions/ | head -1)/trace.md

# DB 확인
sqlite3 data/agent_trace.db "SELECT event_name, event_type, created_at FROM trace_events ORDER BY id DESC LIMIT 20;"

# 차트 확인
ls sessions/$(ls -t sessions/ | head -1)/charts/

# 보고서 확인
ls sessions/$(ls -t sessions/ | head -1)/reports/

# LangSmith 확인
open https://smith.langchain.com
```

---

## 커버리지 정리

| 항목 | 시나리오 | 체크 |
|------|---------|------|
| AG-01 의도 파악 | 1, 2, 3, 4 | |
| AG-01 fallback (NONE) | 4 | |
| AG-02 mock 실행 | 3 | |
| AG-04 EDA (T-12) | 1, 2 | |
| AG-04 Feature Importance (T-13) | 1, 3 | |
| AG-04 시각화 (T-19) | 1, 2 | |
| AG-04 인사이트 (T-14) | 1, 2 | |
| AG-05 보고서 (T-18) | 2, 3 | |
| HITL ① Phase A+B | 3 | |
| HITL ② Phase A+B | 3 | |
| HITL ③ Phase A+B + 재실행 | 3 | |
| HITL ④ Phase A+B + LTM 저장 | 3 | |
| HITL 수정 반영 | 3 | |
| HITL 중 exit 종료 | 5 | |
| T-20 로깅 (MD + SQLite) | 전체 | |
| LangSmith 추적 | 전체 | |