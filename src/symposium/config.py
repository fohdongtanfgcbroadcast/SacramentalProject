from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
METADATA_DIR = DATA_ROOT / "metadata"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

EMBEDDING_MODEL = "BAAI/bge-m3"

# claude CLI 호출 파라미터 — 단일 진실원. web._call_claude 가 여기서 import.
# opus 상속 시 60s 초과·구독 소진 이력으로 sonnet 명시 핀(2026-07-22).
CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_TIMEOUT = 90  # claude 호출 상한(초) — 워커 장기 점유 방지(콜드스타트 마진)

# RAG 관련성 소프트 임계(cosine distance). 최상위 hit 거리가 이 값을 넘으면
# 프롬프트에 '발췌가 무관할 수 있으니 인용을 지어내지 말라'는 주의를 주입한다.
# 실측 캘리브레이션: 관련 질의 ~0.28–0.43, 무관 질의 ~0.49–0.69 (bge-m3 cosine).
# 하드 필터가 아닌 소프트 신호 — 정상 답변의 recall 을 해치지 않기 위함.
RELEVANCE_SOFT_MAX = 0.55

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
