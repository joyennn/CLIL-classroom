# main.py
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Optional, Tuple
from datetime import datetime
import os, uvicorn, traceback, re, json, random

# ─────────────────────────────────────────────────────────────
# Anthropic Python SDK (Claude)
#   pip install anthropic fastapi uvicorn pydantic
# ─────────────────────────────────────────────────────────────
try:
    from anthropic import Anthropic
except Exception as e:
    Anthropic = None
    print("[BOOT] Anthropic SDK import 실패:", e)

# ─────────────────────────────────────────────────────────────
# 경로/기본 설정
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(BASE_DIR, "index.html")
PERSONA_JSON_PATH = os.path.join(BASE_DIR, "persona.json")

# ─────────────────────────────────────────────────────────────
# persona.json 로더 (견고)
#   - 단일 객체 { Class_information, Students } 또는
#     "상단 객체 + 하단 배열" 하이브리드 형태 모두 지원
# ─────────────────────────────────────────────────────────────
def load_persona_json(path: str) -> Tuple[Dict, List[Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    def try_load_all(text: str):
        # (1) 단일 JSON 객체 시도
        try:
            data = json.loads(text)
            class_info = data.get("Class_information", {})
            students = data.get("Students", [])
            # 객체 내부의 다른 "학생 배열"이 있으면 합치기
            for k, v in list(data.items()):
                if k != "Students" and isinstance(v, list) and all(isinstance(x, dict) and x.get("student_id") for x in v):
                    students += v
            return class_info, students
        except Exception:
            pass

        # (2) 상단 객체 + 하단 배열 분리
        split_points = []
        for tok in ["}\n[", "}\r\n[", "}["]:
            idx = text.find(tok)
            if idx != -1:
                split_points.append((idx, tok))
        if not split_points:
            raise ValueError("persona.json 파싱 실패: 지원하지 않는 형식")

        idx, tok = sorted(split_points, key=lambda x: x[0])[0]
        head = text[: idx + 1]                 # '}' 포함
        tail = text[idx + (len(tok) - 1):]     # '['부터

        data_head = json.loads(head)
        arr_tail = json.loads(tail)
        class_info = data_head.get("Class_information", {})
        students = data_head.get("Students", [])
        if isinstance(arr_tail, list):
            students += arr_tail
        else:
            raise ValueError("persona.json tail이 배열이 아님")
        return class_info, students

    return try_load_all(raw)

try:
    CLASS_INFO, STUDENTS = load_persona_json(PERSONA_JSON_PATH)
    # student_id 중복 제거(마지막 항목 우선)
    seen = {}
    for s in STUDENTS:
        sid = s.get("student_id")
        if sid:
            seen[sid] = s
    STUDENTS = list(seen.values())
    print(f"[BOOT] Loaded {len(STUDENTS)} students from persona.json")
except Exception as e:
    print("[BOOT] persona.json 로드 실패:", e)
    CLASS_INFO, STUDENTS = {}, []

# ─────────────────────────────────────────────────────────────
# FastAPI 앱
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="Persona Classroom Chat (Claude)", version="4.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일(이미지 등)
if os.path.isdir(os.path.join(BASE_DIR, "static")):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ─────────────────────────────────────────────────────────────
# 모델들
# ─────────────────────────────────────────────────────────────
Role = Literal["system", "user", "assistant"]

class Message(BaseModel):
    role: Role
    content: str

class VerifyRequest(BaseModel):
    api_key: str = Field(min_length=10)

class ChatRequest(BaseModel):
    api_key: str = Field(min_length=10)
    conversation: List[Message]
    model: str = "claude-3-5-sonnet-20240620"
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_turns: int = Field(default=16, ge=2, le=64)
    # 장면 구성
    active_persona_ids: List[str] = []
    responders_this_turn: Optional[List[str]] = None
    # 언어 규칙
    lang: Literal["auto", "ko", "en"] = "auto"
    # 디버그(옵션): True면 요청/응답 요약 포함
    debug: bool = False

# ─────────────────────────────────────────────────────────────
# 유틸: 학생 요약/선택/가드
# ─────────────────────────────────────────────────────────────
def summarize_student(s: Dict) -> str:
    kp = (s.get("korean_profile") or {})
    lis = (kp.get("listening") or {}).get("level")
    spk = (kp.get("speaking") or {}).get("level")
    rdg = (kp.get("reading") or {}).get("level")
    wtg = (kp.get("writing") or {}).get("level")
    motive = (((s.get("Affective_factor") or {}).get("Motivation") or {}).get("Degree"))
    traits = ", ".join((s.get("personality_traits") or [])[:3])
    fav = ", ".join((s.get("academic_profile") or {}).get("favorite_subjects", [])[:2])
    strug = ", ".join((s.get("academic_profile") or {}).get("struggling_subjects", [])[:2])
    return (
        f"- [{s.get('name')} / {s.get('student_id')} / {s.get('country_of_origin')}]: "
        f"KOR L/S/R/W={lis}/{spk}/{rdg}/{wtg}, 동기={motive}, 성향={traits}"
        + (f", 선호={fav}" if fav else "")
        + (f", 어려움={strug}" if strug else "")
    )

def pick_scene_students(active_ids: List[str]) -> List[Dict]:
    if not active_ids:
        return STUDENTS[:]  # 전체
    aset = set(active_ids)
    return [s for s in STUDENTS if s.get("student_id") in aset]

def choose_responders(auto_list: Optional[List[str]], scene_students: List[Dict], recent_user_turn: bool) -> List[str]:
    if auto_list:
        return auto_list
    k = 3 if recent_user_turn else 2
    ids = [s.get("student_id") for s in scene_students if s.get("student_id")]
    if len(ids) <= k:
        return ids
    random.seed()
    return random.sample(ids, k)

def language_rule(lang: str) -> str:
    if lang == "ko":
        return "수업 언어는 한국어이며, 학생들의 국적과 한국 거주 기간에 따라 한국어 수준이 다를 수 있다."
    if lang == "en":
        return "Language used in the class is English, but the proficiency levels depend on students' nationality and residence period in Korea."
    return ("사용자의 '마지막 user 메시지' 언어(한국어/영어)를 따르되, "
            "혼용 입력일 경우 해당 메시지의 언어로 각각 응답한다.")

def build_system_prompt(scene_students: List[Dict], lang_text: str) -> str:
    roster_text = "\n".join(summarize_student(s) for s in scene_students)
    return f"""
당신은 교실 시뮬레이션을 운영한다.
사용자는 '선생님'이고, 당신은 '학생들'을 연기한다.

[최상위 언어 규칙]
{lang_text}

[형식 규칙]
- 각 발화는 반드시 `[이름]: 내용` 형식.
- 한 턴에 0~2명만 말하기. 전원 동시 발화 금지.
- 같은 턴에 여러 명이 답할 수도, 아무도 안 할 수도 있음.
- 이름을 부르면 그 학생이 우선 답함.
- 선생님의 발화에 대한 응답 뿐 아니라, 학생들끼리 대화를 나눌 수도 있음.

[발화 규칙]
- 학생들은 모두 8세, 초등 2학년으로 이에 맞는 발화를 유지.
- 학생들의 국적과 한국 거주 기간에 따라 한국어 수준에 차이가 있으며, 자주 발화의 오류가 발생함. 때로는 모국어를 사용하기도 함.
- 학생별 성격, 특징, 한국어 능력, 과목 선호에 따라 응답하도록 함.
- 학생들은 수업 내용을 처음으로 배우며, 교과 내용이나 어휘에 어려움을 느낄 수 있음. 

[학생 정보]
{roster_text}
""".strip()

def post_guard(reply: str, scene_students: List[Dict]) -> str:
    """모델 출력을 최대 3줄로 제한 + 교사 발화 제거.
       (빈 결과면 0줄 허용하도록 빈 문자열 반환)"""
    line_pat = re.compile(r'^(?:\[(?P<spk1>.+?)\]|(?P<spk2>[^:\[\]]+))\s*:\s*(?P<msg>.*)$')
    out_lines, seen = [], set()
    for raw in (reply or "").splitlines():
        st = raw.strip()
        if not st:
            continue
        m = line_pat.match(st)
        if not m:
            out_lines.append(st)
            if len(out_lines) >= 3:
                break
            continue
        speaker = (m.group('spk1') or m.group('spk2') or '').strip()
        msg = (m.group('msg') or '').strip()
        if not msg:
            continue
        if speaker in {"선생님", "교사", "Teacher", "teacher"}:
            continue
        if speaker in seen:
            continue
        seen.add(speaker)
        out_lines.append(f"[{speaker}]: {msg}")
        if len(out_lines) >= 3:
            break
    cleaned = "\n".join(out_lines).strip()
    if cleaned:
        return cleaned
    # 0명 응답 허용 (빈 문자열)
    return ""

# ─────────────────────────────────────────────────────────────
# Claude 메시지 변환/정규화
# ─────────────────────────────────────────────────────────────
def to_claude_messages(conversation: List[Message]) -> List[Dict]:
    """
    - system은 제외 (system= 인자로 전달)
    - user → assistant → user … 번갈이 되도록 정규화
    - 같은 role 연속이면 병합
    - 맨 앞이 assistant면 제거
    """
    raw = [{"role": m.role, "text": m.content} for m in conversation if m.role in ("user", "assistant")]

    # 맨 앞 assistant 제거
    while raw and raw[0]["role"] != "user":
        raw.pop(0)

    # 같은 role 병합
    merged = []
    for m in raw:
        if not merged:
            merged.append(m); continue
        if merged[-1]["role"] == m["role"]:
            merged[-1]["text"] += "\n" + m["text"]
        else:
            merged.append(m)

    # 번갈이 강제 (assistant가 중간에 연속되면 스킵)
    normalized = []
    expect = "user"
    for m in merged:
        if m["role"] != expect:
            if expect == "user" and m["role"] == "assistant":
                continue
        normalized.append(m)
        expect = "assistant" if expect == "user" else "user"

    if not normalized:
        normalized = [{"role": "user", "text": "수업을 시작합니다. 학생들만 발화하세요."}]

    return [
        {"role": m["role"], "content": [{"type": "text", "text": m["text"]}]}
        for m in normalized
    ]

# ─────────────────────────────────────────────────────────────
# 라우트
# ─────────────────────────────────────────────────────────────
# Render 헬스체크: HEAD / 를 200으로 응답
@app.head("/")
async def head_root():
    return Response(status_code=200)

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    try:
        with open(HTML_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # 파일이 없어도 200으로 간단한 안내 반환 (헬스체크 안정화)
        return HTMLResponse("<h3>Persona Classroom Chat API</h3>", status_code=200)

@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

@app.get("/personas")
async def list_personas():
    def kor_levels(s):
        kp = (s.get("korean_profile") or {})
        return {
            "L": (kp.get("listening") or {}).get("level"),
            "S": (kp.get("speaking") or {}).get("level"),
            "R": (kp.get("reading") or {}).get("level"),
            "W": (kp.get("writing") or {}).get("level"),
        }
    return {
        "class_info": CLASS_INFO,
        "students": [
            {
                "student_id": s.get("student_id"),
                "name": s.get("name"),
                "country": s.get("country_of_origin"),
                "photo_url": s.get("photo_url") or "",
                "kor": kor_levels(s),
                "motivation": (((s.get("Affective_factor") or {}).get("Motivation") or {}).get("Degree")),
                "traits": (s.get("personality_traits") or [])[:5],
                "favorite": (s.get("academic_profile") or {}).get("favorite_subjects", [])[:3],
                "struggle": (s.get("academic_profile") or {}).get("struggling_subjects", [])[:3],
            }
            for s in STUDENTS
        ],
    }

@app.post("/verify-key")
async def verify_key(req: VerifyRequest):
    if Anthropic is None:
        raise HTTPException(status_code=500, detail="Anthropic SDK 미설치/비호환. `pip install -U anthropic`")
    key = (req.api_key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="API 키가 비어 있습니다.")
    try:
        client = Anthropic(api_key=key)
        _ = client.models.list()
        return {"ok": True}
    except Exception as e:
        print("[/verify-key ERROR]", e.__class__.__name__, ":", str(e))
        traceback.print_exc()
        msg = str(e).lower()
        if "unauthorized" in msg or "invalid" in msg or "authentication" in msg:
            raise HTTPException(status_code=401, detail="API 키가 올바르지 않습니다.")
        if "permission" in msg or "insufficient" in msg or "quota" in msg:
            raise HTTPException(status_code=403, detail="권한/쿼터 문제입니다. 결제/한도를 확인하세요.")
        if "not found" in msg or "404" in msg:
            raise HTTPException(status_code=404, detail="엔드포인트/모델을 찾지 못했습니다.")
        if "rate" in msg and "limit" in msg:
            raise HTTPException(status_code=429, detail="요청 한도 초과")
        if "ssl" in msg or "certificate" in msg:
            raise HTTPException(status_code=503, detail="SSL/인증서 문제로 연결 실패")
        if "timed out" in msg or "timeout" in msg or "connection" in msg:
            raise HTTPException(status_code=503, detail="네트워크 연결 문제로 Anthropic API 접근 불가")
        raise HTTPException(status_code=500, detail="알 수 없는 오류. 서버 로그를 확인하세요.")

@app.post("/chat")
async def chat(req: ChatRequest):
    if Anthropic is None:
        raise HTTPException(status_code=500, detail="Anthropic SDK 미설치/비호환.")
    key = (req.api_key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="API 키가 비어 있습니다.")
    if not req.conversation:
        raise HTTPException(status_code=400, detail="대화 내용이 비어 있습니다.")

    client = Anthropic(api_key=key)

    scene_students = pick_scene_students(req.active_persona_ids)
    if not scene_students:
        raise HTTPException(status_code=400, detail="장면에 포함할 학생이 없습니다. active_persona_ids를 확인하세요.")

    # 언어 규칙 + 시스템 프롬프트
    lang_text = language_rule(req.lang)
    system_prompt = build_system_prompt(scene_students, lang_text)

    # 최근 대화 tail 구성 → Claude 규격으로 변환/정규화
    tail = [m for m in req.conversation][-req.max_turns:]
    claude_messages = to_claude_messages(tail)

    MAX_OUTPUT_TOKENS = 512

    try:
        completion = client.messages.create(
            model=req.model,
            temperature=req.temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=system_prompt,
            messages=claude_messages,
        )
        parts = completion.content or []
        raw_reply = "".join([p.text for p in parts if getattr(p, "type", "") == "text"])
        safe_reply = post_guard(raw_reply, scene_students)

        resp = {"reply": safe_reply}
        if req.debug:
            resp["debug"] = {
                "first_message_role": claude_messages[0]["role"] if claude_messages else None,
                "messages_len": len(claude_messages),
                "system_len": len(system_prompt),
                "model": req.model,
                "temperature": req.temperature,
            }
        return JSONResponse(resp)

    except Exception as e:
        # 콘솔에 상세 스택
        print("[/chat ERROR]", e.__class__.__name__, ":", str(e))
        traceback.print_exc()
        # 에러 유형 분류 + 원문 노출
        msg = str(e).lower()
        if any(x in msg for x in ["bad request", "must start with", "message role", "invalid", "malformed"]):
            raise HTTPException(status_code=400, detail=f"요청 형식 오류(Claude Messages): {e}")
        if "unauthorized" in msg or "api key" in msg or "authentication" in msg:
            raise HTTPException(status_code=401, detail="API 키 인증 실패")
        if "permission" in msg or "insufficient" in msg or "quota" in msg:
            raise HTTPException(status_code=403, detail="권한/쿼터 문제")
        if "rate" in msg and "limit" in msg:
            raise HTTPException(status_code=429, detail="요청 한도 초과")
        # 나머지: 게이트웨이 오류로 반환(원인 노출)
        raise HTTPException(status_code=502, detail=f"모델 호출 실패: {e}")

if __name__ == "__main__":
    # Render는 PORT 환경변수를 제공하므로 반드시 0.0.0.0:$PORT 로 바인딩
    port = int(os.environ.get("PORT", 10000))  # 로컬: 8000
    uvicorn.run("main:app", host="0.0.0.0", port=port)
