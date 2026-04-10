import os
import base64
import json
import tempfile
import re
from typing import List, Optional

import requests as _requests
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Volunteer Coordination API — Gemini Enhanced")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DUMMY_NEEDS = [
    {
        "location": "Downtown Shelter, 123 Main St",
        "category": "shelter",
        "urgency": 5,
        "description": "Urgent need for blankets and cots — overflow expected tonight due to storm.",
    },
    {
        "location": "Riverside Community Center",
        "category": "food",
        "urgency": 4,
        "description": "Running low on canned goods and fresh produce for weekend distribution.",
    },
    {
        "location": "Eastside Clinic, 45 Oak Ave",
        "category": "medical",
        "urgency": 5,
        "description": "Seeking nurses and paramedics to assist with flu vaccine clinic.",
    },
    {
        "location": "Public Library, North Branch",
        "category": "education",
        "urgency": 2,
        "description": "After-school tutoring needed for grades 3-6 on weekday afternoons.",
    },
    {
        "location": "West End Food Bank",
        "category": "food",
        "urgency": 3,
        "description": "Volunteers needed to sort and pack weekly food boxes for 200 families.",
    },
]


def _decode_firebase_credentials(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for extra_pad in ("", "=", "==", "==="):
        try:
            decoded = base64.b64decode(raw + extra_pad)
            return json.loads(decoded.decode("utf-8"))
        except Exception:
            pass
    for extra_pad in ("", "=", "==", "==="):
        try:
            decoded = base64.urlsafe_b64decode(raw + extra_pad)
            return json.loads(decoded.decode("utf-8"))
        except Exception:
            pass
    raise RuntimeError(
        "Could not decode FIREBASE_CREDENTIALS. Expected raw JSON or base64-encoded JSON."
    )


def _init_firebase():
    raw = os.environ.get("FIREBASE_CREDENTIALS")
    if not raw:
        raise RuntimeError("FIREBASE_CREDENTIALS environment variable not set")
    cred_dict = _decode_firebase_credentials(raw)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cred_dict, f)
        tmp_path = f.name
    cred = credentials.Certificate(tmp_path)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    return firestore.client()


try:
    db = _init_firebase()
    _firebase_error = None
    print("Firebase connected successfully.")
except Exception as e:
    db = None
    _firebase_error = str(e)
    import sys
    print(f"WARNING: Firebase init failed: {e}", file=sys.stderr)


def _build_gemini_config():
    base_url = os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY", "")
    if base_url:
        # Replit proxy — no version prefix in path
        url = f"{base_url}/models/gemini-2.5-flash:generateContent"
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        return url, headers
    google_key = os.environ.get("GOOGLE_API_KEY", "")
    if google_key:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        headers = {"x-goog-api-key": google_key, "Content-Type": "application/json"}
        return url, headers
    return None, None


_gemini_url, _gemini_headers = _build_gemini_config()
_gemini_error = None if _gemini_url else "No Gemini credentials configured."
if _gemini_url:
    print("Gemini client ready.")
else:
    import sys
    print(f"WARNING: {_gemini_error}", file=sys.stderr)


def _gemini_generate(prompt: str, image_bytes: bytes | None = None, mime_type: str = "image/jpeg") -> str:
    """Call Gemini 2.5 Flash and return the raw text response."""
    if not _gemini_url:
        raise RuntimeError(_gemini_error)

    if image_bytes:
        b64_img = base64.b64encode(image_bytes).decode()
        contents = [{
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": mime_type, "data": b64_img}},
                {"text": prompt},
            ],
        }]
    else:
        contents = [{"role": "user", "parts": [{"text": prompt}]}]

    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": 4096,
        },
    }
    resp = _requests.post(_gemini_url, json=payload, headers=_gemini_headers, timeout=45)
    resp.raise_for_status()
    data = resp.json()

    candidate = data.get("candidates", [{}])[0]
    finish = candidate.get("finishReason", "")
    parts = candidate.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)

    if finish == "MAX_TOKENS" and not text:
        raise RuntimeError("Gemini hit token limit and returned no text.")

    return text


def _extract_json(raw: str, expect_array: bool = False) -> object:
    """Robustly extract and parse JSON from a Gemini response.

    Strategy (applied in order until one succeeds):
      1. Strip markdown code fences, then direct json.loads
      2. Regex-find the outermost [...] or {...} block and parse that
      3. Re-ask won't help here, so raise with the cleaned raw text for logging
    """
    # ── 1. Strip markdown fences ─────────────────────────────────────────
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # ── 2. Regex extraction of first complete JSON structure ─────────────
    pattern = r"(\[[\s\S]*\])" if expect_array else r"(\{[\s\S]*\})"
    match = re.search(pattern, cleaned)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # If the response is truncated mid-array, try to close it
    if expect_array:
        # Find the last complete object in an array
        objects = re.findall(r"\{[^{}]*\}", cleaned)
        if objects:
            try:
                return json.loads("[" + ",".join(objects) + "]")
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Cannot extract valid JSON from Gemini response. Raw (first 300 chars): {cleaned[:300]}")


_seeded = False


def _seed_if_empty():
    global _seeded
    if db is None or _seeded:
        return
    try:
        col = db.collection("needs")
        docs = list(col.limit(1).stream())
        if not docs:
            for need in DUMMY_NEEDS:
                col.add(need)
            print(f"Seeded {len(DUMMY_NEEDS)} dummy needs into Firestore.")
        _seeded = True
    except Exception as e:
        import sys
        print(f"Seed warning: {e}", file=sys.stderr)


def _require_db():
    if db is None:
        raise HTTPException(status_code=503, detail=f"Firebase not initialized. {_firebase_error}")
    _seed_if_empty()


_GEOCODE_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")


def _geocode(location: str) -> tuple[float, float]:
    """Return (lat, lng) for a location string using Google Geocoding API.
    Falls back to (0.0, 0.0) if unavailable."""
    if not _GEOCODE_KEY or not location.strip():
        return (0.0, 0.0)
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        resp = _requests.get(url, params={"address": location, "key": _GEOCODE_KEY}, timeout=5)
        data = resp.json()
        if data.get("status") == "OK":
            loc = data["results"][0]["geometry"]["location"]
            return (loc["lat"], loc["lng"])
    except Exception:
        pass
    return (0.0, 0.0)


def _require_gemini():
    if not _gemini_url:
        raise HTTPException(status_code=503, detail=f"Gemini not configured. {_gemini_error}")


class NeedIn(BaseModel):
    location: str = Field(..., min_length=1)
    category: str = Field(..., pattern="^(food|medical|shelter|education)$")
    urgency: int = Field(..., ge=1, le=5)
    description: str = Field(..., min_length=1)


class NeedOut(BaseModel):
    id: str
    location: str
    category: str
    urgency: int
    description: str


class MatchResult(BaseModel):
    id: str
    location: str
    category: str
    urgency: int
    description: str
    reason: str
    lat: float = 0.0
    lng: float = 0.0


class ScanRequest(BaseModel):
    image_b64: str
    mime_type: str = "image/jpeg"


class ScanResult(BaseModel):
    location: str
    category: str
    description: str
    urgency: int = 3
    raw_text: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "firebase": "connected" if db is not None else f"error: {_firebase_error}",
        "gemini": "ready" if _gemini_url else f"error: {_gemini_error}",
    }


@app.post("/needs", response_model=NeedOut, status_code=201)
def create_need(need: NeedIn):
    _require_db()
    data = need.model_dump()
    _, doc_ref = db.collection("needs").add(data)
    return NeedOut(id=doc_ref.id, **data)


@app.get("/needs", response_model=List[NeedOut])
def list_needs():
    _require_db()
    docs = (
        db.collection("needs")
        .order_by("urgency", direction=firestore.Query.DESCENDING)
        .stream()
    )
    return [NeedOut(id=doc.id, **doc.to_dict()) for doc in docs]


@app.get("/match", response_model=List[MatchResult])
def match_needs(skills: str = ""):
    _require_db()
    _require_gemini()

    if not skills.strip():
        raise HTTPException(status_code=400, detail="skills query param required")

    # Fetch 5 most urgent needs from Firestore
    docs = (
        db.collection("needs")
        .order_by("urgency", direction=firestore.Query.DESCENDING)
        .limit(5)
        .stream()
    )
    needs = [{"id": doc.id, **doc.to_dict()} for doc in docs]

    if not needs:
        return []

    needs_text = "\n".join(
        f"{i+1}. [ID:{n['id']}] Category:{n['category']} | Urgency:{n['urgency']}/5 | "
        f"Location:{n['location']} | Description:{n['description']}"
        for i, n in enumerate(needs)
    )

    prompt = f"""You are a volunteer coordination assistant. Match a volunteer to the top 3 community needs.

Volunteer skills: "{skills}"

Available needs:
{needs_text}

Return ONLY valid JSON. Do not include explanation or extra text outside the JSON.

Return a JSON array of exactly 3 objects ranked best-to-worst match:
[
  {{"need": "<need_id>", "reason": "<1-2 sentences why this is a great fit>", "urgency": <urgency_number>}},
  {{"need": "<need_id>", "reason": "<1-2 sentences>", "urgency": <urgency_number>}},
  {{"need": "<need_id>", "reason": "<1-2 sentences>", "urgency": <urgency_number>}}
]

Rules:
- "need" must be the exact ID string from the list above
- "urgency" must be the integer from the need's Urgency field
- If fewer than 3 needs exist, return only the available ones
- Output ONLY the JSON array, nothing else"""

    _MATCH_FALLBACK = [
        {"need": n["id"], "reason": "Best available match for your skills.", "urgency": n.get("urgency", 3)}
        for n in needs[:3]
    ]

    try:
        raw = _gemini_generate(prompt)
        ranked = _extract_json(raw, expect_array=True)
        if not isinstance(ranked, list) or len(ranked) == 0:
            ranked = _MATCH_FALLBACK
    except Exception as e:
        import sys
        print(f"Gemini match parse error: {e}", file=sys.stderr)
        ranked = _MATCH_FALLBACK

    needs_by_id = {n["id"]: n for n in needs}
    results = []
    for item in ranked[:3]:
        # Accept both "need" (new format) and "id" (legacy format)
        nid = item.get("need") or item.get("id", "")
        reason = item.get("reason", "Best match for your skills.")
        need = needs_by_id.get(nid)
        if need:
            lat, lng = _geocode(need.get("location", ""))
            results.append(MatchResult(
                reason=reason,
                id=nid,
                lat=lat,
                lng=lng,
                **{k: need[k] for k in ("location", "category", "urgency", "description")},
            ))

    # Guarantee at least something is returned even if IDs didn't match
    if not results:
        for n in needs[:3]:
            lat, lng = _geocode(n.get("location", ""))
            results.append(MatchResult(
                reason="Recommended based on current urgency.",
                id=n["id"],
                lat=lat,
                lng=lng,
                **{k: n[k] for k in ("location", "category", "urgency", "description")},
            ))

    return results


@app.post("/scan", response_model=ScanResult)
def scan_survey(req: ScanRequest):
    _require_gemini()

    try:
        image_bytes = base64.b64decode(req.image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    prompt = """You are processing a handwritten volunteer coordination survey form.

Carefully read the handwritten text in this image and extract the following fields.

Return ONLY valid JSON. Do not include explanation or extra text outside the JSON.

Output this exact JSON structure:
{
  "location": "<where help is needed — address or place name, or empty string if not visible>",
  "category": "<exactly one of: food, medical, shelter, education>",
  "description": "<clear description of what help is needed>",
  "urgency": <integer 1-5 based on language urgency: 5=critical/emergency, 3=moderate, 1=low>,
  "raw_text": "<full verbatim transcription of all handwritten text in the image>"
}

Rules:
- category must be exactly one of: food, medical, shelter, education
- urgency must be an integer between 1 and 5
- If the image has no readable text, still return valid JSON with empty strings and urgency 3
- Output ONLY the JSON object, nothing else"""

    _SCAN_FALLBACK = ScanResult(
        location="",
        category="food",
        description="",
        urgency=3,
        raw_text="",
    )

    valid_categories = {"food", "medical", "shelter", "education"}

    try:
        raw = _gemini_generate(prompt, image_bytes=image_bytes, mime_type=req.mime_type)
        data = _extract_json(raw, expect_array=False)
        if not isinstance(data, dict):
            return _SCAN_FALLBACK
    except Exception as e:
        import sys
        print(f"Gemini scan parse error: {e}", file=sys.stderr)
        return _SCAN_FALLBACK

    category = str(data.get("category", "food")).lower().strip()
    if category not in valid_categories:
        category = "food"

    try:
        urgency = int(data.get("urgency", 3))
        urgency = max(1, min(5, urgency))
    except (TypeError, ValueError):
        urgency = 3

    return ScanResult(
        location=str(data.get("location", "")),
        category=category,
        description=str(data.get("description", "")),
        urgency=urgency,
        raw_text=str(data.get("raw_text", "")),
    )
