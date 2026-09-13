import re


def parse_srt(srt_text: str) -> list[dict]:
    segments = []
    if not srt_text or not srt_text.strip():
        return segments

    blocks = [b.strip() for b in srt_text.strip().split("\n\n") if b.strip()]
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines()]
        if len(lines) < 3:
            continue
        time_line = lines[1]
        if " --> " not in time_line:
            continue
        start_raw, end_raw = time_line.split(" --> ", 1)
        segments.append(
            {
                "start": _to_seconds(start_raw),
                "end": _to_seconds(end_raw),
                "text": "\n".join(lines[2:]).strip(),
            }
        )
    return segments


def to_srt(segments: list[dict], max_gap_ms: float = 100.0) -> str:
    lines = []
    max_gap_s = max_gap_ms / 1000.0
    for idx, seg in enumerate(segments, 1):
        lines.append(str(idx))
        end_time = seg['end']
        
        # Close small gaps to next segment
        if idx < len(segments):
            next_seg = segments[idx]
            gap = next_seg['start'] - end_time
            if 0 < gap <= max_gap_s:
                end_time = next_seg['start']
        
        lines.append(f"{format_timestamp(seg['start'])} --> {format_timestamp(end_time)}")
        lines.append((seg.get("text") or "").strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def clone_with_texts(segments: list[dict], texts: list[str], provider: str, polished: bool = False) -> list[dict]:
    cloned = []
    for seg, text in zip(segments, texts):
        cloned.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": (text or "").strip(),
                "source_text": seg.get("source_text") or seg.get("text", ""),
                "provider": provider,
                "polished": polished,
            }
        )
    return cloned


def format_timestamp(seconds: float) -> str:
    total_ms = int(round(float(seconds) * 1000))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    sec = total_seconds % 60
    total_minutes = total_seconds // 60
    mins = total_minutes % 60
    hrs = total_minutes // 60
    return f"{hrs:02d}:{mins:02d}:{sec:02d},{ms:03d}"


def _to_seconds(raw: str) -> float:
    raw = raw.strip().replace(",", ".")
    parts = raw.split(":")
    if len(parts) != 3:
        return 0.0
    hrs, mins, secs = parts
    return int(hrs) * 3600 + int(mins) * 60 + float(secs)


def split_text_batches(texts: list[str], batch_size: int) -> list[list[str]]:
    return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]


def validate_texts(texts: list[str], expected_len: int) -> bool:
    if len(texts) != expected_len:
        return False
    return all(isinstance(text, str) and text.strip() for text in texts)


def parse_numbered_line_items(raw: str) -> list[tuple[int, str]]:
    """Parse numbered model output while retaining the original cue IDs."""
    # Strip Gemma chain-of-thought tags
    cleaned = re.sub(r"<thought>.*?</thought>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    # Strip everything before first numbered line (thinking prefix)
    first_num = re.search(r"^\s*\d+\.", cleaned, re.MULTILINE)
    if first_num:
        cleaned = cleaned[first_num.start():]
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    items = []
    pattern = re.compile(r"^\s*(\d+)\.\s*(.*?)(?=^\s*\d+\.\s*|\Z)", re.MULTILINE | re.DOTALL)
    for match in pattern.finditer(cleaned):
        body = str(match.group(2) or "").strip()
        if not body:
            continue
        body_lines = []
        for raw_line in body.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith(("Assistant:", "Translation:", "Trợ lý:", "Dịch:", "Note:", "Here", "Sure", "OK", "Let", "I'll", "The")):
                continue
            body_lines.append(stripped)
        normalized = " ".join(body_lines).strip()
        while True:
            nested = re.match(r"^\s*\d+\.\s*(.+?)\s*$", normalized)
            if not nested:
                break
            candidate = nested.group(1).strip()
            if not candidate or candidate == normalized:
                break
            normalized = candidate
        if normalized:
            items.append((int(match.group(1)), normalized))

    if items:
        return items

    fallback_items = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("Assistant:", "Translation:", "Trợ lý:", "Dịch:", "Note:", "Here", "Sure", "OK", "Let", "I'll", "The")):
            continue
        match = re.match(r"^\s*\d+\.\s*(.+?)\s*$", line)
        if match:
            candidate = match.group(1).strip()
            while True:
                nested = re.match(r"^\s*\d+\.\s*(.+?)\s*$", candidate)
                if not nested:
                    break
                inner = nested.group(1).strip()
                if not inner or inner == candidate:
                    break
                candidate = inner
            fallback_items.append((int(match.group(1)), candidate))
    return fallback_items


def parse_numbered_lines(raw: str) -> list[str]:
    """Backward-compatible text-only parser for numbered model output."""
    return [text for _number, text in parse_numbered_line_items(raw)]


def condense_dialogue_for_timeline(text: str, duration: float) -> str:
    """Condense dialogue/subtitle text when its length exceeds natural speech duration for its timeline segment."""
    val = " ".join(str(text or "").replace("\n", " ").split()).strip()
    if not val or duration <= 0:
        return val
    words = val.split()
    budget = max(3, int(round(duration * 3.2)))
    if len(words) <= budget:
        return val

    result = val

    # 1. Condense reporting/narrative lead-in clauses before dialogue
    patterns = [
        (
            r"([A-Za-zÀ-ỹ0-9_]+(?:\s+[A-Za-zÀ-ỹ0-9_]+){0,3})\s+(?:chẳng bận tâm|không thèm để ý|không bận tâm|không chút do dự|không hề do dự|không khỏi cảm thán|vô cùng ngạc nhiên|kinh ngạc|đầy tự tin|tự tin|tức giận|bất lực|hốt hoảng|lo lắng|nghĩ một lúc|suy nghĩ một lát|không cho là đúng)\s*[,，]?\s*(?:đáp lại|trả lời rằng|hét lên rằng|la lên rằng|cảm thán rằng|mở miệng nói rằng|mở miệng nói|nói rằng|đáp|nói):\s*",
            r"\1 đáp: ",
        ),
        (
            r"([A-Za-zÀ-ỹ0-9_]+(?:\s+[A-Za-zÀ-ỹ0-9_]+){0,3})\s+(?:không khỏi mỉm cười|không khỏi cười|mỉm cười|cười khẩy|cười lớn|bật cười)\s*[,，]?\s*(?:nói rằng|đáp lại|nói):\s*",
            r"\1 cười nói: ",
        ),
        (
            r"([A-Za-zÀ-ỹ0-9_]+(?:\s+[A-Za-zÀ-ỹ0-9_]+){0,3})\s+(?:tức giận|bực bội|giận dữ)\s*[,，]?\s*(?:quát|hét|nói):\s*",
            r"\1 quát: ",
        ),
        (
            r"([A-Za-zÀ-ỹ0-9_]+(?:\s+[A-Za-zÀ-ỹ0-9_]+){0,3})\s+(?:nghi ngờ|thắc mắc|tò mò)\s*[,，]?\s*(?:hỏi lại|hỏi rằng|lên tiếng hỏi):\s*",
            r"\1 hỏi: ",
        ),
        (
            r"([A-Za-zÀ-ỹ0-9_]+(?:\s+[A-Za-zÀ-ỹ0-9_]+){0,3})\s+(?:đã\s+)?(?:lên tiếng cảnh báo|cảnh báo anh trai rằng|cảnh báo rằng):\s*",
            r"\1 cảnh báo: ",
        ),
        (
            r"([A-Za-zÀ-ỹ0-9_]+(?:\s+[A-Za-zÀ-ỹ0-9_]+){0,3})\s+(?:nhanh chóng|vội vàng|vội vã)\s*[,，]?\s*(?:lên tiếng|nói):\s*",
            r"\1 vội nói: ",
        ),
        (
            r"([A-Za-zÀ-ỹ0-9_]+(?:\s+[A-Za-zÀ-ỹ0-9_]+){0,3})\s+(?:liền|trực tiếp|bèn|bèn mở miệng|lại tiếp tục)\s*[,，]?\s*(?:nói rằng|đáp rằng|hỏi rằng):\s*",
            r"\1 nói: ",
        ),
    ]
    for pat, rep in patterns:
        new_val = re.sub(pat, rep, result, flags=re.IGNORECASE)
        if new_val != result:
            result = " ".join(new_val.split()).strip()
            if len(result.split()) <= budget:
                return result

    # 2. Phrase contractions in dialogue
    contractions = [
        (r"\blàm sao bắn trúng được chứ\b", "sao bắn trúng nổi"),
        (r"\blàm sao mà\b", "sao"),
        (r"\blàm sao\b", "sao"),
        (r"\bkhông thể nào\b", "không thể"),
        (r"\bnhư thế này\b", "thế này"),
        (r"\bnhư thế đó\b", "thế đó"),
        (r"\bngay lập tức\b", "ngay"),
        (r"\bhoàn toàn không\b", "không hề"),
        (r"\bchắc chắn sẽ\b", "sẽ"),
        (r"\bcó thể sẽ\b", "có thể"),
        (r"\brốt cuộc là\b", "rốt cuộc"),
        (r"\bthật sự là\b", "thực sự"),
        (r"\bthì ra là\b", "hóa ra"),
        (r"\bmột cách\s+", ""),
        (r"\bvề cơ bản\b", ""),
        (r"\bcó thể nói là\b", ""),
    ]
    for p, r in contractions:
        result = re.sub(p, r, result, flags=re.IGNORECASE)
        result = " ".join(result.split()).strip()
        if len(result.split()) <= budget:
            return result

    # 3. Filler word removal if still over budget
    tokens = result.split()
    if len(tokens) > budget:
        filler_words = {"thì", "mà", "đó", "này", "ấy", "vậy", "luôn", "rồi", "đang", "đã", "khá", "rất"}
        filtered = []
        for t in tokens:
            clean_t = re.sub(r"[^\wÀ-ỹ]", "", t).lower()
            if clean_t in filler_words and len(tokens) - len(filtered) > 1:
                continue
            filtered.append(t)
        if len(filtered) >= budget:
            result = " ".join(filtered)
            if len(result.split()) <= budget:
                return result

    # 4. Graceful trimming keeping speaker prefix if present
    cur_tokens = result.split()
    if len(cur_tokens) > budget:
        if ":" in result:
            parts = result.split(":", 1)
            speaker = parts[0].strip() + ": "
            dialogue_tokens = parts[1].strip().split()
            avail = max(2, budget - len(speaker.split()))
            punct = "?" if "?" in parts[1] else ("!" if "!" in parts[1] else ".")
            result = speaker + " ".join(dialogue_tokens[:avail]).strip(" ,;:") + punct
        else:
            punct = "?" if "?" in result else ("!" if "!" in result else ".")
            result = " ".join(cur_tokens[:budget]).strip(" ,;:") + punct

    return result
