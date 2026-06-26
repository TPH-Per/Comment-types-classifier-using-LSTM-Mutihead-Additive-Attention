import re
import unicodedata

# ---------------------------------------------------------------------------
# Teencode mapping (106 entries — matches training notebook)
# ---------------------------------------------------------------------------
TEENCODE = {
    "ko": "không", "k": "không", "kh": "không", "hok": "không",
    "khg": "không", "kg": "không", "khôngg": "không", "khôg": "không",
    "dc": "được", "dk": "được", "đc": "được", "duoc": "được",
    "đưoc": "được", "đươc": "được", "d": "được",
    "vs": "với", "v": "với", "voi": "với",
    "ntn": "như thế nào", "nthe": "như thế", "nth": "như thế",
    "mn": "mọi người", "mng": "mọi người", "m.n": "mọi người",
    "b": "bạn", "bn": "bạn", "bạnè": "bạn",
    "t": "tôi", "m": "mình", "mk": "mình",
    "cx": "cũng", "cung": "cũng", "cug": "cũng",
    "lm": "làm", "làm": "làm", "l": "làm",
    "sp": "sản phẩm", "s.phẩm": "sản phẩm",
    "ok": "tốt", "okie": "tốt", "okey": "tốt", "oki": "tốt",
    "good": "tốt", "gud": "tốt",
    "bad": "tệ", "bed": "tệ",
    "tks": "cảm ơn", "tk": "cảm ơn", "thanks": "cảm ơn", "thank": "cảm ơn",
    "cmt": "bình luận", "cmt": "comment",
    "ship": "giao hàng", "shipper": "người giao hàng",
    "shop": "cửa hàng", "store": "cửa hàng",
    "nv": "nhân viên", "ks": "khách sạn",
    "tp": "thành phố", "tphcm": "thành phố hồ chí minh",
    "sg": "sài gòn", "hn": "hà nội",
    "bt": "bình thường", "bth": "bình thường", "bthg": "bình thường",
    "z": "vậy", "zậy": "vậy", "zè": "vậy",
    "j": "gì", "gj": "gì", "jì": "gì",
    "ui": "ơi", "ới": "ơi", "oi": "ơi",
    "wa": "quá", "qua": "quá", "qá": "quá",
    "nhanh": "nhanh", "nhanh": "nhanh",
    "ak": "à", "àk": "à", "ah": "à", "a": "à",
    "uh": "ừ", "uhm": "ừ", "um": "ừ", "hmm": "ừ",
    "chau": "cháu", "ch": "cháu",
    "e": "em", "a": "anh",
    "ô": "ô", "ôii": "ơi",
    "iêu": "tiêu", "teu": "tiêu",
    "iêu cực": "tiêu cực", "tich cực": "tích cực",
    "r": "rồi", "roi": "rời", "rùi": "rồi",
    "nhìu": "nhiều", "nhieu": "nhiều", "nh": "nhiều",
    "đag": "đang", "dg": "đang", "dang": "đang",
    "h": "giờ", "hok": "không",
    "ms": "mới", "mới": "mới",
    "đt": "điện thoại", "dt": "điện thoại",
    "mk": "mình", "mik": "mình",
    "ny": "người yêu", "gấu": "người yêu",
    "cr": "crush", "ck": "chồng", "vk": "vợ",
    "gc": "góc", "gđ": "gia đình",
    "sv": "sinh viên", "hs": "học sinh",
    "gv": "giảng viên", "thầy": "thầy",
    "cô": "cô", "tks": "cảm ơn",
    "plz": "làm ơn", "pls": "làm ơn",
    "sry": "xin lỗi", "sr": "xin lỗi",
    "hp": "hạnh phúc", "bh": "bây giờ",
    "hqua": "hôm qua", "h.nay": "hôm nay",
    "kq": "kết quả", "tv": "tivi",
    "mxh": "mạng xã hội", "fb": "facebook",
    "ig": "instagram", "tt": "tiktok",
    "ytb": "youtube", "yt": "youtube",
    "mk": "mình", "cx": "cũng",
    "ntn": "như thế nào", "sao": "thế nào",
}

# ---------------------------------------------------------------------------
# Emoji mapping (59 symbols — matches training notebook)
# ---------------------------------------------------------------------------
EMOJI_MAP = {
    "❤️": " tim ", "❤": " tim ", "♥️": " tim ",
    "\U0001f60a": " vui ", "\U0001f60d": " yêu ", "\U0001f618": " hôn ",
    "\U0001f602": " cười ", "\U0001f923": " cười_lớn ",
    "\U0001f62d": " khóc ", "\U0001f622": " buồn ",
    "\U0001f621": " tức_giận ", "\U0001f620": " giận ",
    "\U0001f92f": " ngạc_nhiên ", "\U0001f631": " sợ ",
    "\U0001f60e": " cool ", "\U0001f914": " suy_nghĩ ",
    "\U0001f44d": " tốt ", "\U0001f44e": " tệ ",
    "\U0001f44f": " hoan_hô ", "\U0001f525": " hot ",
    "\U0001f4a5": " nổ ", "\U0001f44b": " vẫy_tay ",
    "\U0001f64f": " cầu_nguyện ", "\U0001f44f": " vỗ_tay ",
    "\U0001f389": " ăn_mừng ", "\U0001f38a": " party ",
    "⭐": " sao ", "\U0001f31f": " sao_sáng ",
    "\U0001f4af": " 100đ ", "\U0001f4a0": " kim_cương ",
    "\U0001f60f": " đểu ", "\U0001f612": " chán ",
    "\U0001f614": " buồn ", "\U0001f615": " phân_vân ",
    "\U0001f616": " đau ", "\U0001f61c": " đùa ",
    "\U0001f61e": " thất_vọng ", "\U0001f624": " bực ",
    "\U0001f625": " mệt ", "\U0001f628": " lo_lắng ",
    "\U0001f629": " mệt ", "\U0001f62b": " mệt ",
    "\U0001f62c": " cười ", "\U0001f630": " lo ",
    "\U0001f632": " bất_ngờ ", "\U0001f633": " xấu_hổ ",
    "\U0001f634": " ngủ ", "\U0001f635": " choáng ",
    "\U0001f637": " bệnh ", "\U0001f911": " giàu ",
    "\U0001f913": " nerd ", "\U0001f917": " ôm ",
    "\U0001f91d": " bắt_tay ", "\U0001f920": " cowboy ",
    "\U0001f921": " hề ", "\U0001f922": " say ",
    "\U0001f924": " thèm ", "\U0001f925": " nói_dối ",
    "\U0001f927": " hắt_xì ", "\U0001f929": " star_eyes ",
    "\U0001f92a": " điên ", "\U0001f92b": " im ",
    "\U0001f92c": " chửi ", "\U0001f92d": " cười ",
    "\U0001f92e": " nôn ", "\U0001f9d0": " lạ ",
}

# ---------------------------------------------------------------------------
# Punctuation rules — normalize repeated punctuation
# ---------------------------------------------------------------------------
PUNCT_RULES = [
    (r"\.{2,}", " ... "),
    (r"!{2,}", " ! "),
    (r"\?{2,}", " ? "),
    (r",,", " , "),
]

# ---------------------------------------------------------------------------
# Safe stopwords (kept minimal — preserve negation words like "không", "chưa")
# ---------------------------------------------------------------------------
SAFE_STOPWORDS = {
    "và", "của", "là", "cho", "với", "các", "một", "để", "từ", "trong",
    "có", "được", "này", "cũng", "như", "nhưng", "hay", "hoặc", "nếu",
    "thì", "lại", "đã", "sẽ", "đang", "về", "theo", "sau", "trên",
    "dưới", "giữa", "bằng", "mà", "khi", "nên", "do", "tuy", "vì",
    "nữa", "rất", "thật", "quá", "lắm", "nhỉ", "ạ", "ơi", "à",
    "ừ", "uh", "vâng", "dạ", "nha", "nhé", "hả",
}

# ---------------------------------------------------------------------------
# Regex patterns compiled once
# ---------------------------------------------------------------------------
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_HTML = re.compile(r"<[^>]+>")
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
_RE_NUMBER = re.compile(r"\b\d+[.,]?\d*\b")
_RE_SPECIAL = re.compile(r"[^\w\sÀ-ɏḀ-ỿ.,!?;:\-\"']")
_RE_MULTI_SPACE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# 10-step preprocessing pipeline (matches training notebook)
# ---------------------------------------------------------------------------

def _normalize_unicode(text: str) -> str:
    """Step 1: Unicode normalization (NFC)."""
    return unicodedata.normalize("NFC", text)


def _text_normalize(text: str) -> str:
    """Step 2: Normalize Vietnamese diacritics."""
    # Fix common encoding issues
    replacements = {
        "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
        "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
        "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _map_emojis(text: str) -> str:
    """Step 3: Replace emojis with Vietnamese tokens."""
    for emoji, token in EMOJI_MAP.items():
        text = text.replace(emoji, token)
    return text


def _apply_punct_rules(text: str) -> str:
    """Step 4: Normalize punctuation patterns."""
    for pattern, repl in PUNCT_RULES:
        text = re.sub(pattern, repl, text)
    return text


def _remove_html_url(text: str) -> str:
    """Step 5: Remove HTML tags, URLs, emails."""
    text = _RE_HTML.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)
    return text


def _lowercase(text: str) -> str:
    """Step 6: Lowercase (after emoji mapping)."""
    return text.lower()


def _expand_teencode(text: str) -> str:
    """Step 7: Expand teencode abbreviations."""
    words = text.split()
    expanded = []
    for w in words:
        lower_w = w.lower().strip(".,!?;:")
        if lower_w in TEENCODE:
            expanded.append(TEENCODE[lower_w])
        else:
            expanded.append(w)
    return " ".join(expanded)


def _word_tokenize(text: str) -> str:
    """Step 8: Vietnamese word segmentation with underthesea."""
    from underthesea import word_tokenize as _tokenize
    return _tokenize(text, format="text")


def _filter_stopwords(text: str) -> str:
    """Step 9: Remove safe stopwords (preserve negation)."""
    words = text.split()
    return " ".join(w for w in words if w.lower() not in SAFE_STOPWORDS)


def _final_cleanup(text: str) -> str:
    """Step 10: Remove special characters and normalize whitespace."""
    text = _RE_SPECIAL.sub(" ", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    return text.strip()


def clean_text_v2(text: str) -> str:
    """Full 10-step preprocessing pipeline matching the training notebook.

    Returns cleaned text as a space-separated string of tokens.
    The caller should .split() the result to get token list.
    """
    text = _normalize_unicode(text)
    text = _text_normalize(text)
    text = _map_emojis(text)
    text = _apply_punct_rules(text)
    text = _remove_html_url(text)
    text = _lowercase(text)
    text = _expand_teencode(text)
    text = _word_tokenize(text)
    text = _filter_stopwords(text)
    text = _final_cleanup(text)
    return text
