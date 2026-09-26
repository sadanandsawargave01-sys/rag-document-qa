import re
from functools import lru_cache


SUPPORTED_LANGUAGE_CODES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
}

FALLBACK_MESSAGES = {
    "en": "Sorry, I could not find relevant information in the provided documents.",
    "hi": "क्षमा करें, मुझे प्रदान किए गए दस्तावेज़ों में इस बारे में जानकारी नहीं मिली।",
    "mr": "क्षमा करा, मला प्रदान केलेल्या दस्तऐवजांमध्ये याबद्दल माहिती आढळली नाही.",
}

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

_MARATHI_MARKERS = {
    "आहे","आहेत","नाही","काय","का","कसे","कशी","कोण",
    "मला","तुम्ही","तुमचा","तुमची","म्हणजे","मध्ये","याबद्दल",
    "सांगा","द्या","करा","झाले","होते","आढळली","दस्तऐवज",
    "कुठे","कोठे","जन्म","जन्मले","झाला","झाली","महाराज"
}

_HINDI_MARKERS = {
    "है","हैं","नहीं","क्या","क्यों","कैसे","कौन","मुझे","आप",
    "आपका","आपकी","मतलब","में","इसके","बारे",
    "बताइए","बताओ","दीजिए","करें","हुआ","थे",
    "मिली","दस्तावेज़",
}


@lru_cache(maxsize=1)
def _get_lingua_detector():
    try:
        from lingua import Language, LanguageDetectorBuilder

        return (
            LanguageDetectorBuilder.from_languages(
                Language.ENGLISH,
                Language.HINDI,
                Language.MARATHI,
            )
            .with_preloaded_language_models()
            .build()
        )
    except Exception:
        return None


def normalize_language_code(language: str | None) -> str:
    if not language:
        return "en"

    value = str(language).strip().lower()

    if value.startswith("mr") or value in {"marathi", "मराठी"}:
        return "mr"
    if value.startswith("hi") or value in {"hindi", "हिन्दी", "हिंदी"}:
        return "hi"
    if value.startswith("en") or value == "english":
        return "en"

    return "en"


def _script_based_language(text: str) -> str | None:
    if not text or not _DEVANAGARI_RE.search(text):
        return None

    words = set(re.findall(r"[\u0900-\u097F]+", text))
    marathi_score = len(words & _MARATHI_MARKERS)
    hindi_score = len(words & _HINDI_MARKERS)

    if marathi_score > hindi_score:
        return "mr"
    if hindi_score > marathi_score:
        return "hi"

    # Marathi uses ळ frequently; Hindi generally does not.
    if "ळ" in text:
        return "mr"

    return None


def detect_query_language(text: str | None, default: str = "en") -> str:
    if not text or not str(text).strip():
        return normalize_language_code(default)

    value = str(text).strip()

    script_lang = _script_based_language(value)

    detector = _get_lingua_detector()
    if detector is not None:
        try:
            detected = detector.detect_language_of(value)
            lingua_lang = normalize_language_code(getattr(detected, "iso_code_639_1", detected).name)
            if script_lang and lingua_lang in {"hi", "mr"} and lingua_lang != script_lang:
                return script_lang
            return lingua_lang
        except Exception:
            pass

    if script_lang:
        return script_lang

    try:
        from langdetect import detect

        return normalize_language_code(detect(value))
    except Exception:
        return normalize_language_code(default)


def language_name(language: str | None) -> str:
    return SUPPORTED_LANGUAGE_CODES.get(normalize_language_code(language), "English")


def get_fallback_message(language: str | None) -> str:
    return FALLBACK_MESSAGES.get(normalize_language_code(language), FALLBACK_MESSAGES["en"])
