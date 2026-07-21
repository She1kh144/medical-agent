import re
from dataclasses import dataclass

SELF_HARM_PATTERNS = [
    r"не\s+проснуться",
    r"(?:уснуть|заснуть)\s+навсегда",
    r"покончить\s+с\s+собой",
    r"свести\s+счеты",
    r"(?:смертельн|летальн)\w*\s+доз",
    r"(?:чтобы|хочу)\s+умереть",
]

RED_FLAG_PATTERNS = [
    r"дав(?:ит|ление)\s+в\s+груди",
    r"бол\w*\s+в\s+груди",
    r"отдает\s+в\s+(?:лев\w*|руку|челюсть|лопатку)",
    r"отека\w*\s+(?:лицо|горло|язык|гортань)",
    r"отек\s+(?:лица|горла|языка|гортани|квинке)",
    r"(?:тяжело|трудно)\s+дышать",
    r"задыха\w*",
    r"передозировк\w*",
    r"выпил\w*\s+(?:всю|целую)\s+(?:пачку|упаковку)",
    r"(?:потерял\w*|без)\s+сознани\w*",
]

CHILD_TERMS = re.compile(
    r"реб[е]нк\w*|дет(?:ям|ск\w*|и\b|ей\b)|доч\w*|сын\w*|малыш\w*|грудничк\w*|подростк\w*"
)

DOSE_INTENT = re.compile(
    r"доз\w*|сколько|дать|давать|рассчита\w*|таблетк\w*|капл\w*|сироп\w*|\bмг\b|приним\w*|пить|выпить"
)

AGE_YEARS = re.compile(r"(\d{1,2})\s*(?:лет|год(?:а|у|ик\w*)?)\b")
AGE_MONTHS = re.compile(r"\d{1,2}\s*месяц")
WEIGHT_KG = re.compile(r"(\d{1,3})\s*кг")

ADULT_AGE = 18
CHILD_WEIGHT_KG = 40

GUARD_MESSAGES = {
    "self_harm": (
        "Я не могу помочь с этим вопросом, но мне важно, чтобы вы получили поддержку. "
        "Пожалуйста, поговорите со специалистом.\n\n"
        "Экстренные службы: 112\n\n"
        "Ваш вопрос передан фармацевту."
    ),
    "red_flag": (
        "Описанные симптомы могут указывать на неотложное состояние. "
        "Не подбирайте лекарство самостоятельно — сразу звоните в скорую помощь: 103 или 112.\n\n"
        "Ваш вопрос передан фармацевту."
    ),
    "pediatric_dosing": (
        "Я не подбираю лекарства и их количество для детей и подростков — это должен "
        "делать врач или фармацевт. Ваш вопрос передан специалисту.\n\n"
        "Это не медицинская консультация, обратитесь к врачу."
    ),
}

@dataclass
class GuardTrigger:
    category: str
    reason: str
    matched: str

def normalize(text: str) -> str:
    return text.lower().replace("ё", "е")

def is_child_context(text: str) -> str | None:
    """Returns the matched child signal, or None."""
    match = CHILD_TERMS.search(text)

    if match:
        return match.group(0)

    for age_match in AGE_YEARS.finditer(text):
        if int(age_match.group(1)) < ADULT_AGE:
            return age_match.group(0)

    if AGE_MONTHS.search(text):
        return AGE_MONTHS.search(text).group(0) # type: ignore

    for weight_match in WEIGHT_KG.finditer(text):
        if int(weight_match.group(1)) <= CHILD_WEIGHT_KG and re.search(r"доз\w*|рассчита\w*", text):
            return weight_match.group(0)

    return None

def check_input(question: str) -> GuardTrigger | None:
    """Deterministic input guard. Returns a trigger to escalate, or None to proceed."""
    text = normalize(question)

    for pattern in SELF_HARM_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return GuardTrigger(
                category="self_harm",
                reason="Запрос связан с намеренным причинением вреда себе.",
                matched=match.group(0),
            )

    for pattern in RED_FLAG_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return GuardTrigger(
                category="red_flag",
                reason="Описаны симптомы возможного неотложного состояния.",
                matched=match.group(0),
            )

    child_signal = is_child_context(text)

    if child_signal and DOSE_INTENT.search(text):
        return GuardTrigger(
            category="pediatric_dosing",
            reason="Вопрос о приеме лекарства ребенком или подростком.",
            matched=child_signal,
        )

    return None