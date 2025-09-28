SPACY_LANGUAGES = [
    "ca",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "hr",
    "it",
    "lt",
    "mk",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sl",
    "sv",
    "uk",
    "zh",
]

CODE2LANG = {
    "ar": "Arabic",
    "ca": "Catalan",
    "cs": "Czech",
    "de": "German",
    "zd": "Swiss German",
    "gsw": "Swiss German",
    "el": "Greek",
    "en": "English",
    # "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    # "fr": "French",
    # "he": "Hebrew",
    # "hi": "Hindi",
    "hr": "Croatian",
    "it": "Italian",
    "kl": "Kalaallisut",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "rm": "Romansh",
    "ro": "Romanian",
    "ru": "Russian",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sv": "Swedish",
    "tr": "Turkish",
    "uk": "Ukrainian",
    # "yue": "Cantonese",
    "zh": "Chinese",
}

LANGUAGES = list(CODE2LANG.keys())


LANG_COLORS = {
    # --- Germanic (red hues) ---
    "en": "#fca5a5",  # English
    "de": "#f87171",  # German
    "gsw": "#f43f5e",  # Swiss German
    #"zd": "#f43f5e",  # Swiss German
    "nl": "#ef4444",  # Dutch
    "sv": "#dc2626",  # Swedish
    # --- Romance (amber/orange hues) ---
    "ca": "#fde047",  # Catalan
    "es": "#facc15",  # Spanish
    "fr": "#fbbf24",  # French
    "it": "#f59e0b",  # Italian
    "pt": "#d97706",  # Portuguese
    "ro": "#b45309",  # Romanian
    "rm": "#78350f",  # Romansh
    # --- Slavic (emerald/green hues) ---
    "sl": "#A2EAD5",  # Slovenian
    "hr": "#6ee7b7",  # Croatian
    "mk": "#34d399",  # Macedonian
    "pl": "#10b981",  # Polish
    "cs": "#059669",  # Czech
    "ru": "#064e3b",  # Russian
    "uk": "#065f46",  # Ukrainian
    # --- Baltic (violet hues) ---
    "lt": "#c084fc",  # Lithuanian
    "lv": "#a855f7",  # Latvian
    # --- Finno-Ugric ---
    "et": "#4ade80",  # Estonian
    # --- Greek ---
    "el": "#3b82f6",  # Greek
    # --- Turkic ---
    "tr": "#f97316",  # Turkish
    # --- Semitic (rose hues) ---
    "ar": "#fb7185",  # Arabic
    "he": "#f43f5e",  # Hebrew
    # --- Indo-Aryan ---
    "hi": "#eab308",  # Hindi
    # --- Sino-Tibetan (cyan hues) ---
    "zh": "#22d3ee",  # Mandarin Chinese
    "yue": "#06b6d4",  # Cantonese
    # --- Basque (indigo) ---
    "eu": "#6366f1",  # Basque
    # --- Eskimo–Aleut ---
    "kl": "#2dd4bf",  # Kalaallisut (Greenlandic)
}

LANG_ORDER = [lang for lang in list(LANG_COLORS.keys()) if lang in LANGUAGES]
