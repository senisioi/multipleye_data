SPACY_LANGUAGES = [
    "ca",
    "de",
    "da",
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
    "da": "Danish",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "ha": "Hausa",
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
    "ur": "Urdu",
    "yue": "Cantonese",
    "zh": "Chinese",
}

LANGUAGES = list(CODE2LANG.keys())


LANG_COLORS = {
    # --- Germanic (red hues) ---
    "en": "#fca5a5",  # English
    "de": "#f87171",  # German
    "gsw": "#f43f5e",  # Swiss German
    "zd": "#f43f5e",  # Swiss German (alt code)
    "nl": "#ef4444",  # Dutch
    "da": "#dc2626",  # Danish
    "sv": "#ad1b1b",  # Swedish

    # --- Basque (indigo unique) ---
    "eu": "#6366f1",  # Basque

    # --- Romance (amber/orange hues) ---
    "ca": "#fde047",  # Catalan
    "es": "#facc15",  # Spanish
    "pt": "#fbbf24",  # Portuguese
    "it": "#f59e0b",  # Italian
    "fr": "#d97706",  # French
    "rm": "#b45309",  # Romansh
    "ro": "#78350f",  # Romanian

    # --- Slavic (emerald/green hues) ---
    "sl": "#A2EAD5",  # Slovenian
    "hr": "#6ee7b7",  # Croatian
    "mk": "#34d399",  # Macedonian
    "pl": "#10b981",  # Czech
    "cs": "#059669",  # Polish
    "uk": "#065f46",  # Ukrainian
    "ru": "#064e3b",  # Russian

    # --- Baltic (violet hues) ---
    "lt": "#c084fc",  # Lithuanian
    "lv": "#a855f7",  # Latvian

    # --- Finno-Ugric ---
    "et": "#4ade80",  # Estonian

    # --- Albanian (standalone, teal/blue-green family) ---
    "sq": "#0d9488",  # Albanian

    # --- Greek ---
    "el": "#3b82f6",  # Greek

    # --- Turkic ---
    "tr": "#f97316",  # Turkish

    # --- Afro-Asiatic (rose hues) ---
    "ar": "#fb7185",  # Arabic
    "he": "#f43f5e",  # Hebrew
    "ha": "#be123c",  # Hausa 

    # --- Indo-Aryan ---
    "hi": "#ea79ec",  # Hindi
    "ur": "#c026d3",  # Urdu

    # --- Sino-Tibetan (cyan hues) ---
    "zh": "#22d3ee",  # Mandarin Chinese
    "yue": "#06b6d4",  # Cantonese

    # --- Eskimo–Aleut ---
    "kl": "#2dd4bf",  # Kalaallisut (Greenlandic)

}

LANG_ORDER = [lang for lang in list(LANG_COLORS.keys())]
