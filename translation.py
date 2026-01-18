"""
translation.py - Rule-based Sign-glish to English translation

Simple rule‑based mapper that converts a list of sign tokens into natural English.
This implementation is deliberately lightweight and does not require any external services.
"""

from typing import List

# ==================== TRANSLATION MAP ====================
# Comprehensive mapping dictionary for ISL to English

_TRANSLATION_MAP = {
    # Greetings
    "HELLO": "Hello",
    "HI": "Hi",
    "GOODBYE": "Goodbye",
    "BYE": "Bye",
    
    # Politeness
    "HELP": "Help",
    "PLEASE": "Please",
    "THANK": "Thank you",
    "THANKS": "Thank you",
    "SORRY": "Sorry",
    "WELCOME": "Welcome",
    
    # Pronouns
    "I": "I",
    "ME": "me",
    "YOU": "you",
    "WE": "we",
    "THEY": "they",
    
    # Basic responses
    "YES": "yes",
    "NO": "no",
    "OK": "okay",
    "GOOD": "good",
    "BAD": "bad",
    
    # Actions
    "WANT": "want",
    "NEED": "need",
    "GO": "go",
    "COME": "come",
    "STOP": "stop",
    "SIT": "sit",
    "STAND": "stand",
    "WALK": "walk",
    "RUN": "run",
    "EAT": "eat",
    "DRINK": "drink",
    "SLEEP": "sleep",
    
    # Objects
    "WATER": "water",
    "FOOD": "food",
    "HOME": "home",
    "SCHOOL": "school",
    "WORK": "work",
    
    # Time
    "MORNING": "good morning",
    "AFTERNOON": "good afternoon",
    "EVENING": "good evening",
    "NIGHT": "good night",
    "TODAY": "today",
    "TOMORROW": "tomorrow",
    "YESTERDAY": "yesterday",
    "NOW": "now",
    
    # Emotions
    "HAPPY": "happy",
    "SAD": "sad",
    "ANGRY": "angry",
    "SURPRISED": "surprised",
    
    # Questions
    "HOW": "how",
    "WHAT": "what",
    "WHERE": "where",
    "WHEN": "when",
    "WHY": "why",
    "WHO": "who",
    
    # Misc
    "ARE": "are",
    "IS": "is",
    "LOVE": "love",
}

def translate_signs(sign_list: List[str]) -> str:
    """Translate a list of sign tokens into a readable English sentence.

    Args:
        sign_list: List of uppercase sign tokens.
    
    Returns:
        A string with the translated sentence.
    
    Example:
        >>> translate_signs(["HELLO", "I", "WANT", "WATER"])
        "Hello I want water"
    """
    if not sign_list:
        return ""
    
    translated_words = []
    
    for token in sign_list:
        # Handle fingerspelled words
        if token.startswith("SPELL_"):
            # Extract the spelled word
            word = token.replace("SPELL_", "").lower()
            translated_words.append(word)
        else:
            # Look up in translation map
            translated = _TRANSLATION_MAP.get(token.upper(), token.lower())
            translated_words.append(translated)
    
    # Join words
    sentence = " ".join(translated_words).strip()
    
    # Capitalize first letter
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    
    return sentence

def add_translation(sign_token: str, english_text: str):
    """Add a new translation to the map.
    
    Args:
        sign_token: The ISL sign token (uppercase)
        english_text: The English translation
    """
    _TRANSLATION_MAP[sign_token.upper()] = english_text

def get_all_translations() -> dict:
    """Get all available translations.
    
    Returns:
        Dictionary of all sign-to-English mappings
    """
    return _TRANSLATION_MAP.copy()

def translate_with_grammar(sign_list: List[str]) -> str:
    """Translate with basic grammar improvements.
    
    Args:
        sign_list: List of sign tokens
    
    Returns:
        Grammatically improved sentence
    """
    sentence = translate_signs(sign_list)
    
    # Basic grammar improvements
    # Add articles
    sentence = sentence.replace(" water", " some water")
    sentence = sentence.replace(" food", " some food")
    
    # Fix common patterns
    sentence = sentence.replace("I want", "I would like")
    sentence = sentence.replace("I need", "I need to")
    
    return sentence

# ==================== MAIN ====================

if __name__ == "__main__":
    # Demo examples
    print("=" * 60)
    print("ISL Sign-to-English Translation Demo")
    print("=" * 60)
    print()
    
    # Test cases
    test_cases = [
        ["HELLO", "HOW", "ARE", "YOU"],
        ["I", "WANT", "WATER", "PLEASE"],
        ["THANK", "YOU"],
        ["GOOD", "MORNING"],
        ["I", "NEED", "HELP"],
        ["SORRY", "I", "AM", "SAD"],
    ]
    
    for signs in test_cases:
        translation = translate_signs(signs)
        print(f"Signs: {' '.join(signs)}")
        print(f"Translation: {translation}")
        print()
    
    print("=" * 60)