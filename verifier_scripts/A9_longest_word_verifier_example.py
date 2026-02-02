"""
Verifier checker for A9: Word with specific vowel/consonant constraints.

Validates that the model's answer adheres to all the specified rules:
- Contains exactly one 'o', one 'a', and one 'e'
- Has at least twice as many consonants as vowels
- One of the vowels is either the first or last letter

Scoring:
- 1 point if all rules are satisfied
- +1 point if the word is longer than 8 characters (9+)
- +1 point if the word is longer than 10 characters (11+)
"""

import re
import os

_ENGLISH_WORDS_CACHE = None


def verify_answer(model_answer: str) -> tuple[bool, str]:
    """
    Check if the model's answer satisfies all the constraints for A15.

    Scoring:
        - 1 point: All rules satisfied
        - +1 point: Word length > 9 characters
        - +1 point: Word length > 11 characters

    Args:
        model_answer: The model's response containing the proposed word

    Returns:
        Tuple of (is_valid, result_message)
        - is_valid: True if at least 1 point is earned, False otherwise
        - result_message: Contains scoring details in format "SCORE:X/3 | <details>"
    """
    if not model_answer or not model_answer.strip():
        return False, "SCORE:0/3 | Response is empty"

    # Extract the word from \boxed{answer} format
    pattern = r"\\boxed\{([^}]+)\}"
    matches = re.findall(pattern, model_answer)

    if not matches:
        return False, r"SCORE:0/3 | No \boxed{answer} pattern found in response"

    # Use the last match found in the answer
    candidate = matches[-1].strip()
    word = candidate.lower()

    # Rule 0: Check if the word exists in the English language
    global _ENGLISH_WORDS_CACHE
    if _ENGLISH_WORDS_CACHE is None:
        try:
            # Construct path relative to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            words_path = os.path.join(script_dir, "..", "media", "words.txt")
            
            with open(words_path, "r", encoding="utf-8") as f:
                # Load words into a set, converting to lowercase for case-insensitive matching
                _ENGLISH_WORDS_CACHE = {line.strip().lower() for line in f}
        except Exception as e:
            # Fallback or error if dictionary cannot be loaded
            # For now, we return valid but log error in message if critical
            # But robustly, we should probably fail or skip. 
            # Given instructions, we must check. If file missing, maybe fail?
            return False, f"SCORE:0/3 | Error loading English dictionary: {str(e)}"

    if word not in _ENGLISH_WORDS_CACHE:
        return False, f"SCORE:0/3 | Word '{word}' is not a valid English word according to the provided dictionary."

    # Rule 1: Contains exactly one 'o', one 'a', and one 'e'
    o_count = word.count('o')
    a_count = word.count('a')
    e_count = word.count('e')

    if o_count != 1:
        return False, f"SCORE:0/3 | Word '{word}' contains {o_count} 'o's, expected exactly 1"
    if a_count != 1:
        return False, f"SCORE:0/3 | Word '{word}' contains {a_count} 'a's, expected exactly 1"
    if e_count != 1:
        return False, f"SCORE:0/3 | Word '{word}' contains {e_count} 'e's, expected exactly 1"

    # Count vowels and consonants
    vowels = 'aeiou'
    vowel_count = sum(1 for char in word if char in vowels)
    consonant_count = len(word) - vowel_count

    # Rule 2: At least twice as many consonants as vowels
    if consonant_count < 2 * vowel_count:
        return False, (
            f"SCORE:0/3 | Word '{word}' has {consonant_count} consonants and {vowel_count} vowels. "
            f"Expected at least {2 * vowel_count} consonants (twice as many as vowels)"
        )

    # Rule 3: One of the vowels is either the first or last letter
    first_char = word[0]
    last_char = word[-1]

    if first_char not in vowels and last_char not in vowels:
        return False, (
            f"SCORE:0/3 | Word '{word}' does not have a vowel as the first or last letter. "
            f"First: '{first_char}', Last: '{last_char}'"
        )

    # All rules satisfied - calculate score based on word length
    word_len = len(word)
    score = 1  # Base score for meeting all rules

    if word_len > 8:
        score += 1
    if word_len > 10:
        score += 1

    result_msg = f"SCORE:{score}/3 | Word: '{word}' ({word_len} chars)"

    return True, result_msg