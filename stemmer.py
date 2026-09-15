"""A compact, self-contained English stemmer.

Not a full Porter Stemmer implementation — just handles the common cases
that matter for skill matching: -ing, -ed, -es, -s, with basic handling of
consonant doubling (debugging -> debug, not debugg) and y->i (studies ->
study). Deliberately dependency-free: no nltk, no download, no network
needed at all, since a heavy dependency already caused problems once.
"""

VOWELS = set("aeiou")


def stem(word):
    word = word.lower().strip()
    if len(word) <= 3:
        return word

    # -ies -> -y  (e.g. "studies" -> "study")
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    # -ing  (with consonant-doubling undo, e.g. "debugging" -> "debug")
    if word.endswith("ing") and len(word) > 5:
        stripped = word[:-3]
        if len(stripped) >= 3 and stripped[-1] == stripped[-2] and stripped[-1] not in VOWELS:
            return stripped[:-1]
        return stripped

    # -ed  (with consonant-doubling undo, e.g. "planned" -> "plan")
    if word.endswith("ed") and len(word) > 4:
        stripped = word[:-2]
        if len(stripped) >= 3 and stripped[-1] == stripped[-2] and stripped[-1] not in VOWELS:
            return stripped[:-1]
        return stripped

    # -es  (e.g. "processes" -> "process")
    if word.endswith("es") and len(word) > 4:
        return word[:-2]

    # -s  (plain plural, e.g. "tests" -> "test"; skip -ss like "process")
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]

    return word


def stems_match(word_a, word_b):
    """Compare two words by stem, with a fallback for silent-e plurals
    (e.g. "database" vs "databases" -> "database" vs "databas")."""
    stem_a, stem_b = stem(word_a), stem(word_b)
    if stem_a == stem_b:
        return True
    # try adding back a silent "e" on either side
    return stem_a + "e" == stem_b or stem_b + "e" == stem_a