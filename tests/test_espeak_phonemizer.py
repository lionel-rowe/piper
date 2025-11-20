from piper.phonemize_espeak import EspeakClause, EspeakPhonemizer, EspeakSentence


def test_phonemize_zero() -> None:
    """Sanity check for phonemizer zero case."""
    phonemizer = EspeakPhonemizer()
    assert phonemizer.phonemize("en-us", "") == []


def test_phonemize() -> None:
    """Sanity check for phonemizer."""
    phonemizer = EspeakPhonemizer()
    assert phonemizer.phonemize("en-us", "test") == [
        EspeakSentence([EspeakClause("test", ["t", "ˈ", "ɛ", "s", "t"])]),
    ]


def test_phonemize_multiple_chunks() -> None:
    """Check phonemizer with multiple chunks."""
    phonemizer = EspeakPhonemizer()
    assert phonemizer.phonemize("en-us", "Hello, world. Hey there!") == [
        EspeakSentence(
            [
                EspeakClause("Hello, ", ["h", "ə", "l", "ˈ", "o", "ʊ", ",", " "]),
                EspeakClause("world. ", ["w", "ˈ", "ɜ", "ː", "l", "d", "."]),
            ]
        ),
        EspeakSentence(
            [
                EspeakClause(
                    "Hey there!", ["h", "ˈ", "e", "ɪ", " ", "ð", "ˈ", "ɛ", "ɹ", "!"]
                ),
            ]
        ),
    ]
