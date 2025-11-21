from piper.phonemize_espeak import EspeakClause, EspeakPhonemizer, EspeakSentence


def test_phonemize_zero() -> None:
    """Sanity check for phonemizer zero case."""
    phonemizer = EspeakPhonemizer()
    assert phonemizer.phonemize("en-us", "") == []


def test_phonemize() -> None:
    """Sanity check for phonemizer."""
    phonemizer = EspeakPhonemizer()
    assert phonemizer.phonemize("en-us", "test") == [
        EspeakSentence([EspeakClause("test", list("tˈɛst"))]),
    ]


def test_phonemize_multiple_chunks() -> None:
    """Check phonemizer with multiple chunks."""
    phonemizer = EspeakPhonemizer()
    assert phonemizer.phonemize("en-us", "Hello, world. Hey there!") == [
        EspeakSentence(
            [
                EspeakClause("Hello, ", list("həlˈoʊ, ")),
                EspeakClause("world. ", list("wˈɜːld.")),
            ]
        ),
        EspeakSentence(
            [
                EspeakClause("Hey there!", list("hˈeɪ ðˈɛɹ!")),
            ]
        ),
    ]
