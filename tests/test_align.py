from enum import IntFlag, auto

from piper.align import CharAlignment, PhonemeAlignment, align
from piper.phonemize_espeak import EspeakClause, EspeakPhonemizer, EspeakSentence


class Phoneme(IntFlag):
    BOS = auto()
    EOS = auto()

    COMMA = auto()
    SPACE = auto()
    EXCLAMATION = auto()

    STRESS = auto()
    LONG = auto()

    a = auto()
    æ = auto()
    ɐ = auto()
    ɑ = auto()
    b = auto()
    d = auto()
    ð = auto()
    e = auto()
    ə = auto()
    ɜ = auto()
    h = auto()
    ɪ = auto()
    l = auto()
    m = auto()
    n = auto()
    o = auto()
    r = auto()
    ɹ = auto()
    s = auto()
    t = auto()
    u = auto()
    ʊ = auto()
    w = auto()

    U_200D = auto()


def test_align_basic():
    espeak_sentence = EspeakSentence(
        [
            EspeakClause("Hello, ", ["h", "ə", "l", "ˈ", "o", "ʊ", ",", " "]),
            EspeakClause("world!", ["w", "ˈ", "ɜ", "ː", "l", "d", "!"]),
        ]
    )

    phoneme_alignments = [
        PhonemeAlignment(phoneme="^", phoneme_ids=[-1], num_samples=Phoneme.BOS),
        PhonemeAlignment(phoneme="h", phoneme_ids=[-1], num_samples=Phoneme.h),
        PhonemeAlignment(phoneme="ə", phoneme_ids=[-1], num_samples=Phoneme.ə),
        PhonemeAlignment(phoneme="l", phoneme_ids=[-1], num_samples=Phoneme.l),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="o", phoneme_ids=[-1], num_samples=Phoneme.o),
        PhonemeAlignment(phoneme="ʊ", phoneme_ids=[-1], num_samples=Phoneme.ʊ),
        PhonemeAlignment(phoneme=",", phoneme_ids=[-1], num_samples=Phoneme.COMMA),
        PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
        PhonemeAlignment(phoneme="w", phoneme_ids=[-1], num_samples=Phoneme.w),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="ɜ", phoneme_ids=[-1], num_samples=Phoneme.ɜ),
        PhonemeAlignment(phoneme="ː", phoneme_ids=[-1], num_samples=Phoneme.LONG),
        PhonemeAlignment(phoneme="l", phoneme_ids=[-1], num_samples=Phoneme.l),
        PhonemeAlignment(phoneme="d", phoneme_ids=[-1], num_samples=Phoneme.d),
        PhonemeAlignment(
            phoneme="!", phoneme_ids=[-1], num_samples=Phoneme.EXCLAMATION
        ),
        PhonemeAlignment(phoneme="$", phoneme_ids=[-1], num_samples=Phoneme.EOS),
    ]

    aligned = list(align(phoneme_alignments, espeak_sentence))

    assert aligned == [
        CharAlignment(
            char_index=0,
            substring="Hello, ",
            num_samples=Phoneme.BOS
            | Phoneme.h
            | Phoneme.ə
            | Phoneme.l
            | Phoneme.STRESS
            | Phoneme.o
            | Phoneme.ʊ
            | Phoneme.COMMA
            | Phoneme.SPACE,
        ),
        CharAlignment(
            char_index=7,
            substring="world!",
            num_samples=Phoneme.w
            | Phoneme.STRESS
            | Phoneme.ɜ
            | Phoneme.LONG
            | Phoneme.l
            | Phoneme.d
            | Phoneme.EXCLAMATION
            | Phoneme.EOS,
        ),
    ]


def test_align_word_splitting():
    espeak_sentence = EspeakSentence(
        [
            EspeakClause(
                "Hello\xa0world!",
                ["h", "ə", "l", "ˈ", "o", "ʊ", " ", "w", "ˈ", "ɜ", "ː", "l", "d", "."],
            ),
        ]
    )

    phoneme_alignments = [
        PhonemeAlignment(phoneme="^", phoneme_ids=[-1], num_samples=Phoneme.BOS),
        PhonemeAlignment(phoneme="h", phoneme_ids=[-1], num_samples=Phoneme.h),
        PhonemeAlignment(phoneme="ə", phoneme_ids=[-1], num_samples=Phoneme.ə),
        PhonemeAlignment(phoneme="l", phoneme_ids=[-1], num_samples=Phoneme.l),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="o", phoneme_ids=[-1], num_samples=Phoneme.o),
        PhonemeAlignment(phoneme="ʊ", phoneme_ids=[-1], num_samples=Phoneme.ʊ),
        PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
        PhonemeAlignment(phoneme="w", phoneme_ids=[-1], num_samples=Phoneme.w),
        PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
        PhonemeAlignment(phoneme="ɜ", phoneme_ids=[-1], num_samples=Phoneme.ɜ),
        PhonemeAlignment(phoneme="ː", phoneme_ids=[-1], num_samples=Phoneme.LONG),
        PhonemeAlignment(phoneme="l", phoneme_ids=[-1], num_samples=Phoneme.l),
        PhonemeAlignment(phoneme="d", phoneme_ids=[-1], num_samples=Phoneme.d),
        PhonemeAlignment(
            phoneme="!", phoneme_ids=[-1], num_samples=Phoneme.EXCLAMATION
        ),
        PhonemeAlignment(phoneme="$", phoneme_ids=[-1], num_samples=Phoneme.EOS),
    ]

    aligned = list(align(phoneme_alignments, espeak_sentence))

    assert aligned == [
        CharAlignment(
            char_index=0,
            substring="Hello\xa0",
            num_samples=Phoneme.BOS
            | Phoneme.h
            | Phoneme.ə
            | Phoneme.l
            | Phoneme.STRESS
            | Phoneme.o
            | Phoneme.ʊ
            | Phoneme.SPACE,
        ),
        CharAlignment(
            char_index=6,
            substring="world!",
            num_samples=Phoneme.w
            | Phoneme.STRESS
            | Phoneme.ɜ
            | Phoneme.LONG
            | Phoneme.l
            | Phoneme.d
            | Phoneme.EXCLAMATION
            | Phoneme.EOS,
        ),
    ]


# def test_align_with_raw_phonetic_input():
#     # espeak_sentence = voice.phonemize_clause_aligned("I am the [[ bˈætmæn ]] not [[bɹˈuːs wˈe‍ɪn]]")

#     espeak_sentence = EspeakSentence([
#         # # Actual: 'aɪɐm ðə' (not 'aɪ ɐm ðə')
#         # # But we add a space here because this test isn't about word boundaries
#         EspeakClause('I am the ', ['a', 'ɪ', ' ', 'ɐ', 'm', ' ', 'ð', 'ə']),
#         EspeakClause('[[ bˈætmæn ]]', [' ', 'b', 'ˈ', 'æ', 't', 'm', 'æ', 'n', ' ']),
#         EspeakClause(' not ', ['n', 'ˈ', 'ɑ', 'ː', 't']),
#         EspeakClause('[[bɹˈuːs wˈe\u200dɪn]]', [' ', 'b', 'ɹ', 'ˈ', 'u', 'ː', 's', ' ', 'w', 'ˈ', 'e', '\u200d', 'ɪ', 'n']),
#     ])

#     phoneme_alignments = [
#         PhonemeAlignment(phoneme="^", phoneme_ids=[-1], num_samples=Phoneme.BOS),
#         PhonemeAlignment(phoneme="a", phoneme_ids=[-1], num_samples=Phoneme.a),
#         PhonemeAlignment(phoneme="ɪ", phoneme_ids=[-1], num_samples=Phoneme.ɪ),
#         PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
#         PhonemeAlignment(phoneme="ɐ", phoneme_ids=[-1], num_samples=Phoneme.ɐ),
#         PhonemeAlignment(phoneme="m", phoneme_ids=[-1], num_samples=Phoneme.m),
#         PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
#         PhonemeAlignment(phoneme="ð", phoneme_ids=[-1], num_samples=Phoneme.ð),
#         PhonemeAlignment(phoneme="ə", phoneme_ids=[-1], num_samples=Phoneme.ə),
#         PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
#         PhonemeAlignment(phoneme="b", phoneme_ids=[-1], num_samples=Phoneme.b),
#         PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
#         PhonemeAlignment(phoneme="æ", phoneme_ids=[-1], num_samples=Phoneme.æ),
#         PhonemeAlignment(phoneme="t", phoneme_ids=[-1], num_samples=Phoneme.t),
#         PhonemeAlignment(phoneme="m", phoneme_ids=[-1], num_samples=Phoneme.m),
#         PhonemeAlignment(phoneme="æ", phoneme_ids=[-1], num_samples=Phoneme.æ),
#         PhonemeAlignment(phoneme="n", phoneme_ids=[-1], num_samples=Phoneme.n),
#         PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
#         PhonemeAlignment(phoneme="n", phoneme_ids=[-1], num_samples=Phoneme.n),
#         PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
#         PhonemeAlignment(phoneme="ɑ", phoneme_ids=[-1], num_samples=Phoneme.ɑ),
#         PhonemeAlignment(phoneme="ː", phoneme_ids=[-1], num_samples=Phoneme.LONG),
#         PhonemeAlignment(phoneme="t", phoneme_ids=[-1], num_samples=Phoneme.t),
#         PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
#         PhonemeAlignment(phoneme="b", phoneme_ids=[-1], num_samples=Phoneme.b),
#         PhonemeAlignment(phoneme="ɹ", phoneme_ids=[-1], num_samples=Phoneme.ɹ),
#         PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
#         PhonemeAlignment(phoneme="u", phoneme_ids=[-1], num_samples=Phoneme.u),
#         PhonemeAlignment(phoneme="ː", phoneme_ids=[-1], num_samples=Phoneme.LONG),
#         PhonemeAlignment(phoneme="s", phoneme_ids=[-1], num_samples=Phoneme.s),
#         PhonemeAlignment(phoneme=" ", phoneme_ids=[-1], num_samples=Phoneme.SPACE),
#         PhonemeAlignment(phoneme="w", phoneme_ids=[-1], num_samples=Phoneme.w),
#         PhonemeAlignment(phoneme="ˈ", phoneme_ids=[-1], num_samples=Phoneme.STRESS),
#         PhonemeAlignment(phoneme="e", phoneme_ids=[-1], num_samples=Phoneme.e),
#         PhonemeAlignment(phoneme="\u200d", phoneme_ids=[-1], num_samples=Phoneme.U_200D),
#         PhonemeAlignment(phoneme="ɪ", phoneme_ids=[-1], num_samples=Phoneme.ɪ),
#         PhonemeAlignment(phoneme="n", phoneme_ids=[-1], num_samples=Phoneme.n),
#         PhonemeAlignment(phoneme="$", phoneme_ids=[-1], num_samples=Phoneme.EOS),
#     ]

#     aligned = list(align(phoneme_alignments, espeak_sentence))

#     assert aligned == [
#         CharAlignment(char_index=0, substring="I ", num_samples=Phoneme.BOS | Phoneme.a | Phoneme.ɪ Phoneme.SPACE),
#         CharAlignment(char_index=0, substring="am ", num_samples=Phoneme.ɐ | Phoneme.m | Phoneme.SPACE),
#         CharAlignment(char_index=5, substring="the ", num_samples=Phoneme.ð | Phoneme.ə | Phoneme.SPACE),
#         CharAlignment(char_index=9, substring="[[ bˈætmæn ]]", num_samples=Phoneme.SPACE | Phoneme.b | Phoneme.STRESS | Phoneme.æ | Phoneme.t | Phoneme.m | Phoneme.æ | Phoneme.n | Phoneme.SPACE),
#         CharAlignment(char_index=20, substring=" not ", num_samples=Phoneme.n | Phoneme.STRESS | Phoneme.ɑ | Phoneme.LONG | Phoneme.t | Phoneme.SPACE),
#         CharAlignment(char_index=25, substring="[[bɹˈuːs wˈe‍ɪn]]", num_samples=Phoneme.SPACE | Phoneme.b | Phoneme.ɹ | Phoneme.STRESS | Phoneme.u | Phoneme.LONG | Phoneme.s | Phoneme.SPACE | Phoneme.w | Phoneme.STRESS | Phoneme.e | Phoneme.U_200D | Phoneme.ɪ | Phoneme.n | Phoneme.EOS),
#     ]
