from enum import IntFlag, auto


class Phoneme(IntFlag):
    """
    We use bitmask flags to make it easy to sum up num_samples for test assertions.

    NOTE: We use `+` instead of `|` to combine flags, because the same phoneme can
    appear multiple times in a word, and we want to count each occurrence. This means
    the result isn't strictly unique, but good enough for testing purposes.
    """

    BOS = auto()
    EOS = auto()

    COMMA = auto()
    SPACE = auto()
    BANG = auto()

    STRESS = auto()
    LONG = auto()

    a = auto()
    æ = auto()
    ɐ = auto()
    ɑ = auto()
    ɒ = auto()
    b = auto()
    d = auto()
    ð = auto()
    e = auto()
    ə = auto()
    ɛ = auto()
    ɜ = auto()
    f = auto()
    h = auto()
    i = auto()
    ɪ = auto()
    j = auto()
    l = auto()
    m = auto()
    n = auto()
    ŋ = auto()
    o = auto()
    r = auto()
    ɹ = auto()
    s = auto()
    t = auto()
    θ = auto()
    u = auto()
    v = auto()
    ʊ = auto()
    ʌ = auto()
    w = auto()
    z = auto()

    U_200D = auto()
