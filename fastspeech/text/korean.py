import re

PAD = '_'
EOS = '~'
PUNC = '!\'(),-.:;?'
SPACE = ' '
_SILENCES = ['sp', 'spn', 'sil']

JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

VALID_CHARS = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + PUNC + SPACE
ALL_SYMBOLS = list(PAD + EOS + VALID_CHARS) + _SILENCES
s_to_i={c: i for i, c in enumerate(ALL_SYMBOLS)}
KOR_SYMBOLS=ALL_SYMBOLS

Kchar_to_id={c: i for i, c in enumerate(KOR_SYMBOLS)}
id_to_Kchar={i: c for i, c in enumerate(KOR_SYMBOLS)}
kor_symbols=KOR_SYMBOLS
symbols= kor_symbols
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def text_to_sequence(text):
    sequence = []
    while len(text):
        m = _curly_re.match(text)
        if not m:
            print('not m!')
            sequence = _symbols_to_sequence(text)
            break
        sequence = _symbols_to_sequence(m.group(2))
        text = m.group(3)
    return sequence


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols.split() if _should_keep_symbol(s)]


def _should_keep_symbol(s):
    return s in _symbol_to_id and s is not '~' and s is not '_'