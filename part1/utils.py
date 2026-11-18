import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import hashlib
import random
import re
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


# Adjacent keys on a QWERTY keyboard, including diagonals. Used to imitate a single-character typo.
KEYBOARD_NEIGHBORS = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg",
    "y": "tugh", "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol",
    "a": "qwsz", "s": "weadzx", "d": "ersfcx", "f": "rtgdvc", "g": "tyhfbn",
    "h": "yugjnb", "j": "uikhmn", "k": "ioljm", "l": "opk",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}

# Chance that a long alphabetic word receives one character substitution.
TYPO_WORD_PROB = 0.5
MIN_WORD_LEN = 4


def _one_typo(word, rng):
    chars = list(word)
    candidates = [i for i, ch in enumerate(chars) if ch.lower() in KEYBOARD_NEIGHBORS]
    if not candidates:
        return word
    index = rng.choice(candidates)
    original = chars[index]
    replacement = rng.choice(KEYBOARD_NEIGHBORS[original.lower()])
    chars[index] = replacement.upper() if original.isupper() else replacement
    return "".join(chars)


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # Keyboard typos: each alphabetic word of length >= 4 is edited with probability
    # TYPO_WORD_PROB. An edit replaces one letter with a neighboring QWERTY key.
    # The label is left unchanged. The RNG is seeded from the original text so the
    # same review always transforms the same way, including under dataset.map.
    text = example["text"]
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    pieces = re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)
    edited = []
    for piece in pieces:
        if piece.isalpha() and len(piece) >= MIN_WORD_LEN and rng.random() < TYPO_WORD_PROB:
            piece = _one_typo(piece, rng)
        edited.append(piece)
    example["text"] = "".join(edited)

    ##### YOUR CODE ENDS HERE ######

    return example
