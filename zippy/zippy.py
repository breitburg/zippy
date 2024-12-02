#!/usr/bin/env python3

# Code to attempt to detect AI-generated text [relatively] quickly via compression ratios
# (C) 2023 Thinkst Applied Research, PTY
# Author: Jacob Torrey <jacob@thinkst.com>

import lzma, argparse, os, itertools
from zlib import compressobj, Z_FINISH
from brotli import compress as brotli_compress, MODE_TEXT
from numpy import array_split
import re, sys, statistics
from abc import ABC, abstractmethod
from enum import Enum
from math import ceil
from typing import List, Optional
from multiprocessing import Pool, cpu_count
from importlib.resources import files
from .__init__ import __version__


class Score:
    def __init__(self, delta: float) -> None:
        self.delta = delta

    @property
    def determination(self) -> str:
        return "AI" if self.delta < 0 else "Human"

    @property
    def confidence(self) -> float:
        return abs(self.delta * 100)

    def __str__(self) -> str:
        return f"{self.determination} {self.confidence:.3f}%"


def clean_text(s: str) -> str:
    """
    Removes formatting and other non-content data that may skew compression ratios (e.g., duplicate spaces)
    """
    # Remove extra spaces and duplicate newlines.
    s = re.sub(" +", " ", s)
    s = re.sub("\t", "", s)
    s = re.sub("\n+", "\n", s)
    s = re.sub("\n ", "\n", s)
    s = re.sub(" \n", "\n", s)

    # Remove non-alphanumeric chars
    s = re.sub(r"[^0-9A-Za-z,\.\(\) \n]", "", s)  # .lower()

    return s


# The prelude file is a text file containing only AI-generated text, it is used to 'seed' the LZMA dictionary
PRELUDE_FILE: str = "ai-generated.txt"
PRELUDE_STR = clean_text(
    files("zippy").joinpath(PRELUDE_FILE).read_text(encoding="utf-8")
)


class AIDetector(ABC):
    """
    Base class for AI detection
    """

    @abstractmethod
    def score_text(self, sample: str) -> Optional[Score]:
        pass


class LzmaLlmDetector(AIDetector):
    """Class providing functionality to attempt to detect LLM/generative AI generated text using the LZMA compression algorithm"""

    def __init__(
        self,
        prelude_file: Optional[str] = None,
        prelude_str: Optional[str] = None,
        prelude_ratio: Optional[float] = None,
        preset: int = 4,
        normalize: bool = False,
    ) -> None:
        """Initializes a compression with the passed prelude file, and optionally the number of digits to round to compare prelude vs. sample compression"""
        self.PRESET: int = preset
        self.c_buf: List[bytes] = []
        self.in_bytes: int = 0
        self.prelude_ratio: float = 0.0
        self.nf: float = 0.0

        if prelude_ratio != None:
            self.prelude_ratio = prelude_ratio

        if prelude_file != None:
            # Read it once to get the default compression ratio for the prelude
            with open(prelude_file, "r", encoding="utf-8") as fp:
                self.prelude_str = fp.read()
            self.prelude_ratio = self._compress(self.prelude_str)
            return
            # print(prelude_file + ' ratio: ' + str(self.prelude_ratio))

        if prelude_str != None:
            self.prelude_str = prelude_str
            if self.prelude_ratio == 0.0:
                self.prelude_ratio = self._compress(prelude_str)
        if normalize:
            self.nf: float = self.prelude_ratio / len(self.prelude_str)

    def _compress(self, s: str) -> float:
        orig_len = len(s.encode())
        c = lzma.LZMACompressor(preset=self.PRESET)
        bytes = c.compress(s.encode())
        bytes += c.flush()
        c_len = len(bytes)
        return c_len / orig_len

    def score_text(self, sample: str) -> Optional[Score]:
        """
        Returns a tuple of a string (AI or Human) and a float confidence (higher is more confident) that the sample was generated
        by either an AI or human. Returns None if it cannot make a determination
        """
        if self.prelude_ratio == 0.0:
            return None

        # print('LZMA: ' + str((self.prelude_ratio, sample_score)))
        delta = self.prelude_ratio - (
            self._compress(self.prelude_str + sample)
            - (self.nf * len(sample) * (len(sample) / len(self.prelude_str)))
        )

        return Score(delta * 100)


class Zippy:
    """
    Class to wrap the functionality of Zippy
    """

    def __init__(
        self,
        preset: Optional[int] = None,
        prelude_file: str = PRELUDE_FILE,
        normalize: bool = False,
    ) -> None:
        self.PRESET = preset
        if prelude_file == PRELUDE_FILE:
            self.PRELUDE_FILE = str(files("zippy").joinpath(PRELUDE_FILE))
            self.PRELUDE_STR = clean_text(
                files("zippy").joinpath(PRELUDE_FILE).read_text()
            )
        else:
            self.PRELUDE_FILE = prelude_file
            with open(self.PRELUDE_FILE, encoding="utf-8") as fp:
                self.PRELUDE_STR = clean_text(fp.read())

        if self.PRESET:
            self.detector = LzmaLlmDetector(
                prelude_str=self.PRELUDE_STR, preset=self.PRESET
            )
        else:
            self.detector = LzmaLlmDetector(prelude_str=self.PRELUDE_STR)


def print_colored(text: str, value: float) -> None:
    """
    Print text with color based on a value between -1 and 1.
    -1 corresponds to bright red
    0 corresponds to white
    1 corresponds to bright green

    Args:
        text (str): The text to print
        value (float): A value between -1 and 1 determining the color

    Raises:
        ValueError: If value is not between -1 and 1
    """
    # Normalize value to be between 0 and 1
    normalized = (max(-1, min(1, value)) + 1) / 2

    # Calculate RGB values
    # Red component: starts at 255 (-1), decreases to 0 (1)
    # Green component: starts at 0 (-1), increases to 255 (1)
    red = int(255 * (1 - normalized))
    green = int(255 * normalized)
    blue = 0

    # ANSI escape code for RGB colors
    color_code = f"\033[38;2;{red};{green};{blue}m"
    reset_code = "\033[0m"

    # Print the colored text
    print(f"{color_code}{text}{reset_code}", end="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", required=False, action="store_true", help="Display the version and exit"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-pp",
        help="Prettify the output",
        required=False,
        action="store_true",
    )
    args = parser.parse_args()

    if args.v:
        print(__version__)
        return

    z = Zippy()

    text = "".join(list(sys.stdin)).strip()
    sentences = re.findall(r"[^.!?]*[.!?]", text)

    scored = [
        (sentence, z.detector.score_text(sentence.strip())) for sentence in sentences
    ]

    if args.pp:
        for score in scored:
            print_colored(score[0], score[1].delta * 100)

        print(f"\n\n")

    print(Score(sum([score[1].delta for score in scored]) / len(scored)))

    print()


if __name__ == "__main__":
    main()
