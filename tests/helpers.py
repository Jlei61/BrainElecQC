"""Named binary streams accepted by the Python import boundary."""
from dataclasses import dataclass
from typing import BinaryIO


@dataclass
class NamedInput:
    stream: BinaryIO
    filename: str
