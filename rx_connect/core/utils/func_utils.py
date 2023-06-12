from itertools import islice
from typing import Generator, List, Sequence, TypeVar

T = TypeVar("T")


def batch_generator(sequence: Sequence[T], batch_size: int) -> Generator[List[T], None, None]:
    """Generates batches from a sequence. Yields lists of items of length batch_size.

    Args:
        sequence (Sequence[T]): The input sequence to generate batches from.
        batch_size (int): The size of the batches to generate.

    Yields:
        Generator[List[T], None, None]: The next batch of items from the input sequence.
    """
    iterator = iter(sequence)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch
