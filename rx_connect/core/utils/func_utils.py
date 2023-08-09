from itertools import islice
from typing import (
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

T = TypeVar("T")
N = TypeVar("N", int, float)


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


@overload
def to_tuple(x: None) -> None:
    ...


@overload
def to_tuple(x: N) -> Tuple[N, N]:
    ...


@overload
def to_tuple(x: Tuple[N, ...]) -> Tuple[N, N]:
    ...


def to_tuple(x: Optional[Union[N, Tuple[N, ...]]]) -> Optional[Tuple[N, N]]:
    """Converts the input argument to a tuple. If the input is already a tuple, then it is
    returned as it is. If the input is a number, then a tuple of length 2 is returned with the
    same number repeated twice.

    Args:
        x (Optional[Union[N, Tuple[N, ...]]]): The input to convert to a tuple. Can be a number or
            a sequence of numbers. If None, then None is returned.

    Returns:
        Tuple[N]: The input converted to a tuple.
    """
    if x is None:
        return None

    if isinstance(x, (int, float)):
        return (x, x)
    elif isinstance(x, Sequence):
        if len(x) != 2:
            raise ValueError(f"Expected a sequence of length 2, but got {len(x)}")
        return cast(Tuple[N, N], tuple(x))
    else:
        raise TypeError(f"Expected a sequence or a number, but got {type(x)}")
