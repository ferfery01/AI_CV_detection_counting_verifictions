import hashlib
from enum import Enum
from typing import List, Optional, Sequence, Type, Union


def str_to_float(s: str, salt: bytes = b"") -> float:
    """Computes a SHA-1 hash of the input string and 'salt', then normalizes this hash into a
    float in the range [0, 1).

    The function encodes the string `s` and concatenates it with `salt` (if provided).
    It then computes a SHA-1 hash of this combination, converting the hash digest into an integer
    representation. This integer is subsequently normalized to a float between 0 (inclusive) and
    1 (exclusive) by dividing it by the maximum possible value of the hash.

    The function returns a deterministic "random" float: for the same input `s` and `salt`, the
    output will be the same. This is useful for generating pseudo-randomness that can be recreated.

    Args:
        s (str): The input string to hash.
        salt (bytes, optional): An optional byte string to concatenate with `s` before hashing.

    Returns:
        float: A deterministic "random" float in the range [0, 1), derived from the SHA-1 hash of the
            input string and salt.
    """
    x = hashlib.sha1(salt + s.encode())
    n = float(int.from_bytes(x.digest(), "little")) / (1 << (x.digest_size * 8))
    return n


def str_to_hash(s: str, length: int = 20, salt: bytes = b"") -> str:
    """Generates a SHA-1 hash from the input string, then returns a substring of the hash up to
    the specified length.

    This function will first compute the SHA-1 hash of the concatenated `salt` and the encoded `s`
    string. The `length` parameter determines the size of the substring of the hash that is returned.
    By default, it returns the full SHA-1 hash, which has a length of 20 characters.
    If a length longer than the size of the SHA-1 hash is requested, a ValueError is raised.

    Args:
        s (str): The input string to hash.
        length (int, optional): The length of the substring of the hash to return.
        salt (bytes, optional): An optional byte string to concatenate with `s` before hashing.

    Returns:
        str: The substring of the SHA-1 hash of the input string.

    Raises:
        ValueError: If the requested length is larger than the size of the SHA-1 hash.
    """
    hash = hashlib.sha1(salt + s.encode()).hexdigest()
    if len(hash) > length:
        return hash[:length]
    else:
        raise ValueError(f"Length param should not exceed SHA1 length, which is {len(hash)}")


def convert_to_string_list(
    input_val: Optional[Union[str, Sequence[str], Enum, Sequence[Enum]]],
    enum_class: Optional[Type[Enum]] = None,
) -> Optional[List[str]]:
    """Convert the provided input value into a standardized list of strings.

    The function handles various input types:
    - Individual strings or Enum values are wrapped into a list.
    - Sequences (like lists or tuples) containing strings or Enum values are converted to a list of strings.

    Args:
        input_val (Optional[Union[str, Sequence[str], Enum, Sequence[Enum]]]): The input value to be converted
        enum_class (Optional[Type[Enum]]):
            The expected Enum class if Enum values are provided. This is used for type validation.

    Returns:
        Optional[List[str]]: The standardized list of strings. If the input is None, the return is also None.

    Raises:
        ValueError: If the input type is not recognized or doesn't match the expected types.
    """
    if input_val is None:
        return None

    input_list = [input_val] if isinstance(input_val, (str, Enum)) else input_val
    if isinstance(input_list, Sequence):
        str_input_list = []
        for item in input_list:
            if isinstance(item, str):
                str_input_list.append(item)
            elif enum_class is not None and isinstance(item, enum_class):
                str_input_list.append(item.name)
            else:
                raise TypeError(f"Expected input type to be str or {enum_class}, but got {type(item)}.")
        return str_input_list
    else:
        allowed_types = [str, enum_class, list, tuple]
        allowed_str = ", ".join([str(t) for t in allowed_types])
        raise ValueError(f"Expected input type to be one of {allowed_str}, but got {type(input_val)}.")
