import hashlib


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
