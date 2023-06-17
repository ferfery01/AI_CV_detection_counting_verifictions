from typing import List, NamedTuple


class CounterModuleOutput(NamedTuple):
    bbox: List[int]
    """Bounding box in the format [X1, Y1, X2, Y2]
    """
    scores: float
    """Confidence score of the bounding box
    """
