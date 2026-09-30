from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


BoolValue = np.bool_
BoolArray = NDArray[BoolValue]

FloatValue = np.float64
FloatArray = NDArray[FloatValue]
FloatFunction2D = Callable[[ArrayLike, ArrayLike], FloatArray | FloatValue]
FloatFunction3D = Callable[[ArrayLike, ArrayLike, ArrayLike], FloatArray | FloatValue]

IntegerValue = np.int64
IntegerArray = NDArray[IntegerValue]
