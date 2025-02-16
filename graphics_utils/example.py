import numpy as np
from graphics_utils import film_refl

res = film_refl([0, 0, 1], [0, 0, 1], 25, 500)

print(res)