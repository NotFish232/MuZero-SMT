from utils.temp import scalar_to_support as mine
from utils.models import scalar_to_support as theirs


import torch as T


num_supports = 5
x = T.linspace(-1000, 1000, 420).reshape(21, 20)

mine_res = mine(x, num_supports)
their_res = theirs(x, num_supports)

print(T.allclose(mine_res, their_res))