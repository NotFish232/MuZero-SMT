from muzero_smt.models import scalar_to_support, support_to_scalar


import torch as T


num_supports = 10
x = T.arange(-50, 50).reshape(100)

res = scalar_to_support(x, num_supports)

print(support_to_scalar(res, num_supports))