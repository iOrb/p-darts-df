from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

# TEST = Genotype(
#   normal = [
#     ('a', 0),
#     ('b', 1),
#     ('c', 0),
#     ('d', 2),
#     ('e', 0),
#     ('f', 3),
#     ('g', 1),
#     ('h', 1),
#     ('i', 0),
#     ('j', 1),
#     ],
#   normal_concat = [1, 10, 20],
#   reduce = [
#     ('a', 0),
#     ('b', 1),
#     ('c', 0),
#     ('d', 2),
#     ('e', 0),
#     ('f', 1),
#     ('g', 0),
#     ('h', 1),
#     ('i', 0),
#     ('j', 5),
#   ],
#   reduce_concat = [1, 10, 20]
# )

# TEST = Genotype(
#   normal = [
#     ('a', 0),
#     ('b', 1),
#     ('c', 2),
#     ('d', 4),
#     ('e', 4),
#     ('f', 5),
#     # ('g', 6),
#     # ('h', 7),
#     ],
#   normal_concat = [3, 4, 6],
#   reduce = [
#     ('a', 0),
#     ('b', 1),
#     ('c', 2),
#     ('d', 3),
#     ('e', 4),
#     ('f', 5),
#   ],
#   reduce_concat = [3, 4, 6]
# )

# TEST = Genotype(
#   normal=[
#     ('a', 1),
#     ('b', 0),
#     ('c', 0),
#     ('d', 0),
#     ('e', 3),
#     ('f', 4),
#     # ('g', 0),
#     # ('h', 0),
#     # ('i', 1),
#     # ('j', 1),
#   ],
#   normal_concat=[2, 3, 4, 5, 6],
#   reduce=[
#     ('sep_conv_5x5', 1),
#     ('sep_conv_7x7', 0),
#     ('max_pool_3x3', 1),
#     ('sep_conv_7x7', 0),
#     ('avg_pool_3x3', 1),
#     ('sep_conv_5x5', 0),
#     ('skip_connect', 3),
#     ('avg_pool_3x3', 2),
#     ('sep_conv_3x3', 2),
#     ('max_pool_3x3', 1),
#   ],
#   reduce_concat=[4, 5, 6],
# )

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
PDARTS = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0),('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3',0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

# TO USE
CIFAR = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

# NT_V0 = Genotype(normal=[('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('avg_pool_3x3', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 1), ('skip_connect', 3)], reduce_concat=range(2, 6))

# spl-1 v-400 skip connects
NT_V1 = Genotype(normal=[('avg_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('avg_pool_3x3', 2), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_5x5', 1), ('skip_connect', 2), ('dil_conv_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

# spl-1 v-400 restricted to 1 skip connect
NT_V2 = Genotype(normal=[('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_5x5', 1), ('skip_connect', 2), ('dil_conv_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))


# ======================

# # SPLITS EXPERIMETS V0
#
# # 40
# n40_0 = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('skip_connect', 1), ('dil_conv_5x5', 1), ('sep_conv_5x5', 4), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 2), ('skip_connect', 3), ('avg_pool_3x3', 1), ('skip_connect', 3)], reduce_concat=range(2, 6))
# n40_1 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 3), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
# n40_2 = Genotype(normal=[('avg_pool_3x3', 0), ('skip_connect', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=range(2, 6))
#
# # 100
# n100_0 = Genotype(normal=[('avg_pool_3x3', 0), ('skip_connect', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
# n100_1 = Genotype(normal=[('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('avg_pool_3x3', 4), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('max_pool_3x3', 0), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))
# n100_2 = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('avg_pool_3x3', 4), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 1), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
#
# # 200
# n200_0 = Genotype(normal=[('sep_conv_5x5', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 3), ('avg_pool_3x3', 1), ('skip_connect', 4)], reduce_concat=range(2, 6))
# n200_1 = Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('skip_connect', 0), ('avg_pool_3x3', 2), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1)], reduce_concat=range(2, 6))
# n200_2 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 2), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('avg_pool_3x3', 3), ('dil_conv_3x3', 3), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('avg_pool_3x3', 0), ('skip_connect', 4)], reduce_concat=range(2, 6))
#
# # 400
# n400_0 = Genotype(normal=[('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('skip_connect', 2), ('dil_conv_5x5', 2), ('avg_pool_3x3', 3), ('sep_conv_3x3', 0), ('dil_conv_3x3', 3)], reduce_concat=range(2, 6))
# n400_1 = Genotype(normal=[('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 2), ('skip_connect', 2), ('max_pool_3x3', 2), ('dil_conv_3x3', 2), ('max_pool_3x3', 3), ('avg_pool_3x3', 3), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 3), ('avg_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
# n400_2 = Genotype(normal=[('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_5x5', 1), ('skip_connect', 2), ('dil_conv_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
#
# # cifar
# cifar = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

# ======================

# SPLITS EXPERIMETS V1
# 40
n40_0 = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('skip_connect', 1), ('dil_conv_5x5', 1), ('sep_conv_5x5', 3), ('avg_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 2), ('skip_connect', 3), ('avg_pool_3x3', 1), ('skip_connect', 3)], reduce_concat=range(2, 6))
n40_1 = Genotype(normal=[('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
n40_2 = Genotype(normal=[('avg_pool_3x3', 0), ('skip_connect', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 0), ('sep_conv_3x3', 3), ('sep_conv_5x5', 3), ('dil_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=range(2, 6))

# 100
n100_0 = Genotype(normal=[('avg_pool_3x3', 0), ('skip_connect', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('avg_pool_3x3', 2), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
n100_1 = Genotype(normal=[('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('max_pool_3x3', 0), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))
n100_2 = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 3), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 1), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

# 200
n200_0 = Genotype(normal=[('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 3), ('avg_pool_3x3', 1), ('skip_connect', 3)], reduce_concat=range(2, 6))
n200_1 = Genotype(normal=[('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('dil_conv_3x3', 0), ('skip_connect', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 2), ('skip_connect', 3), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
n200_2 = Genotype(normal=[('max_pool_3x3', 0), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

# 400
n400_0 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('dil_conv_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('skip_connect', 2), ('dil_conv_5x5', 2), ('avg_pool_3x3', 3), ('sep_conv_3x3', 0), ('dil_conv_3x3', 3)], reduce_concat=range(2, 6))
n400_1 = Genotype(normal=[('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 0), ('skip_connect', 3)], reduce_concat=range(2, 6))
n400_2 = Genotype(normal=[('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 0), ('avg_pool_3x3', 2), ('avg_pool_3x3', 2), ('skip_connect', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_5x5', 1), ('skip_connect', 2), ('dil_conv_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

# cifar
cifar = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))

# ======================
