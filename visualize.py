import os
import sys
from graphviz import Digraph
import genotypes


def plot(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  assert len(genotype) % 2 == 0
  steps = len(genotype) // 2

  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  for i in range(steps):
    for k in [2*i, 2*i + 1]:
      op, j = genotype[k]
      if j == 0:
        u = "c_{k-2}"
      elif j == 1:
        u = "c_{k-1}"
      else:
        u = str(j-2)
      v = str(i)
      g.edge(u, v, label=op, fillcolor="gray")

  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="gray")

  g.render(filename, view=True, format='jpg')


if __name__ == '__main__':
  # if len(sys.argv) != 2:
  #   print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
  #   sys.exit(1)
  # genotype_name = sys.argv[1]

  arch_list = ["n40_0", "n40_1", "n40_2", \
               "n100_0", "n100_1", "n100_2", \
               "n200_0", "n200_1", "n200_2", \
               "n400_0", "n400_1", "n400_2", \
               "cifar"]

  # arch_list = ["n400_1"]

  for arch in arch_list:
    genotype_name = arch
    try:
      genotype = eval('genotypes.{}'.format(genotype_name))
    except AttributeError:
      print("{} is not specified in genotypes.py".format(genotype_name))
      sys.exit(1)

    try:
      os.mkdir(f"out/archs/{genotype_name}")
    except:
      pass
    plot(genotype.normal, f"out/archs/{genotype_name}/normal")
    plot(genotype.reduce, f"out/archs/{genotype_name}/reduction")

