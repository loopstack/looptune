import loop_tool as lt
import random
import torch
import itertools
from itertools import combinations, permutations
import pandas as pd

lt.set_default_backend("loop_nest")
max_loops = 20


def mm(a, b):
    m, n, k = lt.symbols("m n k")
    return (a.to(m, k) * b.to(k, n)).sum(k)

def conv1d(a, b):
    ni, no, ci, co, k = lt.symbols("ni no ci co k")
    a = a.to(ci, no, constraints=[(no, ni + k)])
    return (a * b.to(ci, co, k)).sum(ci, k)

def gen_mm(m, n, k, msplits, ksplits, nsplits):
    A = lt.Tensor(m, k)
    B = lt.Tensor(k, n)
    C = mm(A, B)
    loop_tree = C.loop_tree
    msplits = list(filter(lambda x: x>1, msplits))
    ksplits = list(filter(lambda x: x>1, ksplits))
    nsplits = list(filter(lambda x: x>1, nsplits))
    for s in msplits[::-1]:
      loop_tree = loop_tree.split(0, s)
    for s in ksplits[::-1]:
      loop_tree = loop_tree.split(len(msplits) + 1, s)
    for s in nsplits[::-1]:
      loop_tree = loop_tree.split(len(msplits) + len(ksplits) + 2, s)
    #mss = '-'.join(str(s) for s in msplits)
    #nss = '-'.join(str(s) for s in nsplits)
    #kss = '-'.join(str(s) for s in ksplits)
    #fn = f"dataset/mm_{m}x{n}x{k}_{mss}_{nss}_{kss}.txt"
    mul = loop_tree.ir.nodes[2]
    if loop_tree.ir.order[mul][-1][1][0] % 8 != 0:
      return None
    #with open(fn, 'w') as f:
    #  f.write(loop_tree.ir.serialize())
    return loop_tree


def reorder_loop(loop_tree, perm):
  permuted_tree = loop_tree
  perm = list(perm)
  done = False
  while not done:
    done = True
    for i in range(len(perm) - 1):
      if perm[i] > perm[i+1]:
        done = False
        permuted_tree = permuted_tree.swap_loops(i, i+1)
        perm[i], perm[i+1] = perm[i+1], perm[i]

  if (perm != [i for i in range(len(perm))]):breakpoint()
  return permuted_tree


def permute_annotations(permuted_tree):
  annotated_trees = []
  annotation_compbinations = list(itertools.product([None, 'unroll', 'vectorize'], repeat=len(permuted_tree.loops)))
  for loops_annotations in random.sample(annotation_compbinations, 5):
    new_tree = permuted_tree
    for i, la in enumerate(loops_annotations):
      if la != None:
        new_tree = new_tree.annotate(permuted_tree.loops[i], la)

    print(new_tree)
    annotated_trees.append(new_tree)

  return annotated_trees


def create_permutations_db(loop_tree):
  global max_loops
  all_possible_trees = []
  
  active_loops = 0
  for i, loop_id in enumerate(loop_tree.loops):
    if loop_id != i:
      active_loops = i

  for comb in combinations(range(active_loops), active_loops):
      for perm in permutations(comb):
        try:
          permuted_tree = reorder_loop(loop_tree, perm)
          for permuted_tree in permute_annotations(permuted_tree):
            agent = lt.LoopTreeAgent(permuted_tree)
            
            loops_tensor = agent.get_loops_tensor()
            if len(loops_tensor) < max_loops:
              loops_tensor.extend( [[0] * len(loops_tensor[0])] * (max_loops - len(loops_tensor)))

              all_possible_trees.append(
                (permuted_tree.ir.serialize(), 
                torch.tensor(loops_tensor + [[0] * len(loops_tensor[0])] * (max_loops - len(loops_tensor))), 
                torch.tensor(agent.get_stride_histogram()), 
                permuted_tree.FLOPS() / 1e9))     
              print(permuted_tree)
            else:
              breakpoint()
              print(f"Loop has more loops than {max_loops}")

        except KeyboardInterrupt:
          exit()
        except:
          pass

  return all_possible_trees


def gen_near(mi):
  m = 2**mi
  #for m in range(2 ** (mi-1), 2 ** mi, 2 **(mi-2)):
  for md in range(-1, 2):
    yield m + md

def gen_iter(start=5, end=7):
  out = []
  for mi in range(start, end):
    for m in gen_near(mi):
      #yield m
      out.append(m)
  return out

def gen_split(n):
  def f(splits):
    return list(filter(lambda x: 1 not in x, splits))
  splits = []
  if n <= 2:
    return f(splits)
  for i in range(2, min(n, 6)):
    splits.append((i,))
  if n <= 6:
    return f(splits)
  for i in range(6, min(n, 16), 2):
    splits.append((i,))
  if n <= 16:
    return f(splits)
  for i in range(16, min(n, 64), 4):
    splits.append((i,))
  return f(splits)

def gen_splits(n):
  splits = gen_split(n)
  # EARLY EXIT
  return random.sample(splits, 5)
  return splits
  new_splits = []
  for s in splits:
    if s not in new_splits:
      new_splits.append(s)
    for new_split in gen_split(s[0]):
      i = new_split[0]
      if (s[0] // i > 1):
        new_s = (s[0] // i, i)
        if new_s not in new_splits:
          new_splits.append(new_s)
  return random.sample(new_splits, 10)
  return new_splits

count = 0
from tqdm import tqdm




def generate_loop_trees():
  df = pd.DataFrame(columns=['ir', 'loops_tensor', 'program_tensor', 'gflops'])
  count = 0
  examples_count = []

  for m, n, k in itertools.product(*[gen_iter() for _ in range(3)]):
    for ms in gen_splits(m):
      for ns in gen_splits(n):
        for ks in gen_splits(k):
          new_lt = gen_mm(m, n, k, ms, ks, ns)
          if new_lt is not None:
            count += 1            
            # breakpoint()
            df_example = pd.DataFrame(create_permutations_db(new_lt),columns=['ir', 'loops_tensor', 'program_tensor', 'gflops'])
            df = pd.concat([df, df_example])
            examples_count.append(df_example.shape[0])

            print(f" m: {m}, n: {n}, k: {k} [gen: {count}]", end="\r", flush=True)
            if count > 100:
              print(examples_count)
              return df
            


df = generate_loop_trees()
df.head()

breakpoint()
df.to_pickle("tensor_dataset.pkl")  

