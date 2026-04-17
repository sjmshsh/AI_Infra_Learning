
import numpy as np

def run_fake_model(seqs, max_len: int = 15, min_val: int = -1, max_val: int = 999, eos: int=-1):
  token_ids = [eos if len(seq) >= max_len else np.random.randint(min_val, max_val + 1) for seq in seqs]
  return token_ids

