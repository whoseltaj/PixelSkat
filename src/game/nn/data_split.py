# File: src/game/nn/data_split.py

import os
import numpy as np

def main(in_npz="data/datask.npz",
         out_dir="data",
         train_frac=0.70,
         val_frac=0.15):

    arr = np.load(in_npz)
    X, Y = arr["X"], arr["Y"]
    N = X.shape[0]
    print(f"Loaded {N} examples from {in_npz}")

    rng = np.random.default_rng(seed=42)
    idx = rng.permutation(N)

    n_train = int(train_frac * N)
    n_val   = int(val_frac  * N)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train : n_train + n_val]
    test_idx  = idx[n_train + n_val :]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val,   Y_val   = X[val_idx],   Y[val_idx]
    X_test,  Y_test  = X[test_idx],  Y[test_idx]

    print("Split sizes:")
    print(f"  train: {len(train_idx)}")
    print(f"  val:   {len(val_idx)}")
    print(f"  test:  {len(test_idx)}")

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "train.npz"), X=X_train, Y=Y_train)
    np.savez(os.path.join(out_dir, "val.npz"),   X=X_val,   Y=Y_val)
    np.savez(os.path.join(out_dir, "test.npz"),  X=X_test,  Y=Y_test)
    print(f"Saved train/val/test to {out_dir}/{{train,val,test}}.npz")

if __name__ == "__main__":
    main()
