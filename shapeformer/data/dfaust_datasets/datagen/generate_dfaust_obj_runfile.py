import numpy as np
import h5py
import os
from config import DFAUST_dir

tdir = f"{DFAUST_dir}/data"
if not os.path.exists(tdir):
    os.mkdir(tdir)

fns = [f"{DFAUST_dir}/registrations_m.hdf5", f"{DFAUST_dir}/registrations_f.hdf5"]

lines = []
for fn in fns:
    with h5py.File(fn, "r") as f:
        for key in f.keys():
            if key=="faces":
                continue
            sid = key.split("_")[0]
            aid = "_".join( key.split("_")[1:] )
            lines.append(f"python write_sequence_to_obj.py --path {fn} --seq {aid} --sid {sid} --tdir {tdir}")
            print(lines[-1])
with open('generate_dfaust_obj_all.sh', 'w') as f:
    print(*lines, sep="\n", file=f)


