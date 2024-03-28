#!/usr/bin/env python3

import sys

fname = sys.argv[-1]

from rPPG.utils import npio
import numpy as np
import time

print('Starting load')
starttime = time.time()

data, meta = npio.loadMeta(fname)
# Do some operation to force it all to be read
res = np.sum(data)

endtime = time.time()
print(f'Time to read {len(data)} length video{"" if not meta.gt() else " (meta length " + str(len(meta.gt())) + ")"}: {endtime-starttime:.4f}')
