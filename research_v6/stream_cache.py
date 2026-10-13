"""Exact immutable cache of common-random-number observation streams.

This affects generation cost only: each action's nth chunk is produced by the
original simulator, then reused for another method on precisely the same data.
The evaluator truth remains isolated inside the simulator.
"""
from collections import OrderedDict
import numpy as np
from . import likelihood,sequential
Original=likelihood.GaussianStream
DATA=OrderedDict()
class CachedStream:
    def __init__(self,f,truth,seed_value,model_constants):
        self.key=(f.hash,truth,seed_value);self.used=[0]*4
        if self.key not in DATA:
            DATA[self.key]=(Original(f,truth,seed_value,model_constants),[[] for _ in range(4)])
        DATA.move_to_end(self.key)
        while len(DATA)>4:DATA.popitem(last=False)
        self.data=DATA[self.key]
    def chunk(self,a,n):
        source,chunks=self.data;k=self.used[a]
        if k==len(chunks[a]):chunks[a].append(source.chunk(a,n))
        result=chunks[a][k]
        if len(result[0])!=n:raise ValueError('chunk size changed for identical stream')
        self.used[a]+=1;return result
def install():
    likelihood.GaussianStream=CachedStream;sequential.GaussianStream=CachedStream
def verify(f,constants,seed):
    a=Original(f,0,seed,constants);b=CachedStream(f,0,seed,constants)
    for act in (3,1,3,0,2):
        x,e=a.chunk(act,256);xx,ee=b.chunk(act,256)
        if not np.array_equal(x,xx) or not np.array_equal(e,ee):raise AssertionError('cached stream differs')
    c=CachedStream(f,0,seed,constants)
    aa=Original(f,0,seed,constants)
    for act in (3,1,3,0,2):
        x,e=aa.chunk(act,256);xx,ee=c.chunk(act,256)
        if not np.array_equal(x,xx) or not np.array_equal(e,ee):raise AssertionError('reused stream differs')
    return True
