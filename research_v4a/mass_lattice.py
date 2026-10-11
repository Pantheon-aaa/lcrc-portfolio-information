"""Check whether a numerical mass interval can hide another discrete mass level."""
from functools import lru_cache
from bisect import bisect_right
import numpy as np

@lru_cache(maxsize=128)
def lattice(probabilities):
    ratios=[float(x).as_integer_ratio() for x in probabilities];D=max(d for n,d in ratios);coefficients=tuple(n*(D//d) for n,d in ratios)
    half=len(coefficients)//2;parts=[]
    for block in (coefficients[:half],coefficients[half:]):
        values=[0]
        for c in block:values += [v+c for v in values]
        parts.append(sorted(set(values)))
    return coefficients,parts[0],parts[1],sum(coefficients)

def exact_primary_level(p,success_mask,objective_upper):
    if len(p)>24:return False
    coefficients,left,right,total=lattice(tuple(map(float,p)));incumbent=sum(c for c,yes in zip(coefficients,success_mask) if yes)
    if incumbent==total:return True
    next_value=None
    for a in left:
        i=bisect_right(right,incumbent-a)
        if i<len(right):
            value=a+right[i]
            if next_value is None or value<next_value:next_value=value
    if next_value is None:return True
    # Include conversion/normalization roundoff; compare exact integers thereafter.
    upper=float(objective_upper)+8*np.finfo(float).eps
    n,d=upper.as_integer_ratio()
    return next_value*d>n*total
