#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:07:12 2020

@author: andrew
"""
import numpy as np

fakedigit = np.array(np.random.randint(0,255,size=(784)))

poss_indices = np.array([-58,-57,-56,-55,-54,-26,2,30,58,57,56,55,54,26,-2,-30])
poss_indices = poss_indices + 320

old_indices = np.array([265,266,294])
other_old_indices = np.array([0,0,0,0])

np.concatenate((old_indices, other_old_indices))

#print(old_indices)

poss_indices2 = poss_indices[poss_indices>=0]

#poss_indices2 = poss_indices[np.invert(np.in1d(poss_indices, old_indices))]
#poss_indices3 = poss_indices2[poss_indices2>=0]

values = fakedigit[poss_indices2]
negate = sum(np.in1d(poss_indices, old_indices))
values[negate]=0

negate = negate + 1
print(negate)
#print(poss_indices2)
#print(negate)
#print(values)


maxind = np.where(values == values.max())
newind = poss_indices2[maxind[0]]

print(maxind)
print(newind)
print(values)
print(poss_indices2)

#poss_indices_2 = np.concatenate(poss_indices, old_indices)


check = np.isin(poss_indices, 322)
sum(check)
#print(poss_indices)
#print(poss_indices2)
#print(poss_indices3)


circle_indices = np.array([-58,-57,-56,-55,-54,-26,2,30,58,57,56,55,54,26,-2,-30])
nearby_indices = np.array([-29,-28,-27,-1,0,1,27,28,29])
negate_indices = np.concatenate((nearby_indices, circle_indices))
print(negate_indices)

sum(np.isin(negate_indices, 2))


newind=[4]
rand = np.random.randint(0,len(newind),size=(1))
print(int(rand))
print(newind[int(rand)])