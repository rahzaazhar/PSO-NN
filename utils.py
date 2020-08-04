import torch
import torchvision
import torch.nn as nn
import itertools

#function to generate masks given a model
#function to apply mask to model
#function to convert between vector and mask

#PSO class 
#no of particles, particle dim, func call(fitness)/nn module,
# perform one PSO step
# attr1 -> global pos
# attr2 ->  
# attr3 -> 


# Vanilla B x C x H x W
# PSO P x (B x C x H x W)
# 



class vecToMask(object):
    '''
    slices -> Dx1 array
    '''
    def __init__(self,slices=None):
        assert  slices is not None
        self.slices = slices
    def __call__(self,vector):
        return [vector[...,slice_start:slice_start+self.slices[i+1]] for i, slice_start in enumerate(itertools.accumulate(self.slices[:-1]))]
 


def test_vec_to_mask():
    pos = torch.arange(570)
    vtoM = vecToMask([0,20,50,500])
    masks = vtoM(pos)
    print(len(masks))
    print(masks[0])

if __name__ == "__main__":
    test_vec_to_mask()


