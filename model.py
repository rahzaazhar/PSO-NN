import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from utils import vecToMask


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5*5*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.conv_to_mask = vecToMask([0,20,50,500,10])
        self.mask_unsqueeze = lambda x : x.unsqueeze(-1).unsqueeze(-1)
        self.process_mask = lambda x : 10*F.tanh(x)
        self.mask_len = sum([20,50,500,10])

    def forward(self, x, pos):
        #pos = self.process_mask(pos)
        #mask = self.conv_to_mask(pos)
        x = F.relu(self.conv1(x))
        #x = x*self.mask_unsqueeze(mask[0])
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        #x = x*self.mask_unsqueeze(mask[1])
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))
        #x = x*mask[2]
        x = self.fc2(x)
        #x = x*mask[3]
        return x
    
    def name(self):
        return "LeNet"


'''class PSO(object):
    def __init__(self,particles,inertia,social_w,cog_w):
        self.particles = particles
        self.velocities = torch.zeros_like(particles.size())
        self.local_best_postions = particles
    
    def step(self):
        self.velocities = ((self.inertia*self.velocity) +
                            (self.c_cognitive*torch.rand(1)*(self.particle_best-self.position)) +
                            (self.c_social*torch.rand(1)*(self.global_best-self.position)))
        self.position = self.position + self.velocity
    

train i steps until condition met
Create and initialise particles

def evaluate_fitness(particles,model,train_loader):
'''
     
