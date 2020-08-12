import torch
import numpy as np

class Swarm(object):
    def __init__(self,model,num_particles,fitness_func):
        self.num_particles = num_particles
        self.model = model
        self.particle_dim = model.mask_len
        self.particles = torch.nn.Parameter(torch.randn(num_particles,self.particle_dim))
        self.fitness_func = fitness_func
    
    def evaulate(self,batch):
        x, y = batch
        fitness_list = []
        for pos in self.particles:
            pred = self.model(x,pos)
            fitness = self.fitness_func(pred,y)
            fitness_list.append(fitness)
        self.fitness_list = torch.stack(fitness_list,dim=0)

        


class PSO(object):
    def __init__(self,particles,cognitive_constant,social_constant,inertia):
        self.cognitive_constant = cognitive_constant
        self.social_constant = social_constant
        self.inertia = inertia
        self.social_fitness = np.inf
        self.social_position = None
        self.cognitive_fitness = None
        self.cognitive_postions = particles
        self.velocity = torch.ones_like(particles)
        self.positions = particles
    
    
    def __call__(self,fitness_list):
        if self.cognitive_fitness is None:
            self.cognitive_fitness = fitness_list

        cognitive = self.cognitive_constant*torch.rand(1)*(self.cognitive_postions-self.positions)
        social = self.social_constant*torch.rand(1)*(self.social_position-self.positions)
        self.velocity = self.inertia*self.velocity+cognitive+social
        self.positions.add_(self.velocity)

        local_mask = self.fitness_list < self.cognitive_fitness
        self.cognitive_postions[local_mask] = self.positions[local_mask]
        self.cognitive_fitness[local_mask] = fitness_list[local_mask]
        if (fitness_list.min() < self.social_fitness):
            self.social_fitness = fitness_list.min()
            self.social_position = self.positions[fitness_list.argmin()]




