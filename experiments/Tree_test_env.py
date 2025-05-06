import argparse
import os
import random
import subprocess
import time

import numpy as np

import torch
import torch.nn as nn

from kingdomtreeworking import bigBatch

if __name__ == "__main__":


    class Transpose(nn.Module):
        def __init__(self, permutation):
            super().__init__()
            self.permutation = permutation

        def forward(self, x):
            return x.permute(self.permutation)
        
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    class Agent(nn.Module):
        def __init__(self, mapsize=16 * 16):
            self.resources_ost = 5
            self.owner_ost = 11
            self.unit_types_ost = 13
            self.current_actions_ost = 21
            self.terrain_ost = 27
            self.x_dtype = None

            super(Agent, self).__init__()

            self.mapsize = mapsize
            h, w, c = (16,16,29)  # height, width, channels
            self.map_shape = (h, w, c)  # (height, width, channels)
            self.tree_output_a = 3  # 3 actions, 7 parameters per action
            self.tree_output_p = 7
            self.tree_output_c = self.tree_output_a * self.tree_output_p  # 21 channels for the tree output

            self.encoder = nn.Sequential(
                Transpose((0, 3, 1, 2)), # Transpose to (batch, channels, height, width)
                layer_init(nn.Conv2d(c+self.tree_output_c, 64, kernel_size=3, padding=1)), 
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, padding=1)), 
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 128, kernel_size=3, padding=1)),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.ReLU(),
                layer_init(nn.Conv2d(128, 256, kernel_size=3, padding=1)),
                nn.MaxPool2d(3, stride=2, padding=1),
            )

            self.tree_expander = nn.Sequential(
                Transpose((0, 3, 1, 2)), # Transpose to (batch, channels, height, width)
                layer_init(nn.Conv2d(self.tree_output_a * self.tree_output_p, 32, kernel_size=3, padding=1)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, self.tree_output_c, kernel_size=3, padding=1)),
                nn.ReLU(),
                Transpose((0, 2, 3, 1)),
            )

            self.actor = nn.Sequential(
                layer_init(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(32, 78, 3, stride=2, padding=1, output_padding=1)),
                Transpose((0, 2, 3, 1)),
            )
            
            self.critic = nn.Sequential(
                nn.Flatten(),
                layer_init(nn.Linear(256, 128)),
                nn.ReLU(),
                layer_init(nn.Linear(128, 1), std=1),
            )
            
            self.register_buffer("mask_value", torch.tensor(-1e8))

        def tree_transform(self,x):
            self.x_dtype = x.dtype  # Store the dtype of x for later use
            # for the gold value, take the boolean product of owner, base, and resource features, then sum over the map dimensions
            base_idx = self.unit_types_ost + 2  # base unit type index
            worker_idx = self.unit_types_ost + 4  # worker index
            barracks_idx = self.unit_types_ost + 3  # barracks unit type index
            light_idx = self.unit_types_ost + 5  # light unit type index
            heavy_idx = self.unit_types_ost + 6  # heavy unit type index
            ranged_idx = self.unit_types_ost + 7  # ranged unit type index
            resource_idx = self.unit_types_ost + 1  # resource unit type index
            resource_v = torch.tensor([1, 2, 3, 4], dtype=torch.float32).to(x.device)
            is_producing_idx = self.current_actions_ost + 4  # is_producing index


            #MAKE WORKER MASKS ZEROS INITIALLY. MAKE COORDINATES OF BASE ZERO TOO. CREATE WORKER MAPS. TEST



            # finds the owner feature map for the given owner and multiplies it with the unit feature map for the specified unit type
            # x[:, :, :, self.owner_ost + owner] is the owner feature map for the given owner
            # x[:, :, :, unit_idx] is the unit feature map for the specified unit type
            # the result is a tensor of shape (num_envs, height, width) where each channel corresponds to the unit type for the given owner
            def map_units_owner(x, unit_idx, owner):
                return torch.einsum('iab,iab->iab', x[:, :, :, unit_idx], x[:, :, :, self.owner_ost + owner])

            # Calculate the gold value by summing the product of owner, base, and resource features
            # x[:, :, :, owner_idx] is the owner feature map for the given owner
            # x[:, :, :, base_idx] is the base feature map
            # x[:, :, :, self.resources_ost+1:self.resources_ost+5] is the resource feature map
            # resource_v is a tensor that maps the resource features to their respective values

            returns = [None,None]
            resource_map = x[:, :, :, resource_idx]
            for owner in [0,1]:
                owner_idx = self.owner_ost + owner  # Owner index for the given owner

                if False:  # Debugging information
                    print("Shape of owner_x:", x[:, :, :, owner_idx].shape)
                    print("Shape of base_x:", x[:, :, :, base_idx].shape)
                    print("Shape of resource_x:", x[:, :, :, self.resources_ost+1:self.resources_ost+5].shape)
                    print("Shape of resource_v:", resource_v.shape)
                    print("Type of elements in x:", x.dtype)
                    print("Type of elements in resource_v:", resource_v.dtype)

                gold_value = torch.einsum('iab,iab,iabr,r->i', x[:, :, :, owner_idx].float(), x[:, :, :, base_idx].float(), x[:, :, :, self.resources_ost+1:self.resources_ost+5].float(), resource_v)
                

                base_map = map_units_owner(x, base_idx, owner)
                base_coordinates = torch.nonzero(base_map[0,:,:], as_tuple=False)  # Get the coordinates of the base for the given owner


                if True:  # Debugging information
                    print("Base coordinates for owner", owner, ":", base_coordinates)
                # Initialize the worker masks for both owners
                # These are one in the corner that contains the base and zero elsewhere
                worker_mask = torch.zeros((x.shape[0], x.shape[1], x.shape[2]), dtype=torch.float32, device=x.device)
                if base_coordinates.shape[0] == 0:
                    # Every worker is an attack worker, there is no base anymore
                    worker_mask[:, :, :] = 0.0
                elif base_coordinates[0, 0] > 8:
                    worker_mask[:, base_coordinates[0, 0]-1:, base_coordinates[0, 1]-1:] = 1.0  # Set the mask for the given owner
                else:
                    worker_mask[:, :base_coordinates[0, 0]+2, :base_coordinates[0, 1]+2] = 1.0


                    

                # Create the worker map by multiplying the worker feature map with the owner feature map for the given owner        
                worker_map = torch.einsum('iab,iab,iab->iab', x[:, :, :, worker_idx], x[:, :, :, owner_idx], worker_mask)

                attack_workers = torch.sum(torch.einsum('iab,iab,iab->iab', x[:, :, :, worker_idx], x[:, :, :, owner_idx], torch.ones_like(worker_mask)-worker_mask), (1,2))

                barracks_map = map_units_owner(x, barracks_idx, owner)


                light_units = torch.sum(map_units_owner(x, light_idx, owner),(1,2))
                heavy_units = torch.sum(map_units_owner(x, heavy_idx, owner),(1,2))
                ranged_units = torch.sum(map_units_owner(x, ranged_idx, owner),(1,2))
                workers_units = torch.sum(worker_map, (1, 2))
                barracks_units = torch.sum(barracks_map, (1, 2))
                available_barracks = torch.sum(
                    torch.einsum('iab,iab,iab->iab', x[:, :, :, owner_idx], x[:, :, :, barracks_idx], x[:, :, :, is_producing_idx]==0.0), (1, 2)
                )
                
                scalars = [gold_value,workers_units,attack_workers,light_units,heavy_units,ranged_units, barracks_units, available_barracks]
                # stack scalars to a tensor of shape (num_envs, 8)
                scalars = torch.stack(scalars, dim=1).to(torch.float32)

                # Print scalars of batch 0
                if True:  # Debugging information
                    print("Scalars for owner", owner, ":", scalars)  # Print the scalars for the first environment
                    print("Shape of scalars for owner", owner, ":", scalars.shape)
                    print("Type of elements in scalars for owner", owner, ":", scalars.dtype)
                    # print shape of maps
                    print("Shape of worker_map for owner", owner, ":", worker_map.shape)


                returns[owner] = [scalars.cpu(), worker_map.cpu(), barracks_map.cpu(), resource_map.cpu(), base_map.cpu()]
            
            
            return returns

        def econ_tree(self, x):
            """
            :param x: input tensor of shape (num_envs, height, width, channels+1)
            :return: tree-augmented state of shape (num_envs, height, width, channels+1)
            """

            tree_input = self.tree_transform(x)  # Transform the input

            # tree_input is a tuple of (scalars, worker_map, barracks_map, obstacle_map) for each owner
            
            tree_vector = bigBatch(tree_input)  # (num_envs, tree_output_c)
            

            tree_output = tree_vector.repeat(self.map_shape[0], self.map_shape[1], 1, 1).permute(2,0,1,3)  # Batch, Height, Width, Channels
            tree_output = tree_output.to(x.device)  # Ensure the tensor is on the same device as x
            tree_output = tree_output.to(x.dtype)  # Ensure the tensor has the same dtype as x

            if False:  # Debugging information
                print("Shape of tree_vector:", tree_vector.shape)
                print("Shape of tree_output:", tree_output.shape)
                print("Type of tree_output:", tree_output.dtype)
                print("Shape of x:", x.shape)

            return self.tree_expander(tree_output)

        def augment_with_tree(self, x):
            """
            :param x: input tensor of shape (num_envs, height, width, channels)
            :return: tree-augmented state of shape (num_envs, height, width, channels+1)
            """
            tree_out = self.econ_tree(x)

            if False:  # Debugging information
                print("Shape of tree_out:", tree_out.shape)  # Should be (num_envs, height, width, tree_output_c)
                print("Type of tree_out:", tree_out.dtype)
                print("Shape of x before concatenation:", x.shape)
            x = torch.cat((x, tree_out), dim=3)  # Concatenate the tree output to the input
            if False:  # Debugging information
                print("Shape of x after concatenation:", x.shape)  # Should be (num_envs, height, width, channels + tree_output_c)
            return x

        def get_action_and_value(self, x, action=None, invalid_action_masks=None, envs=None, device=None):
            """
            :return:
                (1) action (shape = [num_envs, width*height, 7], where 7 = dimensionality of per-unit action)
                (2) log probability of action (shape = [num_envs])
                (3) entropy (shape = [num_envs])
                (4) invalid action masks
                (5) Critic's prediction
            """
            tree_augmented_state = self.augment_with_tree(x)
            hidden = self.encoder(tree_augmented_state)
            logits = self.actor(hidden)
            return logits, None, None, None, self.get_value(x)

        def get_value(self, x):
            return self.critic(self.encoder(self.augment_with_tree(x)))
        



    def test_agent():
        """
        Test the agent with a dummy input tensor.
        :param agent: Agent instance
        :param dummy_input: Dummy input tensor of shape (num_envs, height, width, channels)
        :return: None
        """
        # Define agent
        agent = Agent()

        # create a dummy input tensor to test the forward pass
        dummy_input = torch.zeros((20, 16, 16, 29), dtype=torch.float32)  # Batch size of 20, height and width of 16, channels of 29

        # Set some random values for testing

        # base at (5,5), worker at (4,4) for player 1
        dummy_input[0:20, 5, 5, 11] = 1.0  # Ownership for player 1
        dummy_input[0:20, 3, 3, 15] = 1.0  # Base for player 1
        dummy_input[0:20, 3, 3, 9] = 1.0  # Resource for player 1
        dummy_input[0:20, 4, 4, 11] = 1.0  # Ownership for player 1
        dummy_input[0:20, 4, 4, 17] = 1.0  # Worker for player 1

        # attack worker at (8,8) for player 1
        dummy_input[0:20, 8, 8, 11] = 1.0  # Ownership for player 1
        dummy_input[0:20, 8, 8, 17] = 1.0  # Worker for player 1

        # resource at (0,0) and (15,15) for null player
        dummy_input[0:20, 0, 0, 10] = 1.0  # Resource for null player
        dummy_input[0:20, 0, 0, 14] = 1.0  # Resource for null player
        dummy_input[0:20, 15, 15, 10] = 1.0  # Resource for null player
        dummy_input[0:20, 15, 15, 14] = 1.0  # Resource for null player

        # heavy unit at (5,7) for player 1
        dummy_input[0:20, 5, 7, 11] = 1.0  # Ownership for player 1
        dummy_input[0:20, 5, 7, 19] = 1.0  # Heavy unit for player 1

        # base at (13,13), worker at (14,14) for player 2
        dummy_input[0:20, 13, 13, 12] = 1.0  # Ownership for player 2
        dummy_input[0:20, 13, 13, 15] = 1.0  # Base for player 2
        dummy_input[0:20, 13, 13, 9] = 1.0  # Resource for player 2

        dummy_input[0:20, 14, 14, 12] = 1.0  # Ownership for player 2
        dummy_input[0:20, 14, 14, 17] = 1.0  # Worker for player 2

        # heavy unit at (10,10) for player 2
        dummy_input[0:20, 10, 10, 12] = 1.0  # Ownership for player 2
        dummy_input[0:20, 10, 10, 19] = 1.0  # Heavy unit for player 2

        if False:  # Debugging information
            print("Dummy input created with shape:", dummy_input.shape)
            # print one sample of the dummy input
            print("Sample dummy input:", dummy_input[0,:,:,0])  # Print first 5 channels of the first sample

            # print size of encoder output
            print("Encoder output size:", agent.encoder_old(dummy_input).shape)
            # print size of encoder output after tree augmentation
            print("Encoder output size after tree augmentation:", agent.encoder(agent.augment_with_tree(dummy_input)).shape)


        # Forward pass through the agent
        output = agent.get_action_and_value(dummy_input)
        print("Forward pass successful. Output shape:", output[0].shape)
    
    test_agent()  # Test the agent with a dummy input tensor

        

