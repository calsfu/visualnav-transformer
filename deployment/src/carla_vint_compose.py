import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import matplotlib.pyplot as plt
import yaml
import pygame

import sys
# sys.path.append("/home/coler/Desktop/cole_vint/visualnav-transformer/train/vint_train/training")
# from train_utils import get_action
import torch
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time
import random
import math
from utils import to_numpy, transform_images, load_model

# UTILS
from topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC)

# carla
import glob
import os
import logging 

try:
    sys.path.append('/home/coler/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg')
except IndexError:
    pass

import carla

# CONSTANTS
MODEL_WEIGHTS_PATH = "../model_weights"
MODEL_CONFIG_PATH = "../config/models.yaml"
EPS = 1e-8
MAX_SPEED = 10
MAX_STEERING_ANGLE = 0.610865

# GLOBALS
subgoal = []

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# camera_tick = 1.5
# camera_pixels_x = 320 #1280
# camera_pixels_y = 240 #800
# camera_horiz_fov = 72

camera_hz = 4
camera_freq = 1. / float(camera_hz)
camera_pixels_x = 160 #320 * 2 #1280
camera_pixels_y = 120 #240 * 2 #800
camera_horiz_fov = 72


class CarlaVint:
    def __init__(self,
                carla_port = 2000):
        
        #config
        self.dt = 0.05
        self.close_threshold = 3
        self.command = 1

        # connect to client
        self.client = carla.Client('localhost', carla_port)
        self.client.set_timeout(4.0)

        # create world and map
        self.world = self.client.load_world('Town01')
        # self.world = self.client.get_world()
        self.map = self.world.get_map()

        # Spawn ego vehicle
        ego_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name','ego')
        print('\nEgo role_name is set')
        ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
        ego_bp.set_attribute('color',ego_color)
        print('\nEgo color is set')

        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if 0 < number_of_spawn_points:
            ego_transform = random.choice(spawn_points)
            self.vehicle = self.world.spawn_actor(ego_bp,ego_transform)
            print('\nEgo is spawned')
        else: 
            logging.warning('Could not found any spawn points')

        spector = self.world.get_spectator()
        spector.set_transform(self.vehicle.get_transform())

        # Spawn camera
        cam_bp = None
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(camera_pixels_x))
        cam_bp.set_attribute("image_size_y",str(camera_pixels_y))
        cam_bp.set_attribute("fov",str(camera_horiz_fov))
        # cam_bp.set_attribute("sensor_tick",str(camera_tick))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        self.ego_lone = self.world.spawn_actor(cam_bp,cam_transform,attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)

        cam_bp = None
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        cam_bp.set_attribute("image_size_x",str(camera_pixels_x))
        cam_bp.set_attribute("image_size_y",str(camera_pixels_y))
        cam_bp.set_attribute("fov",str(camera_horiz_fov))
        cam_bp.set_attribute("sensor_tick",str(camera_freq))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        self.ego_depth = self.world.spawn_actor(cam_bp,cam_transform,attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)

        cam_bp = None
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        cam_bp.set_attribute("image_size_x",str(camera_pixels_x))
        cam_bp.set_attribute("image_size_y",str(camera_pixels_y))
        cam_bp.set_attribute("fov",str(camera_horiz_fov))
        cam_bp.set_attribute("sensor_tick",str(camera_freq))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        self.ego_seg = self.world.spawn_actor(cam_bp,cam_transform,attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        
        # lane invasion sensor
        lane_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        lane_location = carla.Location(0,0,0)
        lane_rotation = carla.Rotation(0,0,0)
        lane_transform = carla.Transform(lane_location,lane_rotation)
        self.lane_sensor = self.world.spawn_actor(lane_bp,lane_transform,attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)

        self.is_invaded = False
        self.count = 0
        self.invade_count = 0
        def save_lane(event):
            self.is_invaded = True
            # print("Lane invasion")

        self.lane_sensor.listen(lambda event: save_lane(event))

        self.camera_display = self.world.spawn_actor(
                self.world.get_blueprint_library().find('sensor.camera.rgb'),
                carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                attach_to=self.vehicle)
        
        self.saved_depth = None
        self.saved_seg = None
        self.saved_lone = None

        def save_lone(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape(
                image.height, image.width, -1)
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            # image = PILImage.fromarray(array)
            self.saved_lone = array

        

        def save_depth(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            depth = np.empty([array.shape[0], array.shape[1], array.shape[2]])
            depth[:,:,0] = ((array[:,:,0] + array[:,:,1] * 256.0 + array[:,:,2] * 256.0 * 256.0)/((256.0 * 256.0 * 256.0) - 1))
            depth[:,:,1] = depth[:,:,0]
            depth[:,:,2] = depth[:,:,0]
            depth = depth * 1000 # farthest depth that can be seen is 1km
            depth = np.clip(depth, 0, 12) # farthest distance the oakd can infer is 12m
            depth = depth * 255 / 12
            self.saved_depth = depth

        def save_seg(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            array = array * (255 / 28) # max object id
            self.saved_seg = array

        def update_display(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape(
                image.height, image.width, -1)
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            array = array.swapaxes(0, 1)
            surface = pygame.surfarray.make_surface(array)
            self.display.blit(surface, (0, 0))
            self.display.blit(self.font.render(f'{self.command_dict[self.command]}', True, (0,0,0)), (10, 10))

        self.ego_lone.listen(lambda image: save_lone(image))
        self.ego_depth.listen(lambda image: save_depth(image))
        self.ego_seg.listen(lambda image: save_seg(image))
        
        self.camera_display.listen(lambda image: update_display(image))

        # render at get keyboard control
        pygame.init()
        self.display = pygame.display.set_mode((camera_pixels_x * 4,4 * camera_pixels_y), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 30)
        self.command_dict = {0: "LEFT", 1: "STRAIGHT", 2: "RIGHT"}


        # VINT SETUP
        # global context_size
        self.context_queue = []
        self.context_size = None  

        # load model parameters
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_paths = yaml.safe_load(f)

        model_config_path = model_paths["vint"]["config_path"]
        with open(model_config_path, "r") as f:
            self.model_params = yaml.safe_load(f)

        self.context_size = self.model_params["context_size"]

        # load model weights
        ckpth_path = '../model_weights/composed_4hr.pth'
        if os.path.exists(ckpth_path):
            print(f"Loading model from {ckpth_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
        self.model = load_model(
            ckpth_path,
            self.model_params,
            device,
        )
        self.model = self.model.to(device)
        self.model.eval()


    def step(self):
        # print(len(self.context_queue))
        pygame.event.clear()

        if len(self.saved_depth ) == 0 or len(self.saved_seg) == 0:
            return
        img_comp = np.empty([self.saved_lone.shape[0], self.saved_lone.shape[1], self.saved_lone.shape[2]])
        img_comp[:,:,0] = np.mean(self.saved_lone, axis=2)
        img_comp[:,:,1] = np.mean(self.saved_depth, axis=2)
        img_comp[:,:,2] = self.saved_seg[:,:,0]
        self.context_queue.append(PILImage.fromarray(img_comp.astype(np.uint8)))
        if(len(self.context_queue) > self.model_params["context_size"] + 1):
            self.context_queue.pop(0)

        if len(self.context_queue) > self.model_params["context_size"]:
            transf_obs_imgs = transform_images(self.context_queue, self.model_params["image_size"]).to(device)

            # save first image from transf_obs_imgs
            # if self.count % 10 == 0:
            #     img = transf_obs_imgs[0].cpu().numpy().transpose(1, 2, 0)
            #     img = (img * 255).astype(np.uint8)
            #     img = PILImage.fromarray(img)
                # img.save(f'./data/9{self.count}.jpg')

            # if self.count % 10 == 0:
            #     self.context_queue[0].save(f'./data/9{self.count}.jpg')


            # events  = pygame.event.get()
            # if events:
            #     if events[-1].type == pygame.QUIT:
            #         pygame.quit()
            #         sys.exit()

            #shape of transf obs
            # print(transf_obs_imgs.shape)
            

            # get continous key press of left arrow, up arrow, right arrow
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.command = 0
            elif keys[pygame.K_RIGHT]:
                self.command = 2
            else:
                self.command = 1

            goal_data = torch.tensor(self.command).to(device)

            
            # transf_obs_imgs = torch.cat(transf_obs_imgs, dim=0).to(device)
            # goal_data = torch.cat(goal_data, dim=0).to(device)
            distances, waypoints = self.model(transf_obs_imgs, goal_data)

            distances = to_numpy(distances)
            waypoints = to_numpy(waypoints)
            
            # look for closest node and choose the middle waypoint
            min_dist_idx = np.argmin(distances)
           
            # chose subgoal and output waypoints
            # if distances[min_dist_idx] > self.close_threshold:
            chosen_waypoint = waypoints[min_dist_idx][2]
            # else:
            #     chosen_waypoint = waypoints[min(
            #         min_dist_idx + 1, len(waypoints) - 1)][2]
            print(waypoints)
            speed, steer = self._ackerman_control(chosen_waypoint)
            # print(f"Speed: {speed}, Steer: {steer}")

            ackerman = carla.VehicleAckermannControl(steer=steer,speed=speed)
            self.vehicle.apply_ackermann_control(ackerman)

        # self.world.wait_for_tick(0.05) # wait for next camera update
        self.world.tick()
        self.clock.tick()
        self.count += 1
        if self.is_invaded:
            self.invade_count += 1
            self.is_invaded = False
        pygame.display.flip()
        pygame.event.clear()
        # time.sleep(.5)

    def game_loop(self):
        try:
            while True:
                self.step()
        finally:
            print("Destroying actors")
            self.vehicle.destroy()
            self.ego_lone.destroy()
            self.camera_display.destroy()
            pygame.quit()
            if self.count > 0:
                print(self.invade_count/self.count)
    
    def _ackerman_control(self, waypoint):
        """
        Compute the speed and steering angle to move the car toward the relative waypoint.
        
        Parameters:
            car_pos (tuple): (x, y) position of the car.
            car_heading (float): Current heading angle of the car (in radians).
            relative_waypoint (tuple): (dx, dy) relative waypoint in the car's local frame.

        Returns:
            speed (float): Speed to move the car.
            steering_angle (float): Steering angle for Ackerman steering.
        """
        # Extract relative waypoint in local frame
        if len(waypoint) == 2:
            dx, dy = waypoint
        else:
            dx, dy, hx, hy = waypoint

        # Compute distance to the waypoint
        distance = math.sqrt(dx**2 + dy**2)

        # Target angle in the local frame
        target_angle = math.atan2(dy, dx)

        # Compute heading error
        heading_error = target_angle  # Since it's in the local frame, no need to subtract car_heading

        # Compute speed (proportional to distance)
        speed = min(MAX_SPEED, distance * 5)

        # Compute steering angle (proportional to heading error)
        steering_angle = max(-MAX_STEERING_ANGLE, min(MAX_STEERING_ANGLE, heading_error))

        return speed, steering_angle



def main():
    env = CarlaVint()
    env.game_loop()

if __name__ == '__main__':
    main()