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
MAX_V = 10
MAX_W = .5

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

        # connect to client
        self.client = carla.Client('localhost', carla_port)
        self.client.set_timeout(4.0)

        # create world and map
        self.world = self.client.load_world('Town01')
        self.map = self.world.get_map()

        self.saved_left = None
        self.saved_right = None
        self.saved_mid = None

        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        # Spawn ego vehicle
        ego_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name','ego')
        print('\nEgo role_name is set')
        ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
        ego_bp.set_attribute('color',ego_color)
        print('\nEgo color is set')

        if 0 < number_of_spawn_points:
            #random.shuffle(spawn_points)
            ego_transform = spawn_points[0]
            self.vehicle = self.world.spawn_actor(ego_bp,ego_transform)
            print('\nEgo is spawned')
        else: 
            logging.warning('Could not found any spawn points')

        spector = self.world.get_spectator()
        spector.set_transform(self.vehicle.get_transform())

        # Spawn ego vehicle
        cam_bp = None
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(camera_pixels_x))
        cam_bp.set_attribute("image_size_y",str(camera_pixels_y))
        cam_bp.set_attribute("fov",str(camera_horiz_fov))
        # cam_bp.set_attribute("sensor_tick",str(camera_freq))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,360 - 72,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        self.ego_left = self.world.spawn_actor(cam_bp,cam_transform,attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)

        cam_bp = None
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(camera_pixels_x))
        cam_bp.set_attribute("image_size_y",str(camera_pixels_y))
        cam_bp.set_attribute("fov",str(camera_horiz_fov))
        # cam_bp.set_attribute("sensor_tick",str(camera_freq))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        self.ego_mid = self.world.spawn_actor(cam_bp,cam_transform,attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)

        cam_bp = None
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(camera_pixels_x))
        cam_bp.set_attribute("image_size_y",str(camera_pixels_y))
        cam_bp.set_attribute("fov",str(camera_horiz_fov))
        # cam_bp.set_attribute("sensor_tick",str(camera_freq))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,72,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        self.ego_right = self.world.spawn_actor(cam_bp,cam_transform,attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)

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

        def save_left(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.saved_left = array

        def save_mid(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.saved_mid = array

        def save_right(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.saved_right = array

        self.ego_left.listen(lambda image: save_left(image))
        self.ego_mid.listen(lambda image: save_mid(image))
        self.ego_right.listen(lambda image: save_right(image))

        self.camera_display = self.world.spawn_actor(
                self.world.get_blueprint_library().find('sensor.camera.rgb'),
                carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                attach_to=self.vehicle)
        
        def save_lone(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape(
                image.height, image.width, -1)
            array = array[:, :, :3]
            image = PILImage.fromarray(array)
            # if self.context_size is not None:
            #     if len(self.context_queue) < self.context_size + 1:
            #         self.context_queue.append(image)
            #     else:
            #         self.context_queue.pop(0)
            #         self.context_queue.append(image)

        def update_display(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).reshape(
                image.height, image.width, -1)
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            array = array.swapaxes(0, 1)
            surface = pygame.surfarray.make_surface(array)
            self.display.blit(surface, (0, 0))

        self.ego_lone.listen(lambda image: save_lone(image))
        self.camera_display.listen(lambda image: update_display(image))

        # render at get keyboard control
        pygame.init()
        self.display = pygame.display.set_mode((camera_pixels_x * 4,4 * camera_pixels_y), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

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
        ckpth_path = '../model_weights/wide.pth'
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
        if self.saved_left is None or self.saved_mid is None or self.saved_right is None:
            return
        self.context_queue.append(PILImage.fromarray(np.concatenate([self.saved_left, self.saved_mid, self.saved_right], axis=1)[:,::3,:]))
        if(len(self.context_queue) > self.model_params["context_size"] + 1):
            self.context_queue.pop(0)
        if len(self.context_queue) > self.model_params["context_size"]:
            
            transf_obs_imgs = transform_images(self.context_queue, self.model_params["image_size"]).to(device)

            # events  = pygame.event.get()
            # if events:
            #     if events[-1].type == pygame.QUIT:
            #         pygame.quit()
            #         sys.exit()
            
            

            # one hot of size 3
            # goal_data = torch.zeros((1, 3), dtype=torch.int64)

            # get continous key press of left arrow, up arrow, right arrow
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                goal_data = 0
            elif keys[pygame.K_RIGHT]:
                goal_data = 2
            else:
                goal_data = 1
            

            goal_data = torch.tensor(goal_data).to(device)

            
            # transf_obs_imgs = torch.cat(transf_obs_imgs, dim=0).to(device)
            # goal_data = torch.cat(goal_data, dim=0).to(device)
            # print(transf_obs_imgs.shape)
            distances, waypoints = self.model(transf_obs_imgs, goal_data)

            distances = to_numpy(distances)
            waypoints = to_numpy(waypoints)
            
            # look for closest node and choose the middle waypoint
            min_dist_idx = np.argmin(distances)
           
            # chose subgoal and output waypoints
            if distances[min_dist_idx] > self.close_threshold:
                chosen_waypoint = waypoints[min_dist_idx][1]
            else:
                chosen_waypoint = waypoints[min(
                    min_dist_idx + 1, len(waypoints) - 1)][1]
                
            speed, steer = self._controller(chosen_waypoint)
            print(f"Speed: {speed}, Steer: {steer}")

            ackerman = carla.VehicleAckermannControl(steer=steer,speed=speed)
            self.vehicle.apply_ackermann_control(ackerman)

        # self.world.wait_for_tick(0.05) # wait for next camera update
        self.world.tick()
        self.clock.tick()
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
            self.ego_left.destroy()
            self.ego_mid.destroy()
            self.ego_right.destroy()
            pygame.quit()

    def _controller(self, waypoint):
        assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
        if len(waypoint) == 2:
            dx, dy = waypoint
        else:
            dx, dy, hx, hy = waypoint
        # this controller only uses the predicted heading if dx and dy near zero
        if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
            v = 0
            w = np.clip(np.arctan2(hy, hx), -np.pi, np.pi)/self.dt		
        elif np.abs(dx) < EPS:
            v =  0
            w = np.sign(dy) * np.pi/(2*self.dt)
        else:
            v = dx / self.dt
            w = np.arctan(dy/dx) / self.dt
        v = np.clip(v, 0, MAX_V)
        w = np.clip(w, -MAX_W, MAX_W)
        return v, w



def main():
    env = CarlaVint()
    env.game_loop()

if __name__ == '__main__':
    main()