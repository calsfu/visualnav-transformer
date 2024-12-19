import numpy as np
import sys
import os
import random
import time
import pickle

try:
    sys.path.append('/home/coler/CARLA_0.9.15/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg')
except IndexError:
    pass

import carla

class CollectData:
    def __init__(self, carla_port = 2000):
        '''
        Want a dataset with this structure
        traj1
            img1
            img2
            ...
            imgN
            traj_data.pkl
        traj2
            img1
            img2
            ...
            imgN
            traj_data.pkl
        N = number of images
        traj_data.pkl contains the following:
            [position] : [N, 2] np array of xy coordinates
            [yaw] : [N,] np array of yaw angles
        '''
        self.dt = 0.05
        self.carla_port = carla_port
        self.client = carla.Client('localhost', self.carla_port)
        self.client.set_timeout(4.0)
        self.world = self.client.load_world('Town01')
        self.blueprint_library = self.world.get_blueprint_library()
        self.command_threshold = 0.05
        self.vehicle = None
        self.camera = None
        self.image = None
        self.traj_data = {
            'position': [],
            'yaw': [],
            'command' : []
        }
        self.count = 0
        self.time = time.time()
        try:
            os.makedirs(f'./data/{self.time}')
        except FileExistsError:
            pass  # Directory already exists
        self.spawn_ego_vehicle()
        self.spawn_camera()

    def spawn_ego_vehicle(self):
        spawn_points = self.world.get_map().get_spawn_points()
        ego_transform = random.choice(spawn_points)

        if self.vehicle is None:
            ego_bp = self.blueprint_library .find('vehicle.tesla.model3')
            ego_bp.set_attribute('role_name','ego')
            print('\nEgo role_name is set')
            ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
            ego_bp.set_attribute('color',ego_color)
            print('\nEgo color is set')
            self.vehicle = self.world.spawn_actor(ego_bp, ego_transform)
            # set autopilot
            self.vehicle.set_autopilot(True)
        else:
            self.vehicle.set_target_velocity(carla.Vector3D(0,0,0))
            self.vehicle.set_target_angular_velocity(carla.Vector3D(0,0,0))
            self.vehicle.set_transform(ego_transform)
        
        # set spectator cam
        spectator = self.world.get_spectator()
        spectator.set_transform(ego_transform)
        

    def spawn_camera(self):
        # spawn camera
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(320))
        camera_bp.set_attribute('image_size_y', str(240))
        camera_bp.set_attribute('fov', str(90))
        camera_bp.set_attribute('sensor_tick', '.25')  
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda image: self.save_data(image))

    def save_data(self, image):
        position = self.vehicle.get_location()
        yaw = self.vehicle.get_transform().rotation.yaw
        self.traj_data['position'].append([position.x, position.y])
        self.traj_data['yaw'].append(yaw)
        self.traj_data['command'].append(-1)

        if self.count > 5:
            p1 = np.array(self.traj_data['position'][-5])
            yaw = np.radians(self.traj_data['yaw'][-5])
            displacement = np.array([np.cos(yaw), np.sin(yaw)]) * 5
            p2 = p1 + displacement
            p3 = np.array(self.traj_data['position'][-1])
            d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
            
            if d > self.command_threshold:
                self.traj_data['command'][-5] = 2
            elif d < -self.command_threshold:
                self.traj_data['command'][-5] = 0
            else:
                self.traj_data['command'][-5] = 1
            
            print(self.traj_data['command'][-5])

        # save image to file
        image.save_to_disk(f'./data/{self.time}/{self.count}.jpg')
        if self.count == 150:
            self._reset()
        self.count += 1
            

    def _reset(self):
        with open(f'./data/{self.time}/traj_data.pkl', 'wb') as f:
            pickle.dump(self.traj_data, f)

        self.time = time.time()
        try:
            os.mkdir(f'./data/{self.time}')
        except FileExistsError:
            pass  # Directory already exists
        
        self.count = 0
        self.traj_data = {
            'position': [],
            'yaw': [],
            'command' : []
        }
        self.spawn_ego_vehicle()

    def _step(self):
        
        self.world.tick()

    def loop(self):
        while True:
            self._step()

def main():
    collect_data = CollectData()
    collect_data.loop()

if __name__ == '__main__':
    main()

        
        
        