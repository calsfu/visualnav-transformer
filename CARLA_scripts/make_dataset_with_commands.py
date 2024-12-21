import sys
import numpy as np
import argparse
import logging
import random
import rospy
import rosbag
import pygame
import random
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, UInt8MultiArray, Int32

try:
    sys.path.append('../PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg')
except IndexError:
    pass
import carla

ack_data = []
car_data = []
saved_lone = []
saved_left = []
saved_mid = []
saved_right = []
saved_depth = []
saved_seg = []

# OAK-D values
camera_hz = 4
camera_freq = 1. / float(camera_hz)
camera_pixels_x = 160 #320 * 2 #1280
camera_pixels_y = 120 #240 * 2 #800
camera_horiz_fov = 72

collided = False

class imu:
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
    def set(self, xyz):
        self.x = xyz.x
        self.y = xyz.y
        self.z = xyz.z
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    def get_z(self):
        return self.z

class CarlaControlNode:
    def __init__(self):
        # Get argvs
        argparser = argparse.ArgumentParser(
            description=__doc__)
        argparser.add_argument(
            "--run",
            "-r",
            default=0,
            type=int,
            required=True
        )
        argparser.add_argument(
            '--host',
            metavar='H',
            default='127.0.0.1',
            help='IP of the host server (default: 127.0.0.1)')
        argparser.add_argument(
            '-p', '--port',
            metavar='P',
            default=2000,
            type=int,
            help='TCP port to listen to (default: 2000)')
        args = argparser.parse_args()
        #client.load_world('Town01')
        
        # Start up pygame to watch vehicle drive
        pygame.init()
        screen = pygame.display.set_mode((camera_pixels_x * 2, camera_pixels_y * 2))
        pygame.display.set_caption('image')
        surface = pygame.image.load("open_pygame/test.png").convert()
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        # Connect to CARLA
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(2.0)
        # self.world = self.client.load_world('Town01')
        # self.world.set_weather(carla.WeatherParameters.ClearNoon)
        self.world = self.client.get_world()

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
            random.shuffle(spawn_points)
            ego_transform = spawn_points[0]
            self.vehicle = self.world.spawn_actor(ego_bp,ego_transform)
            print('\nEgo is spawned')
        else: 
            logging.warning('Could not found any spawn points')

        imu_holder = imu()
        cam_ld = None
        cam_ld = self.world.get_blueprint_library().find('sensor.other.imu')
        ld_location = carla.Location(0,0,0)
        ld_rotation = carla.Rotation(0,0,0)
        ld_transform = carla.Transform(ld_location,ld_rotation)
        ego_imu = self.world.spawn_actor(cam_ld,ld_transform,attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)

        def print_vals(image):
            imu_holder.set(image.accelerometer)
            control = self.vehicle.get_control()
            velocity = self.vehicle.get_velocity()
            acceleration = self.vehicle.get_acceleration()
            ack = self.vehicle.get_transform()
            # throttle, steering, brake, 
            # x vel, y vel, z vel,
            # x accel, y accel, z accel
            # ack location x, y, z, ack rotation yaw
            ack_thing = [ack.location.x, ack.location.y, ack.location.z, ack.rotation.yaw]
            data = [control.throttle, (control.steer,0)[abs(control.steer) < 0.00001], control.brake]#, 
                             #imu_holder.get_x(), imu_holder.get_y(), imu_holder.get_z(),
                            #  velocity.x, velocity.y, velocity.z,
                            #  acceleration.x, acceleration.y, acceleration.z]
            data = np.array(data)
            global car_data
            car_data.append(data)
            global ack_data
            ack_data.append(ack_thing)

        ego_imu.listen(lambda image: print_vals(image))

        cam_ld = None
        cam_ld = self.world.get_blueprint_library().find('sensor.other.collision')
        ld_location = carla.Location(0,0,0)
        ld_rotation = carla.Rotation(0,0,0)
        ld_transform = carla.Transform(ld_location,ld_rotation)
        ego_collision = self.world.spawn_actor(cam_ld,ld_transform,attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        
        def collision_detected(collision):
            print("Collision detected")
            global collided
            collided = True

        ego_collision.listen(lambda collision: collision_detected(collision))

        # Spawn camera
        cam_bp = None
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(camera_pixels_x))
        cam_bp.set_attribute("image_size_y",str(camera_pixels_y))
        cam_bp.set_attribute("fov",str(camera_horiz_fov))
        cam_bp.set_attribute("sensor_tick",str(camera_freq))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        ego_lone = self.world.spawn_actor(cam_bp,cam_transform,attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)

        cam_bp = None
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(camera_pixels_x))
        cam_bp.set_attribute("image_size_y",str(camera_pixels_y))
        cam_bp.set_attribute("fov",str(camera_horiz_fov))
        cam_bp.set_attribute("sensor_tick",str(camera_freq))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,360 - 72,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        ego_left = self.world.spawn_actor(cam_bp,cam_transform,attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)

        cam_bp = None
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(camera_pixels_x))
        cam_bp.set_attribute("image_size_y",str(camera_pixels_y))
        cam_bp.set_attribute("fov",str(camera_horiz_fov))
        cam_bp.set_attribute("sensor_tick",str(camera_freq))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        ego_mid = self.world.spawn_actor(cam_bp,cam_transform,attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)

        cam_bp = None
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(camera_pixels_x))
        cam_bp.set_attribute("image_size_y",str(camera_pixels_y))
        cam_bp.set_attribute("fov",str(camera_horiz_fov))
        cam_bp.set_attribute("sensor_tick",str(camera_freq))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,72,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        ego_right = self.world.spawn_actor(cam_bp,cam_transform,attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)

        cam_bp = None
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        cam_bp.set_attribute("image_size_x",str(camera_pixels_x))
        cam_bp.set_attribute("image_size_y",str(camera_pixels_y))
        cam_bp.set_attribute("fov",str(camera_horiz_fov))
        cam_bp.set_attribute("sensor_tick",str(camera_freq))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        ego_depth = self.world.spawn_actor(cam_bp,cam_transform,attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)

        cam_bp = None
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        cam_bp.set_attribute("image_size_x",str(camera_pixels_x))
        cam_bp.set_attribute("image_size_y",str(camera_pixels_y))
        cam_bp.set_attribute("fov",str(camera_horiz_fov))
        cam_bp.set_attribute("sensor_tick",str(camera_freq))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,0,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        ego_seg = self.world.spawn_actor(cam_bp,cam_transform,attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        
        def save_lone(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            global saved_lone
            saved_lone.append(array)

        def save_left(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            global saved_left
            saved_left.append(array)

        def save_mid(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            global saved_mid
            saved_mid.append(array)

        def save_right(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            global saved_right
            saved_right.append(array)

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
            global saved_depth
            saved_depth.append(depth)

        def save_seg(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            array = array * (255 / 28) # max object id
            global saved_seg
            saved_seg.append(array)

        ego_lone.listen(lambda image: save_lone(image))
        ego_left.listen(lambda image: save_left(image))
        ego_mid.listen(lambda image: save_mid(image))
        ego_right.listen(lambda image: save_right(image))
        ego_depth.listen(lambda image: save_depth(image))
        ego_seg.listen(lambda image: save_seg(image))

        global ack_data
        global car_data
        global saved_lone
        global saved_left
        global saved_mid
        global saved_right
        global saved_depth
        global saved_seg

        buffer = 5
        command_msg = Int32()
        command_msg_array = [command_msg] * buffer
        data_msg = Float32MultiArray()
        data_msg_array = [data_msg] * buffer
        ack_msg = Float32MultiArray()
        ack_msg_array = [ack_msg] * buffer
        control_msg = UInt8MultiArray()
        control_msg_array = [control_msg] * buffer
        wide_msg = UInt8MultiArray()
        wide_msg_array = [wide_msg] * buffer
        composed_msg = UInt8MultiArray()
        composed_msg_array = [composed_msg] * buffer
        mt = np.array([0, 0])
        traj_position_array = [mt] * buffer
        traj_yaw_array = [0] * buffer
        user_view_array = np.empty([camera_pixels_y * 2, camera_pixels_x * 2, 3, buffer])
        
        transform = self.vehicle.get_transform()
        transform.location.z += 2
        spectator = self.world.get_spectator()
        spectator.set_transform(transform)
        self.vehicle.set_autopilot(True)
    
        frame = 0
        seconds_between_perturbation = 15

        command_threshold = 0.15
        command = 0

        global collided
        collisions = 0

        run_number = args.run
        autopilot = True
        frame_of_last_purtebatotion = 0

        minutes_of_data = 20
        bag = rosbag.Bag('/data/linden/4_hr_dataset/' + str(run_number) + '_0.bag', 'w')
        while not rospy.is_shutdown():
            world_snapshot = self.world.wait_for_tick() # make sure the camera has called back
            if frame == minutes_of_data * 60 * camera_hz:
                bag.close()
                print("Done capturing data")
                break

            if frame % (60 * camera_hz) == 0:
                print("Minute " + str(frame / (60 * camera_hz)))

            transform = self.vehicle.get_transform()
            transform.location.z += 2
            spectator.set_transform(transform)

            if collided:
                collided = False
                collisions = collisions + 1
                bag.close()
                bag = rosbag.Bag('/data/linden/4_hr_dataset/' + str(run_number) + '_' + str(collisions) + '.bag', 'w')

                spawn_points = self.world.get_map().get_spawn_points()
                number_of_spawn_points = len(spawn_points)
                if 0 < number_of_spawn_points:
                    random.shuffle(spawn_points)
                    ego_transform = spawn_points[0]
                    self.vehicle.set_transform(ego_transform)
                    print('\nCar teleported')
                else: 
                    logging.warning('Could not found any spawn points')

            while not (len(car_data) > 0 and len(saved_lone) > 0 and len(saved_left) > 0 and len(saved_mid) > 0 and \
                len(saved_right) > 0 and len(saved_depth) > 0 and len(saved_seg) > 0):
                pass
            
            position = self.vehicle.get_location()
            yaw = self.vehicle.get_transform().rotation.yaw
            traj_position_array[frame % buffer] = np.array([position.x, position.y])
            traj_yaw_array[frame % buffer] = yaw
            p1 = np.array(traj_position_array[(frame + 1) % buffer]) # from buffer frames ago
            yaw = np.radians(traj_yaw_array[(frame + 1) % buffer])
            displacement = np.array([np.cos(yaw), np.sin(yaw)]) * 5
            p2 = p1 + displacement
            p3 = np.array(traj_position_array[frame % buffer])
            d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
            
            if d > command_threshold:
                command = 2
            elif d < -command_threshold:
                command = 0
            else:
                command = 1
            
            print(command)
            if frame_of_last_purtebatotion + (seconds_between_perturbation * camera_hz) < frame:
                rndm = random.random()
                if rndm > .9:
                    frame_of_last_purtebatotion = frame + 2
                    self.vehicle.set_autopilot(False)
                    steer = self.vehicle.get_control().steer
                    throttle = self.vehicle.get_control().throttle
                    rndm = random.random()
                    if(rndm < .5):
                        rndm = -rndm - .5
                    steer = steer + rndm
                    if steer > 1.:
                        steer = 1
                    if steer < -1.:
                        steer = -1
                    self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))
            if frame == frame_of_last_purtebatotion:
                self.vehicle.set_autopilot(True)
                #throttle = self.vehicle.get_control().throttle
                #self.vehicle.apply_control(carla.VehicleControl(throttle=throttle))
                pass

            arr_data = car_data[0]
            ack_vals = ack_data[0]
            img_lone = saved_lone[0]
            img_left = saved_left[0]
            img_mid = saved_mid[0]
            img_right = saved_right[0]
            img_wide = np.concatenate((img_left, img_mid, img_right), axis=1)[:,::3,:]
            img_depth = saved_depth[0]
            img_seg = saved_seg[0]
            ack_data = []
            car_data = []
            saved_lone = []
            saved_left = []
            saved_mid = []
            saved_right = []
            saved_depth = []
            saved_seg = []

            img_comp = np.empty([img_lone.shape[0], img_lone.shape[1], img_lone.shape[2]])
            img_comp[:,:,0] = np.mean(img_lone, axis=2)
            img_comp[:,:,1] = np.mean(img_depth, axis=2)
            img_comp[:,:,2] = img_seg[:,:,0]

            temp1 = np.concatenate((img_lone, img_wide), axis=0)
            temp2 = np.concatenate((img_comp, np.empty([img_lone.shape[0], img_lone.shape[1], img_lone.shape[2]])), axis=0)
            temp3 = np.concatenate((temp1, temp2), axis=1)
            user_view_array[:,:,:,frame % buffer] = temp3

            command_msg.data = command
            data_msg = Float32MultiArray()
            ack_msg = Float32MultiArray()
            control_msg = UInt8MultiArray()
            wide_msg = UInt8MultiArray()
            composed_msg = UInt8MultiArray()
            data_msg.data = arr_data
            ack_msg.data = ack_vals
            control_msg.data = img_lone.flatten().astype(np.uint8).tolist()
            wide_msg.data = img_wide.flatten().astype(np.uint8).tolist()
            composed_msg.data = img_comp.flatten().astype(np.uint8).tolist()

            data_msg_array[frame % buffer] = data_msg
            ack_msg_array[frame % buffer] = ack_msg
            control_msg_array[frame % buffer] = control_msg
            wide_msg_array[frame % buffer] = wide_msg
            composed_msg_array[frame % buffer] = composed_msg

            if frame >= buffer:
                bag.write('/command', command_msg)
                bag.write('/odometry', ack_msg_array[(frame + 1) % buffer])
                bag.write('/throt_steer_brk', data_msg_array[(frame + 1) % buffer])
                bag.write('/camera/control', control_msg_array[(frame + 1) % buffer])
                bag.write('/camera/wide', wide_msg_array[(frame + 1) % buffer])
                bag.write('/camera/composed', composed_msg_array[(frame + 1) % buffer])

                surface = pygame.surfarray.make_surface(user_view_array[:,:,:,(frame + 1) % buffer].swapaxes(0, 1))
                screen.blit(surface, (0, 0))
                pygame.display.flip()
            frame = frame + 1

if __name__ == '__main__':
    try:
        node = CarlaControlNode()
    except rospy.ROSInterruptException:
        pass
