import sys
import numpy as np
import argparse
import logging
import random
import rospy
import pygame
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray

try:
    sys.path.append('../PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg')
except IndexError:
    pass
import carla

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

class CarlaControlNode:
    def __init__(self):
        # Get argvs
        argparser = argparse.ArgumentParser(
            description=__doc__)
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
        argparser.add_argument(
            '-d', '--deploy',
            metavar='D',
            default=1,
            type=int,
            help='Take driving input from ViNT')
        args = argparser.parse_args()

        # Initialize ROS node
        rospy.init_node('carla_control_node', anonymous=True)
        self.host = rospy.get_param('~host', '127.0.0.1')
        self.port = rospy.get_param('~port', 2000)
        self.role_name = rospy.get_param('~role_name', 'self.vehicle')
        pub_driving = rospy.Publisher('driving_reporting', String, queue_size=10)
        pub_camera = rospy.Publisher('camera', Image, queue_size=10)
        
        # Start up pygame to watch vehicle drive
        pygame.init()
        screen = pygame.display.set_mode((camera_pixels_x, camera_pixels_y))
        pygame.display.set_caption('image')
        surface = pygame.image.load("open_pygame/test.png").convert()
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        # Connect to CARLA
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        #self.world = self.client.load_world('Town01')
        #self.world.set_weather(carla.WeatherParameters.ClearNoon)

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
            #random.shuffle(spawn_points)
            ego_transform = spawn_points[0]
            self.vehicle = self.world.spawn_actor(ego_bp,ego_transform)
            print('\nEgo is spawned')
        else: 
            logging.warning('Could not found any spawn points')
        
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
        
        global saved_lone
        global saved_left
        global saved_mid
        global saved_right
        global saved_depth
        global saved_seg

        # Watch the wheels, I guess
        transform = self.vehicle.get_transform()
        transform.location.z += 2
        spectator = self.world.get_spectator()
        spectator.set_transform(transform)

        # For driving to create a rosbag, or using ViNT
        if args.deploy:
            rospy.Subscriber('/cmd_vel_mux/input/navi', Twist, self.waypoint)
        else:
            self.vehicle.set_autopilot(True)

        # Main loop
        rate = rospy.Rate(10) # 10hz
        bridge = CvBridge()
        world_snapshot = self.world.wait_for_tick() # make sure the camera has called back
        while not rospy.is_shutdown():
            transform = self.vehicle.get_transform()
            transform.location.z += 2
            spectator.set_transform(transform)
            control = self.vehicle.get_control()

            while not (len(saved_lone) > 0 and len(saved_left) > 0 and len(saved_mid) > 0 and \
                len(saved_right) > 0 and len(saved_depth) > 0 and len(saved_seg) > 0):
                pass

            img_lone = saved_lone[0]
            img_left = saved_left[0]
            img_mid = saved_mid[0]
            img_right = saved_right[0]
            img_wide = np.concatenate((img_left, img_mid, img_right), axis=1)[:,::3,:]
            img_depth = saved_depth[0]
            img_seg = saved_seg[0]
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

            pub_driving.publish(str(control.throttle) + "," + str((control.steer,0)[abs(control.steer) < 0.00001]) + ",: " + str(control.brake))
            if True: #len(img_wide) != 0:
                r,g,b = cv2.split(img_lone)
                img_lone = cv2.merge([b,g,r])
                pub_camera.publish(bridge.cv2_to_imgmsg(img_lone, "bgr8"))
                img = cv2.cvtColor(img_lone, cv2.COLOR_BGR2RGB)
                surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                screen.blit(surface, (0, 0))
                pygame.display.flip()
            rate.sleep()

    def waypoint(self, msg):
        print(msg)
        # control = carla.VehicleControl()
        # control.throttle = max(0.0, min(1.0, float(msg.data[0])))  # Forward/backward speed
        # control.steer = max(-1.0, min(1.0, float(msg.data[1])))   # Steering
        control = carla.VehicleAckermannControl()
        control.speed = msg.linear.x
        control.steer = msg.angular.z
        
        self.vehicle.apply_ackermann_control(control)

if __name__ == '__main__':
    try:
        node = CarlaControlNode()
    except rospy.ROSInterruptException:
        pass
