#!/usr/bin/env python3

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    R            : restart level

STARTING in a moment...
"""

import argparse
import pathlib

from skimage import io

from pointpillars.second.core import box_np_ops
from pointpillars.second.create_data import _create_reduced_point_cloud
from pointpillars.second.data.kitti_common import _extend_matrix, add_difficulty_to_annos, get_velodyne_path, \
    get_image_path, get_label_path, get_label_anno, get_calib_path
from pointpillars.second.pytorch.inference import TorchInferenceContext
from visualization.kitti_util import draw_projected_box3d, comp_box_3d

try:
    import pygame
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
    from numpy.linalg import pinv, inv
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

from carla import image_converter
from carla.client import make_carla_client
from carla.planner.map import CarlaMap
from carla.tcp import TCPConnectionError
from carla.transform import Transform

from utils.timer import Timer
from core.dataexport import *
from core.bounding_box import create_kitti_data_point
from utils.carla_utils import KeyboardHelper, MeasurementsDisplayHelper
from core.constants import *
from core.settings import make_carla_settings
from utils import lidar_utils, degrees_to_radians
import time
from math import cos, sin


CONFIG_FILE = "pointpillars/second/configs/carla/car/xyres_16.proto"
MODEL_PATH = "pointpillars/models/carla_model_1_AP_68.60/voxelnet-890880.tckpt"


""" OUTPUT FOLDER GENERATION """
PHASE = "testing"
OUTPUT_FOLDER = os.path.join("_out", PHASE)
folders = ['calib', 'image_2', 'label_2', 'velodyne', 'planes']


def maybe_create_dir(path):
    if not os.path.exists(directory):
        os.makedirs(directory)


for folder in folders:
    directory = os.path.join(OUTPUT_FOLDER, folder)
    maybe_create_dir(directory)

""" DATA SAVE PATHS """
GROUNDPLANE_PATH = os.path.join(OUTPUT_FOLDER, 'planes/{0:06}.txt')
LIDAR_PATH = os.path.join(OUTPUT_FOLDER, 'velodyne/{0:06}.bin')
LABEL_PATH = os.path.join(OUTPUT_FOLDER, 'label_2/{0:06}.txt')
IMAGE_PATH = os.path.join(OUTPUT_FOLDER, 'image_2/{0:06}.png')
CALIBRATION_PATH = os.path.join(OUTPUT_FOLDER, 'calib/{0:06}.txt')


def carla_anno_to_corners(info, annos=None):
    rect = info['calib/R0_rect']
    P2 = info['calib/P2']
    Tr_velo_to_cam = info['calib/Tr_velo_to_cam']
    if annos is None:
        annos = info['annos']
    dims = annos['dimensions']
    loc = annos['location']
    rots = annos['rotation_y']
    scores = None
    if 'score' in annos:
        scores = annos['score']
    boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
    boxes_lidar = box_np_ops.box_camera_to_lidar(boxes_camera, rect,
                                                 Tr_velo_to_cam)
    boxes_corners = box_np_ops.center_to_corner_box3d(
        boxes_lidar[:, :3],
        boxes_lidar[:, 3:6],
        boxes_lidar[:, 6],
        origin=[0.5, 0.5, 0],
        axis=2)
    return boxes_corners, scores, boxes_lidar


def get_carla_info(
        idx,
        path,
        training=True,
        label_info=True,
        velodyne=False,
        calib=False,
        extend_matrix=True,
        relative_path=True,
        with_imageshape=True):
    root_path = pathlib.Path(path)
    image_info = {'image_idx': idx, 'pointcloud_num_features': 4}
    annotations = None
    if velodyne:
        image_info['velodyne_path'] = get_velodyne_path(
            idx, path, training, relative_path)
    image_info['img_path'] = get_image_path(idx, path, training,
                                            relative_path)
    if with_imageshape:
        img_path = image_info['img_path']
        if relative_path:
            img_path = str(root_path / img_path)
        image_info['img_shape'] = np.array(
            io.imread(img_path).shape[:2], dtype=np.int32)
    if label_info:
        label_path = get_label_path(idx, path, training, relative_path)
        if relative_path:
            label_path = str(root_path / label_path)
        annotations = get_label_anno(label_path)
    if calib:
        calib_path = get_calib_path(
            idx, path, training, relative_path=False)
        with open(calib_path, 'r') as f:
            lines = f.readlines()
        P0 = np.array(
            [float(info) for info in lines[0].split(' ')[1:13]]).reshape(
            [3, 4])
        P1 = np.array(
            [float(info) for info in lines[1].split(' ')[1:13]]).reshape(
            [3, 4])
        P2 = np.array(
            [float(info) for info in lines[2].split(' ')[1:13]]).reshape(
            [3, 4])
        P3 = np.array(
            [float(info) for info in lines[3].split(' ')[1:13]]).reshape(
            [3, 4])
        if extend_matrix:
            P0 = _extend_matrix(P0)
            P1 = _extend_matrix(P1)
            P2 = _extend_matrix(P2)
            P3 = _extend_matrix(P3)
        image_info['calib/P0'] = P0
        image_info['calib/P1'] = P1
        image_info['calib/P2'] = P2
        image_info['calib/P3'] = P3
        R0_rect = np.array([
            float(info) for info in lines[4].split(' ')[1:10]
        ]).reshape([3, 3])
        if extend_matrix:
            rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
            rect_4x4[3, 3] = 1.
            rect_4x4[:3, :3] = R0_rect
        else:
            rect_4x4 = R0_rect
        image_info['calib/R0_rect'] = rect_4x4
        Tr_velo_to_cam = np.array([
            float(info) for info in lines[5].split(' ')[1:13]
        ]).reshape([3, 4])
        Tr_imu_to_velo = np.array([
            float(info) for info in lines[6].split(' ')[1:13]
        ]).reshape([3, 4])
        if extend_matrix:
            Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
            Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
        image_info['calib/Tr_velo_to_cam'] = Tr_velo_to_cam
        image_info['calib/Tr_imu_to_velo'] = Tr_imu_to_velo
    if annotations is not None:
        image_info['annos'] = annotations
        add_difficulty_to_annos(image_info)
    return image_info


def _create_reduced_point_cloud(data_path,
                                info,
                                save_path=None,
                                back=False):
    v_path = info['velodyne_path']
    v_path = pathlib.Path(data_path) / v_path
    points_v = np.fromfile(str(v_path), dtype=np.float32, count=-1).reshape([-1, 4])
    rect = info['calib/R0_rect']
    P2 = info['calib/P2']
    Trv2c = info['calib/Tr_velo_to_cam']
    # first remove z < 0 points
    # keep = points_v[:, -1] > 0
    # points_v = points_v[keep]
    # then remove outside.
    if back:
        points_v[:, 0] = -points_v[:, 0]
    points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2,
                                                info["img_shape"])
    return points_v


class CarlaGame(object):
    def __init__(self, carla_client, args):
        self.torchInference = TorchInferenceContext()
        self.torchInference.build(CONFIG_FILE)
        self.torchInference.restore(MODEL_PATH)

        self.client = carla_client
        self._carla_settings, self._intrinsic, self._camera_to_car_transform, self._lidar_to_car_transform = make_carla_settings(
            args)
        self._timer = None
        self._display = None
        self._main_image = None
        self._mini_view_image1 = None
        self._mini_view_image2 = None
        self._enable_autopilot = args.autopilot
        self._lidar_measurement = None
        self._map_view = None
        self._is_on_reverse = False
        self._city_name = args.map_name
        self._map = CarlaMap(self._city_name, 16.43,
                             50.0) if self._city_name is not None else None
        self._map_shape = self._map.map_image.shape if self._city_name is not None else None
        self._map_view = self._map.get_map(
            WINDOW_HEIGHT) if self._city_name is not None else None
        self._position = None
        self._agent_positions = None
        self.captured_frame_no = 0
        self._measurements = None
        self._extrinsic = None
        # To keep track of how far the car has driven since the last capture of data
        self._agent_location_on_last_capture = None
        self._frames_since_last_capture = 0
        # How many frames we have captured since reset
        self._captured_frames_since_restart = 0
        self._det_annos = None
        self._carla_info = None
        self._exists_vehicles = False

    def execute(self):
        """Launch the PyGame."""
        pygame.init()
        self._initialize_game()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                reset = self._on_loop()
                if not reset:
                    self._on_render()
        finally:
            pygame.quit()

    def _initialize_game(self):
        if self._city_name is not None:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH + int((WINDOW_HEIGHT /
                                     float(self._map.map_image.shape[0])) * self._map.map_image.shape[1]),
                 WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

        logging.debug('pygame started')
        self._on_new_episode()

    def _on_new_episode(self):
        self._carla_settings.randomize_seeds()
        self._carla_settings.randomize_weather()
        scene = self.client.load_settings(self._carla_settings)
        number_of_player_starts = len(scene.player_start_spots)
        player_start = np.random.randint(number_of_player_starts)
        logging.info('Starting new episode...')
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False

        # Reset all tracking variables
        self._agent_location_on_last_capture = None
        self._frames_since_last_capture = 0
        self._captured_frames_since_restart = 0

    def _on_loop(self):
        self._timer.tick()
        measurements, sensor_data = self.client.read_data()
        is_stuck = self._frames_since_last_capture >= NUM_EMPTY_FRAMES_BEFORE_RESET
        is_stuck = False
        is_enough_datapoints = (self._captured_frames_since_restart + 1) % NUM_RECORDINGS_BEFORE_RESET == 0

        if (is_stuck or is_enough_datapoints) and GEN_DATA:
            logging.warning("Is stucK: {}, is_enough_datapoints: {}".format(
                is_stuck, is_enough_datapoints))
            self._on_new_episode()
            # If we dont sleep, the client will continue to render
            return True

        # (Extrinsic) Rt Matrix
        # (Camera) local 3d to world 3d.
        # Get the transform from the player protobuf transformation.
        world_transform = Transform(
            measurements.player_measurements.transform
        )
        # Compute the final transformation matrix.
        self._extrinsic = world_transform * self._camera_to_car_transform
        self._measurements = measurements
        self._last_player_location = measurements.player_measurements.transform.location
        self._main_image = sensor_data.get('CameraRGB', None)
        self._lidar_measurement = sensor_data.get('Lidar32', None)
        self._depth_image = sensor_data.get('DepthCamera', None)
        # Print measurements every second.
        if self._timer.elapsed_seconds_since_lap() > 1.0:
            if self._city_name is not None:
                # Function to get car position on map.
                map_position = self._map.convert_to_pixel([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])
                # Function to get orientation of the road car is in.
                lane_orientation = self._map.get_lane_orientation([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])

                MeasurementsDisplayHelper.print_player_measurements_map(
                    measurements.player_measurements,
                    map_position,
                    lane_orientation, self._timer)
            else:
                MeasurementsDisplayHelper.print_player_measurements(
                    measurements.player_measurements, self._timer)
            # Plot position on the map as well.
            self._timer.lap()

        control = self._get_keyboard_control(pygame.key.get_pressed())
        # Set the player position
        if self._city_name is not None:
            self._position = self._map.convert_to_pixel([
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.transform.location.z])
            self._agent_positions = measurements.non_player_agents

        if control is None:
            self._on_new_episode()
        elif self._enable_autopilot:
            self.client.send_control(
                measurements.player_measurements.autopilot_control)
        else:
            self.client.send_control(control)

    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        control = KeyboardHelper.get_keyboard_control(
            keys, self._is_on_reverse, self._enable_autopilot)
        if control is not None:
            control, self._is_on_reverse, self._enable_autopilot = control
        return control

    def _on_render(self):
        datapoints = []

        if self._main_image is not None and self._depth_image is not None:
            # Convert main image
            image = image_converter.to_rgb_array(self._main_image)

            # Retrieve and draw datapoints
            image, datapoints = self._generate_datapoints(image)

            # Draw lidar
            # Camera coordinate system is left, up, forwards
            if VISUALIZE_LIDAR:
                # Calculation to shift bboxes relative to pitch and roll of player
                rotation = self._measurements.player_measurements.transform.rotation
                pitch, roll, yaw = rotation.pitch, rotation.roll, rotation.yaw
                # Since measurements are in degrees, convert to radians

                pitch = degrees_to_radians(pitch)
                roll = degrees_to_radians(roll)
                yaw = degrees_to_radians(yaw)
                print('pitch: ', pitch)
                print('roll: ', roll)
                print('yaw: ', yaw)

                # Rotation matrix for pitch
                rotP = np.array([[cos(pitch), 0, sin(pitch)],
                                 [0, 1, 0],
                                 [-sin(pitch), 0, cos(pitch)]])
                # Rotation matrix for roll
                rotR = np.array([[1, 0, 0],
                                 [0, cos(roll), -sin(roll)],
                                 [0, sin(roll), cos(roll)]])

                # combined rotation matrix, must be in order roll, pitch, yaw
                rotRP = np.matmul(rotR, rotP)
                # Take the points from the point cloud and transform to car space
                point_cloud = np.array(self._lidar_to_car_transform.transform_points(
                    self._lidar_measurement.data))
                point_cloud[:, 2] -= LIDAR_HEIGHT_POS
                point_cloud = np.matmul(rotRP, point_cloud.T).T
                # Transform to camera space by the inverse of camera_to_car transform
                point_cloud_cam = self._camera_to_car_transform.inverse().transform_points(point_cloud)
                point_cloud_cam[:, 1] += LIDAR_HEIGHT_POS
                image = lidar_utils.project_point_cloud(
                    image, point_cloud_cam, self._intrinsic, 1)

            if self._det_annos is not None:
                scores = self._det_annos["score"]
                b_boxes = self._det_annos["bbox"]
                dimens = self._det_annos["dimensions"]
                rotation_y = self._det_annos["rotation_y"]
                location = self._det_annos["location"]

                for i, box in enumerate(b_boxes):
                    if scores[i] * 100 > 35:
                        x, y, z = location[i]
                        h, l, w = dimens[i]
                        ry = rotation_y[i]

                        # draw 3D
                        box3d_pts_2d, box3d_pts_3d = comp_box_3d(h, w, l, x, y, z, ry, self._carla_info['calib/P2'])
                        image = draw_projected_box3d(image, box3d_pts_2d, color=(0, 0, 255), thickness=1)

                        # draw 2D
                        # x1, y1, x2, y2 = b_boxes[i]
                        # cv2.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)

            # Display image
            surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))

            self._display.blit(surface, (0, 0))
            if self._map_view is not None:
                self._display_agents(self._map_view)
            pygame.display.flip()

            # Determine whether to save files
            # distance_driven = self._distance_since_last_recording()
            # has_driven_long_enough = distance_driven is None or distance_driven > DISTANCE_SINCE_LAST_RECORDING
            if (self._timer.step + 1) % STEPS_BETWEEN_RECORDINGS == 0:
                if datapoints:
                    # Avoid doing this twice or unnecessarily often
                    if not VISUALIZE_LIDAR:
                        # Calculation to shift bboxes relative to pitch and roll of player
                        rotation = self._measurements.player_measurements.transform.rotation
                        pitch, roll, yaw = rotation.pitch, rotation.roll, rotation.yaw
                        # Since measurements are in degrees, convert to radians

                        pitch = degrees_to_radians(pitch)
                        roll = degrees_to_radians(roll)
                        yaw = degrees_to_radians(yaw)
                        print('pitch: ', pitch)
                        print('roll: ', roll)
                        print('yaw: ', yaw)

                        # Rotation matrix for pitch
                        rotP = np.array([[cos(pitch), 0, sin(pitch)],
                                         [0, 1, 0],
                                         [-sin(pitch), 0, cos(pitch)]])
                        # Rotation matrix for roll
                        rotR = np.array([[1, 0, 0],
                                         [0, cos(roll), -sin(roll)],
                                         [0, sin(roll), cos(roll)]])

                        # combined rotation matrix, must be in order roll, pitch, yaw
                        rotRP = np.matmul(rotR, rotP)
                        # Take the points from the point cloud and transform to car space
                        point_cloud = np.array(self._lidar_to_car_transform.transform_points(
                            self._lidar_measurement.data))
                        point_cloud[:, 2] -= LIDAR_HEIGHT_POS
                        point_cloud = np.matmul(rotRP, point_cloud.T).T
                    self._update_agent_location()
                    # Save screen, lidar and kitti training labels together with calibration and groundplane files
                    self._save_training_files(datapoints, point_cloud)
                    self._get_predictions(image)
                    self.captured_frame_no += 1
                    self._captured_frames_since_restart += 1
                    self._frames_since_last_capture = 0
                else:
                    logging.debug(
                        "Could save datapoint".format(
                            DISTANCE_SINCE_LAST_RECORDING))
            else:
                self._frames_since_last_capture += 1
                logging.debug(
                    "Could not save training data - no visible agents of selected classes in scene")

    def _get_predictions(self, image):
        self._carla_info = get_carla_info(idx=self.captured_frame_no,
                                          path="_out",
                                          training=False,
                                          label_info=False,
                                          velodyne=True,
                                          calib=True,
                                          relative_path=True)
        points = _create_reduced_point_cloud(data_path="_out", info=self._carla_info)

        logging.info("Attempting to predict")
        inputs = self.torchInference.get_inference_input_dict(self._carla_info, points)
        with self.torchInference.ctx():
            self._det_annos = self.torchInference.inference(inputs)[0]
            print(self._det_annos)

    def _update_agent_location(self):
        self._agent_location_on_last_capture = self._measurements.player_measurements.transform.location

    def _generate_datapoints(self, image):
        """ Returns a list of datapoints (labels and such) that are generated this frame together with the main image image """
        datapoints = []
        image = image.copy()

        # Remove this
        rotRP = np.identity(3)
        # Stores all datapoints for the current frames
        for agent in self._measurements.non_player_agents:
            if should_detect_class(agent) and GEN_DATA:
                image, kitti_datapoint = create_kitti_data_point(
                    agent, self._intrinsic, self._extrinsic.matrix, image, self._depth_image,
                    self._measurements.player_measurements, rotRP)
                if kitti_datapoint:
                    datapoints.append(kitti_datapoint)

        return image, datapoints

    def _save_training_files(self, datapoints, point_cloud):
        logging.info("Attempting to save at timer step {}, frame no: {}".format(
            self._timer.step, self.captured_frame_no))
        groundplane_fname = GROUNDPLANE_PATH.format(self.captured_frame_no)
        lidar_fname = LIDAR_PATH.format(self.captured_frame_no)
        kitti_fname = LABEL_PATH.format(self.captured_frame_no)
        img_fname = IMAGE_PATH.format(self.captured_frame_no)
        calib_filename = CALIBRATION_PATH.format(self.captured_frame_no)

        save_groundplanes(
            groundplane_fname, self._measurements.player_measurements, LIDAR_HEIGHT_POS)
        save_ref_files(OUTPUT_FOLDER, self.captured_frame_no)
        save_image_data(
            img_fname, image_converter.to_rgb_array(self._main_image))
        save_kitti_data(kitti_fname, datapoints)
        save_lidar_data(lidar_fname, point_cloud,
                        LIDAR_HEIGHT_POS, LIDAR_DATA_FORMAT)
        save_calibration_matrices(
            calib_filename, self._intrinsic, self._extrinsic)

    def _display_agents(self, image):
        image = image[:, :, :3]

        new_window_width = (float(WINDOW_HEIGHT) / float(self._map_shape[0])) * \
                           float(self._map_shape[1])
        surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        w_pos = int(
            self._position[0] * (float(WINDOW_HEIGHT) / float(self._map_shape[0])))
        h_pos = int(self._position[1] *
                    (new_window_width / float(self._map_shape[1])))
        pygame.draw.circle(surface, [255, 0, 0, 255], (w_pos, h_pos), 6, 0)
        for agent in self._agent_positions:
            if agent.HasField('vehicle'):
                agent_position = self._map.convert_to_pixel([
                    agent.vehicle.transform.location.x,
                    agent.vehicle.transform.location.y,
                    agent.vehicle.transform.location.z])
                w_pos = int(
                    agent_position[0] * (float(WINDOW_HEIGHT) / float(self._map_shape[0])))
                h_pos = int(
                    agent_position[1] * (new_window_width / float(self._map_shape[1])))
                pygame.draw.circle(
                    surface, [255, 0, 255, 255], (w_pos, h_pos), 4, 0)

        self._display.blit(surface, (WINDOW_WIDTH, 0))


def should_detect_class(agent):
    """ Returns true if the agent is of the classes that we want to detect.
        Note that Carla has class types in lowercase """
    return True in [agent.HasField(class_type.lower()) for class_type in CLASSES_TO_LABEL]


def parse_args():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='logging.info debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-m', '--map-name',
        metavar='M',
        default=None,
        help='plot the map of the current city (needs to match active map in '
             'server, options: Town01 or Town02)')
    args = argparser.parse_args()
    return args


def main():
    args = parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    logging.info(__doc__)

    while True:
        try:
            with make_carla_client(args.host, args.port, timeout=100) as client:
                game = CarlaGame(client, args)
                game.execute()
                break
        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info('\nCancelled by user. Bye!')
