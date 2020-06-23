import os

import cv2
import tensorflow as tf
import argparse
import numpy as np
from dddssd.lib.core.config import cfg, cfg_from_file
import dddssd.lib.dataset.maps_dict as maps_dict
from dddssd.lib.dataset.data_provider.data_provider import DataFromList, MultiProcessMapData, BatchDataNuscenes
from dddssd.lib.modeling import choose_model
from dddssd.lib.utils import box_3d_utils
from dddssd.lib.utils.anchors_util import project_to_image_space_corners
from dddssd.lib.utils.kitti_util import project_to_image
from dddssd.lib.utils.points_filter import get_point_filter_in_image, get_point_filter
from visualization.kitti_util import comp_box_3d


def parse_args():
    parser = argparse.ArgumentParser(description='Tester')
    parser.add_argument('--cfg', required=True, help='Config file for testing')
    parser.add_argument('--restore_model_path', required=True,
                        help='Restore model path e.g. log/model.ckpt [default: None]')
    parser.add_argument('--img_list', default='val', help='Train/Val/Trainval list')
    parser.add_argument('--split', default='training', help='Dataset split')

    # some evaluation threshold
    parser.add_argument('--cls_threshold', default=0.3, help='Filtering Predictions')
    parser.add_argument('--no_gt', action='store_true', help='Used for test set')
    args = parser.parse_args()

    return args


class Evaluator:
    def __init__(self, config, model_path, cls_threshold=0.3):
        self.batch_size = 1# config.TRAIN.CONFIG.BATCH_SIZE
        self.gpu_num = config.TRAIN.CONFIG.GPU_NUM
        self.num_workers = config.DATA_LOADER.NUM_THREADS
        self.log_dir = config.MODEL.PATH.EVALUATION_DIR
        self.cls_list = config.DATASET.KITTI.CLS_LIST
        self.extents = config.DATASET.POINT_CLOUD_RANGE
        self.extents = np.reshape(self.extents, [3, 2])
        self.is_training = False

        self.cls_thresh = float(cls_threshold)
        self.restore_model_path = model_path

        # save dir
        self.log_dir = os.path.join(self.log_dir, self.restore_model_path)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_file = open(os.path.join(self.log_dir, 'log_train.txt'), 'w')
        self._log_string('**** Saving Evaluation results to the path %s ****' % self.log_dir)

        # model list
        self.model_func = choose_model()
        self.model_list, self.pred_list, self.placeholders = self._build_model_list()
        placeholders_list = []
        for model in self.model_list:
            placeholders_list.append(model.placeholders)
        self.placeholders_list = placeholders_list

        self.dataset_iter = self.load_batch()

        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.saver.restore(self.session, self.restore_model_path)
        self._check()

    def evaluate(self, sample_id):
        # sess = tf.Session()
        # self.saver.restore(sess, self.restore_model_path)
        result = self._get_prediction(self.session, self.cls_thresh, sample_id)
        # sess.close()
        # print(result)
        return result

    def infer(self, sample):
        # self.dataset_iter = self.load_batch()
        feed_dict = self.create_feed_dict(sample)
        pred_bbox_3d_op, pred_cls_score_op, pred_cls_category_op = self.session.run(self.pred_list, feed_dict=feed_dict)

        calib_P, sample_name = self.info
        calib_P = calib_P[0]

        select_idx = np.where(pred_cls_score_op >= self.cls_thresh)[0]
        pred_cls_score_op = pred_cls_score_op[select_idx]
        pred_cls_category_op = pred_cls_category_op[select_idx]
        pred_bbox_3d_op = pred_bbox_3d_op[select_idx]
        pred_bbox_corners_op = box_3d_utils.get_box3d_corners_helper_np(pred_bbox_3d_op[:, :3], pred_bbox_3d_op[:, -1],
                                                                        pred_bbox_3d_op[:, 3:-1])
        pred_bbox_2d = project_to_image_space_corners(pred_bbox_corners_op, calib_P, img_shape=(384, 1248))

        result = {
            "score": [],
            "class": [],
            "bbox": [],
            "location": [],
            "dimensions": [],
            "vertices": []
        }

        for idx in range(len(pred_cls_score_op)):
            cls_idx = int(pred_cls_category_op[idx])
            result["class"].append(self.cls_list[cls_idx])
            result["bbox"].append(
                [float(pred_bbox_2d[idx, 0]), float(pred_bbox_2d[idx, 1]), float(pred_bbox_2d[idx, 2]),
                 float(pred_bbox_2d[idx, 3])])
            result["dimensions"].append(
                [float(pred_bbox_3d_op[idx, 4]), float(pred_bbox_3d_op[idx, 5]), float(pred_bbox_3d_op[idx, 3])])
            result["location"].append(
                [float(pred_bbox_3d_op[idx, 0]), float(pred_bbox_3d_op[idx, 1]), float(pred_bbox_3d_op[idx, 2]),
                 float(pred_bbox_3d_op[idx, -1])])
            result["score"].append(float(pred_cls_score_op[idx]))
            b2d, b3d = comp_box_3d(float(pred_bbox_3d_op[idx, 4]), float(pred_bbox_3d_op[idx, 5]), float(pred_bbox_3d_op[idx, 3]),
                        float(pred_bbox_3d_op[idx, 0]), float(pred_bbox_3d_op[idx, 1]), float(pred_bbox_3d_op[idx, 2]),
                 float(pred_bbox_3d_op[idx, -1]), calib_P)
            result["vertices"].append([b2d, b3d])
            # result["vertices"].append(project_to_image(pred_bbox_corners_op[idx], calib_P))
        return result


    def _get_prediction(self, sess, cls_thresh, sample_id):
        feed_dict = self.create_feed_dict(sample_id)
        pred_bbox_3d_op, pred_cls_score_op, pred_cls_category_op = sess.run(self.pred_list, feed_dict=feed_dict)

        calib_P, sample_name = self.info
        calib_P = calib_P[0]

        select_idx = np.where(pred_cls_score_op >= cls_thresh)[0]
        pred_cls_score_op = pred_cls_score_op[select_idx]
        pred_cls_category_op = pred_cls_category_op[select_idx]
        pred_bbox_3d_op = pred_bbox_3d_op[select_idx]
        pred_bbox_corners_op = box_3d_utils.get_box3d_corners_helper_np(pred_bbox_3d_op[:, :3], pred_bbox_3d_op[:, -1], pred_bbox_3d_op[:, 3:-1])
        pred_bbox_2d = project_to_image_space_corners(pred_bbox_corners_op, calib_P, img_shape=(384, 1248))

        result = {
            "score": [],
            "class": [],
            "bbox": [],
            "location": [],
            "dimensions": [],
            "vertices": []
        }

        for idx in range(len(pred_cls_score_op)):
            cls_idx = int(pred_cls_category_op[idx])
            # print('%s %0.2f %d %d ' % (self.cls_list[cls_idx], 0., 0, -10))
            # print('%0.2f %0.2f %0.2f %0.2f ' % (float(pred_bbox_2d[idx, 0]), float(pred_bbox_2d[idx, 1]), float(pred_bbox_2d[idx, 2]), float(pred_bbox_2d[idx, 3])))
            # print('%0.2f %0.2f %0.2f ' % (float(pred_bbox_3d_op[idx, 4]), float(pred_bbox_3d_op[idx, 5]), float(pred_bbox_3d_op[idx, 3])))
            # print('%0.2f %0.2f %0.2f %0.2f ' % (float(pred_bbox_3d_op[idx, 0]), float(pred_bbox_3d_op[idx, 1]), float(pred_bbox_3d_op[idx, 2]), float(pred_bbox_3d_op[idx, -1])))
            # print('%0.9f\n' % float(pred_cls_score_op[idx]))
            result["class"].append(self.cls_list[cls_idx])
            result["bbox"].append([float(pred_bbox_2d[idx, 0]), float(pred_bbox_2d[idx, 1]), float(pred_bbox_2d[idx, 2]), float(pred_bbox_2d[idx, 3])])
            result["dimensions"].append([float(pred_bbox_3d_op[idx, 4]), float(pred_bbox_3d_op[idx, 5]), float(pred_bbox_3d_op[idx, 3])])
            result["location"].append([float(pred_bbox_3d_op[idx, 0]), float(pred_bbox_3d_op[idx, 1]), float(pred_bbox_3d_op[idx, 2]), float(pred_bbox_3d_op[idx, -1])])
            result["score"].append(float(pred_cls_score_op[idx]))
            vert2d, vert3d = comp_box_3d(result["dimensions"][idx][0], result["dimensions"][idx][1], result["dimensions"][idx][2],
                        result["location"][idx][0], result["location"][idx][1], result["location"][idx][2],
                        result["location"][idx][3], calib_P)
            result["vertices"].append([vert2d, vert3d])
            # result["vertices"].append(project_to_image(pred_bbox_corners_op[idx], calib_P))
        return result

    def _check(self):
        checkpoint_dirs = self.restore_model_path

        if os.path.isdir(checkpoint_dirs):
            cur_model_path = tf.train.latest_checkpoint(checkpoint_dirs)
        else:
            cur_model_path = checkpoint_dirs
        if not cur_model_path:
            raise Exception('Please provide valid checkpoint path')
        self._log_string('**** Test New Result ****')
        self._log_string('Assign From checkpoint: %s' % cur_model_path)

    def _build_model_list(self):
        model_list = []
        model = self.model_func(self.batch_size, self.is_training)
        model.model_forward()
        model_list.append(model)

        # get prediction results, bs = 1
        pred_list = self._set_evaluation_tensor(model)

        # placeholders
        placeholders = model.placeholders

        return model_list, pred_list, placeholders

    def _set_evaluation_tensor(self, model):
        pred_bbox_3d = tf.squeeze(model.output[maps_dict.PRED_3D_BBOX][-1], axis=0)
        pred_cls_score = tf.squeeze(model.output[maps_dict.PRED_3D_SCORE][-1], axis=0)
        pred_cls_category = tf.squeeze(model.output[maps_dict.PRED_3D_CLS_CATEGORY][-1], axis=0)
        pred_list = [pred_bbox_3d, pred_cls_score, pred_cls_category]
        return pred_list

    def _log_string(self, out_str):
        self.log_file.write(out_str + '\n')
        self.log_file.flush()
        print(out_str)

    def create_feed_dict(self, s):
        sample = next(self.dataset_iter, None)
        # sample = self.load_sample(s)
        points, sem_labels, sem_dists, label_boxes_3d, ry_cls_label, residual_angle, \
        label_classes, calib_P, sample_name = sample
        self.info = [calib_P, sample_name]

        feed_dict = dict()
        for i in range(1):
            cur_placeholder = self.placeholders_list[i]
            begin_idx = i*self.batch_size
            end_idx = (i+1)*self.batch_size

            feed_dict[cur_placeholder[maps_dict.PL_POINTS_INPUT]] = points[begin_idx:end_idx]

            feed_dict[cur_placeholder[maps_dict.PL_LABEL_SEMSEGS]] = sem_labels[begin_idx:end_idx]
            feed_dict[cur_placeholder[maps_dict.PL_LABEL_DIST]] = sem_dists[begin_idx:end_idx]

            feed_dict[cur_placeholder[maps_dict.PL_LABEL_BOXES_3D]] = label_boxes_3d[begin_idx:end_idx]
            feed_dict[cur_placeholder[maps_dict.PL_LABEL_CLASSES]] = label_classes[begin_idx:end_idx]
            feed_dict[cur_placeholder[maps_dict.PL_ANGLE_CLS]] = ry_cls_label[begin_idx:end_idx]
            feed_dict[cur_placeholder[maps_dict.PL_ANGLE_RESIDUAL]] = residual_angle[begin_idx:end_idx]

            feed_dict[cur_placeholder[maps_dict.PL_CALIB_P2]] = calib_P[begin_idx:end_idx]

        return feed_dict

    def load_batch(self):
        """
        make data with batch_size per thread
        """
        perm = np.arange(1).tolist()  # a list indicates each data
        dp = DataFromList(perm, is_train=False, shuffle=False)
        dp = MultiProcessMapData(dp, self.load_sample, 1)

        use_concat = [0, 0, 0, 2, 2, 2, 2, 0, 0]
        dp = BatchDataNuscenes(dp, 1, use_concat=use_concat)
        dp.reset_state()
        dp = dp.get_data()
        return dp

    def load_sample(self, sample, pipename):
        sample_dict, biggest_label_num = self.preprocess_samples(sample)
        biggest_label_num = 0

        sem_labels = sample_dict[maps_dict.KEY_LABEL_SEMSEG]
        sem_dists = sample_dict[maps_dict.KEY_LABEL_DIST]
        points = sample_dict[maps_dict.KEY_POINT_CLOUD]
        calib = sample_dict[maps_dict.KEY_STEREO_CALIB]

        label_boxes_3d = np.zeros([1, 7], np.float32)
        label_classes = np.zeros([1], np.int32)
        cur_label_num = 1
        ry_cls_label = np.zeros([1], np.int32)
        residual_angle = np.zeros([1], np.float32)

        # randomly choose points
        pts_num = points.shape[0]
        pts_idx = np.arange(pts_num)
        if pts_num >= cfg.MODEL.POINTS_NUM_FOR_TRAINING:
            sampled_idx = np.random.choice(pts_idx, cfg.MODEL.POINTS_NUM_FOR_TRAINING, replace=False)
        else:
            # pts_num < model_util.points_num_for_training
            # first random choice pts_num, replace=False
            sampled_idx_1 = np.random.choice(pts_idx, pts_num, replace=False)
            sampled_idx_2 = np.random.choice(pts_idx, cfg.MODEL.POINTS_NUM_FOR_TRAINING - pts_num, replace=True)
            sampled_idx = np.concatenate([sampled_idx_1, sampled_idx_2], axis=0)

        sem_labels = sem_labels[sampled_idx]
        sem_dists = sem_dists[sampled_idx]
        points = points[sampled_idx, :]

        biggest_label_num = max(biggest_label_num, cur_label_num)

        return biggest_label_num, points, sem_labels, sem_dists, label_boxes_3d, ry_cls_label, residual_angle,\
               label_classes, calib.P, sample_dict[maps_dict.KEY_SAMPLE_NAME]

    # Preprocess data
    def preprocess_samples(self, sample_id):
        img_filename = os.path.join("_out/testing/image_2", '%06d.png' % int(sample_id))
        lidar_filename = os.path.join("_out/testing/velodyne", '%06d.bin' % int(sample_id))
        calib_filename = os.path.join("_out/testing/calib", '%06d.txt' % int(sample_id))

        import dddssd.lib.utils.kitti_util as utils
        biggest_label_num = 0
        img = utils.load_image(img_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_shape = img.shape

        calib = utils.Calibration(calib_filename)

        points = utils.load_velo_scan(lidar_filename)
        points_intensity = points[:, 3:]
        points = points[:, :3]
        # filter out this, first cast it to rect
        points = calib.project_velo_to_rect(points)

        img_points_filter = get_point_filter_in_image(points, calib, image_shape[0], image_shape[1])
        voxelnet_points_filter = get_point_filter(points, self.extents)
        img_points_filter = np.logical_and(img_points_filter, voxelnet_points_filter)
        img_points_filter = np.where(img_points_filter)[0]
        points = points[img_points_filter]
        points_intensity = points_intensity[img_points_filter]

        sem_labels = np.ones([points.shape[0]], dtype=np.int)
        sem_dists = np.ones([points.shape[0]], dtype=np.float32)

        points = np.concatenate([points, points_intensity], axis=-1)

        if np.sum(sem_labels) == 0:
            return None, biggest_label_num

        # img_list is test
        sample_dict = {
            maps_dict.KEY_LABEL_SEMSEG: sem_labels,
            maps_dict.KEY_LABEL_DIST: sem_dists,
            maps_dict.KEY_POINT_CLOUD: points,
            maps_dict.KEY_STEREO_CALIB: calib,
            maps_dict.KEY_SAMPLE_NAME: sample_id
        }
        return sample_dict, biggest_label_num

    def preprocess_sample(self, sample):
        import dddssd.lib.utils.kitti_util as utils
        biggest_label_num = 0
        img = sample["image_2"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_shape = img.shape

        calib = utils.Calibration("", calib=sample["calib"])

        # points = utils.load_velo_scan(lidar_filename)
        points = sample["velodyne"]
        points_intensity = points[:, 3:]
        points = points[:, :3]
        # filter out this, first cast it to rect
        points = calib.project_velo_to_rect(points)

        img_points_filter = get_point_filter_in_image(points, calib, image_shape[0], image_shape[1])
        voxelnet_points_filter = get_point_filter(points, self.extents)
        img_points_filter = np.logical_and(img_points_filter, voxelnet_points_filter)
        img_points_filter = np.where(img_points_filter)[0]
        points = points[img_points_filter]
        points_intensity = points_intensity[img_points_filter]

        sem_labels = np.ones([points.shape[0]], dtype=np.int)
        sem_dists = np.ones([points.shape[0]], dtype=np.float32)

        points = np.concatenate([points, points_intensity], axis=-1)

        if np.sum(sem_labels) == 0:
            return None, biggest_label_num

        # img_list is test
        sample_dict = {
            maps_dict.KEY_LABEL_SEMSEG: sem_labels,
            maps_dict.KEY_LABEL_DIST: sem_dists,
            maps_dict.KEY_POINT_CLOUD: points,
            maps_dict.KEY_STEREO_CALIB: calib,
            maps_dict.KEY_SAMPLE_NAME: sample["idx"]
        }
        return sample_dict, biggest_label_num

if __name__ == '__main__':
    args = parse_args()
    cfg_from_file(args.cfg)

    # set bs, gpu_num and workers_num to be 1
    cfg.TRAIN.CONFIG.BATCH_SIZE = 1  # only support bs=1 when testing
    cfg.TRAIN.CONFIG.GPU_NUM = 1
    cfg.DATA_LOADER.NUM_THREADS = 1
    if args.no_gt:
        cfg.TEST.WITH_GT = False

    cur_evaluator = Evaluator(config=cfg, model_path="dddssd/log/2020-05-23 11:49:17.807186/model-142032")
    cur_evaluator.evaluate("000002")
    print("**** Finish evaluation steps ****")
