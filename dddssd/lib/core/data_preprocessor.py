import argparse

from dddssd.lib.core.config import cfg, cfg_from_file
from dddssd.lib.dataset.dataloader import choose_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--cfg', required=True, help='Config file for training')
    parser.add_argument('--split', default='training', help='Dataset split: training')
    parser.add_argument('--img_list', default='train', help='Train/Val/Trainval/Test list')
    args = parser.parse_args()

    return args
 

if __name__ == '__main__':
    args = parse_args()
    cfg_from_file(args.cfg)

    if args.img_list == 'test':
        # if test, no ground truth available
        cfg.TEST.WITH_GT = False 
    if args.img_list == 'test':
        # if val, no mix up dataset
        cfg.TRAIN.AUGMENTATIONS.MIXUP.OPEN = False

    dataset_func = choose_dataset()
    dataset = dataset_func('preprocessing', split=args.split, img_list=args.img_list, is_training=False)

    dataset.preprocess_batch()
