import os
import argparse
import torch
from test_MT_util import test_all_case
# from networks.EGNet import build_model
from networks.MTMT import build_model
# from networks.EGNet_onlyDSS import build_model
# from networks.EGNet_task3 import build_model
from visualize import shadow_a_bins_plot


def test_calculate_metric(FLAGS):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    print(FLAGS.snapshot_path)
    num_classes = 1
    if not os.path.exists(FLAGS.test_save_path):
        os.makedirs(FLAGS.test_save_path)

    if FLAGS.dataset_name == 'ImageNet_val':
        FLAGS.test_save_path = './models/model_SBU_EGNet_ablation/prediction_ImageNet-val-categories/'+FLAGS.class_name

        img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(FLAGS.root_path, FLAGS.class_name)) if f.endswith('.JPEG')]
        data_path = [(os.path.join(FLAGS.root_path, FLAGS.class_name, img_name + '.JPEG'),
                '*****')
                for img_name in img_list]
    
    if FLAGS.dataset_name == 'ISTD-Test':
        img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(FLAGS.root_path, 'test_A')) if f.endswith('.png')]
        data_path = [(os.path.join(FLAGS.root_path, 'test_A', img_name + '.png'),
                      os.path.join(FLAGS.root_path, 'test_B', img_name + '.png'))
                      for img_name in img_list]

    if FLAGS.dataset_name == 'SBU-Test':
        img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(FLAGS.root_path, 'ShadowImages')) if f.endswith('.jpg')]
        data_path = [(os.path.join(FLAGS.root_path, 'ShadowImages', img_name + '.jpg'),
                      os.path.join(FLAGS.root_path, 'ShadowMasks', img_name + '.png'))
                      for img_name in img_list]

    if FLAGS.dataset_name == 'UCF-Test':
        img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(FLAGS.root_path, 'GroundTruth')) if f.endswith('.png')]
        data_path = [(os.path.join(FLAGS.root_path, 'GroundTruth', img_name + '.png'),
                      os.path.join(FLAGS.root_path, 'InputImages', img_name + '.jpg'))
                      for img_name in img_list]
    

    net = build_model('resnext101').cuda()
    net.load_state_dict(torch.load(FLAGS.snapshot_path))
    print("init weight from {}".format(FLAGS.snapshot_path))
    net.eval()

    avg_metric = test_all_case(net, data_path, num_classes=num_classes,
                               save_result=FLAGS.save_result, test_save_path=FLAGS.test_save_path, trans_scale=FLAGS.scale, 
                               GT_access=FLAGS.GT_access, shadow_count=FLAGS.shadow_count, class_name=FLAGS.class_name, 
                               dataset_name=FLAGS.dataset_name)

    return avg_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--root_path', type=str, default='./data/ISTD_test/', help='Test dataset location')
    parser.add_argument('--root_path', type=str, default='./data/UCF/', help='Test dataset location')
    # parser.add_argument('--root_path', type=str, default='./data/UCF/', help='Test dataset location')
    # parser.add_argument('--root_path', type=str, default='./object_recognition_models/data/val/', help='Test dataset location')
    parser.add_argument('--model', type=str,  default='EGNet', help='model_name')
    parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
    parser.add_argument('--base_lr', type=float,  default=0.005, help='base learning rate')
    parser.add_argument('--edge', type=float, default='10', help='edge learning weight')
    parser.add_argument('--epoch_name', type=str,  default='iter_7000.pth', help='choose one epoch/iter as pretained')
    parser.add_argument('--consistency', type=float,  default=1.0, help='consistency')
    parser.add_argument('--scale', type=int,  default=416, help='batch size of 8 with resolution of 416*416 is exactly OK')
    parser.add_argument('--subitizing', type=float,  default=5.0, help='subitizing loss weight')
    parser.add_argument('--repeat', type=int,  default=6, help='repeat')
    parser.add_argument('--dataset_name', type=str,  default='UCF-Test')
    parser.add_argument('--snapshot_path', type=str, default='./models/model_SBU_EGNet_ablation/iter_10000.pth')
    parser.add_argument('--GT_access', default=True)
    parser.add_argument('--save_result', default=False)
    parser.add_argument('--shadow_count', default=False)
    parser.add_argument('--class_name', type=str, default="")
    parser.add_argument('--test_save_path', type=str, default='./models/model_SBU_EGNet_ablation/prediction_ImageNet-val-categories/')

    FLAGS = parser.parse_args()

    if FLAGS.dataset_name == "ImageNet_val":
        folder = FLAGS.root_path
        subfolders = [ f.name for f in os.scandir(folder) if f.is_dir() ]
        for subf in subfolders:
            metric = test_calculate_metric_ImageNet(FLAGS)
            shadow_a_bins_plot(subf)
    
    if FLAGS.dataset_name == "ISTD-Test" or FLAGS.dataset_name == "SBU-Test" or FLAGS.dataset_name == "UCF-Test":
        metric = test_calculate_metric(FLAGS)
        # Calculate overall statistic for the dataset (BER) and write it into the file
        if FLAGS.GT_access:
            with open('./models/model_SBU_EGNet_ablation/record/test_record_EGNet_meanteacher.txt', 'a') as f:
                f.write(snapshot_path+' ')
                f.write(str(metric)+' '+FLAGS.dataset_name+'\r\n')
            print('Test ber results: {}'.format(metric))
