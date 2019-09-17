import os
import sys
import os.path
from options.test_options import TestOptions
from data import _create_dataset, create_dataloader
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html
import numpy as np
import glob
import torch
import cv2
import argparse
import options.options as option


# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
#
#
# def compute_var(imgs):
# #     trans = np.transpose(imgs, [0, 3, 2, 1])
# #     gray = np.empty((imgs.shape[0], 256, 256))
# #     for i in range(imgs.shape[0]):
# #         gray[i] = rgb2gray(trans[i])
# #     var_result = np.var(gray, axis=0)
# #     return np.sum(var_result) * 255. / (np.prod(var_result.shape))

def compute_var(imgs):
    trans = np.transpose(imgs, [0, 3, 2, 1])
    var_result = np.var(trans, axis=0)
    return np.sum(var_result) * 255. / (np.prod(var_result.shape))

# def rgb2lab(in_img,mean_cent=False):
#     from skimage import color
#     img_lab = color.rgb2lab(in_img)
#     if(mean_cent):
#         img_lab[:,:,0] = img_lab[:,:,0]-50
#     return img_lab
#
#
# def compute_var(imgs):
#     # print(imgs.shape)
#     trans = np.transpose(imgs, [0, 3, 2, 1])
#     trans = (trans + 1.) / 2
#     # gray = np.empty((imgs.shape[0], 256, 256))
#     # for i in range(imgs.shape[0]):
#     #     gray[i] = rgb2gray(trans[i])
#     # var_result = np.var(gray, axis=0)
#     lab_imgs = np.empty((imgs.shape[0], 256, 256))
#     for i in range(imgs.shape[0]):
#         lab_imgs[i] = rgb2lab(trans[i])[:, :, 0]
#     print(lab_imgs.max(), lab_imgs.min())
#     var_result = np.var(lab_imgs, axis=0)
#     # return np.sum(var_result) * 255. / (np.prod(var_result.shape))
#     return np.sum(var_result) / (np.prod(var_result.shape))


def downsample(x):

    x = x.astype(np.uint32)
    width = x.shape[0]
    height = x.shape[1]
    if width % 2 == 1:
        x = x[:-1, :, :]
    if height % 2 == 1:
        x = x[:, :-1, :]
    width = x.shape[0]
    height = x.shape[1]
    # print width, height
    skip_zero_index_w = np.arange(0, width, 2)
    skip_one_index_w = np.arange(1, width, 2)
    skip_zero_index_h = np.arange(0, height, 2)
    skip_one_index_h = np.arange(1, height, 2)
    lt = x[skip_zero_index_w, :, :][:, skip_zero_index_h, :]
    lb = x[skip_one_index_w, :, :][:, skip_zero_index_h, :]
    rt = x[skip_zero_index_w, :, :][:, skip_one_index_h, :]
    rb = x[skip_one_index_w, :, :][:, skip_one_index_h, :]
    downsample_1 = (lt + lb + rt + rb) / 4.0

    width = downsample_1.shape[0]
    height = downsample_1.shape[1]
    if width % 2 == 1:
        downsample_1 = downsample_1[:-1, :, :]
    if height % 2 == 1:
        downsample_1 = downsample_1[:, :-1, :]
    width = downsample_1.shape[0]
    height = downsample_1.shape[1]
    skip_zero_index_w = np.arange(0, width, 2)
    skip_one_index_w = np.arange(1, width, 2)
    skip_zero_index_h = np.arange(0, height, 2)
    skip_one_index_h = np.arange(1, height, 2)
    lt = downsample_1[skip_zero_index_w, :, :][:, skip_zero_index_h, :]
    lb = downsample_1[skip_one_index_w, :, :][:, skip_zero_index_h, :]
    rt = downsample_1[skip_zero_index_w, :, :][:, skip_one_index_h, :]
    rb = downsample_1[skip_one_index_w, :, :][:, skip_one_index_h, :]
    x = (lt + lb + rt + rb) / 4.0

    return x


def downsample_scale(img, scale):
    # print(img.shape[0], img.shape[1])
    return cv2.resize(img, dsize=(int(img.shape[0] / scale), int(img.shape[1] / scale)), interpolation=cv2.INTER_CUBIC)


def downsample_t(x):

    x = x.type(torch.IntTensor)
    width = x.shape[0]
    height = x.shape[1]
    if width % 2 == 1:
        x = x[:-1, :, :]
    if height % 2 == 1:
        x = x[:, :-1, :]
    width = x.shape[0]
    height = x.shape[1]
    # print width, height
    skip_zero_index_w = np.arange(0, width, 2)
    skip_one_index_w = np.arange(1, width, 2)
    skip_zero_index_h = np.arange(0, height, 2)
    skip_one_index_h = np.arange(1, height, 2)
    lt = x[skip_zero_index_w, :, :][:, skip_zero_index_h, :]
    lb = x[skip_one_index_w, :, :][:, skip_zero_index_h, :]
    rt = x[skip_zero_index_w, :, :][:, skip_one_index_h, :]
    rb = x[skip_one_index_w, :, :][:, skip_one_index_h, :]
    downsample_1 = (lt + lb + rt + rb) / 4.0

    width = downsample_1.shape[0]
    height = downsample_1.shape[1]
    if width % 2 == 1:
        downsample_1 = downsample_1[:-1, :, :]
    if height % 2 == 1:
        downsample_1 = downsample_1[:, :-1, :]
    width = downsample_1.shape[0]
    height = downsample_1.shape[1]
    skip_zero_index_w = np.arange(0, width, 2)
    skip_one_index_w = np.arange(1, width, 2)
    skip_zero_index_h = np.arange(0, height, 2)
    skip_one_index_h = np.arange(1, height, 2)
    lt = downsample_1[skip_zero_index_w, :, :][:, skip_zero_index_h, :]
    lb = downsample_1[skip_one_index_w, :, :][:, skip_zero_index_h, :]
    rt = downsample_1[skip_zero_index_w, :, :][:, skip_one_index_h, :]
    rb = downsample_1[skip_one_index_w, :, :][:, skip_one_index_h, :]
    x = (lt + lb + rt + rb) / 4.0

    return x


def add_dict(map, key, val):
    if key in map:
        map[key].append(val)
    else:
        map[key] = [val]


def get_caffe_results(root_path, ori_path):
    img_list = sorted(glob.glob(root_path))
    data_length = len(img_list)
    result = np.empty((data_length))
    print("------", data_length)
    mapping = {}

    for i, v in enumerate(img_list):
        filename = v.split('/')[-1]
        base_name_index = filename.index("_", 4)
        base_name = filename[(base_name_index + 1):]
        # img = cv2.imread(v, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(v)
        down_img = downsample(img)
        # down_img = downsample_scale(img, scale)
        base_img = cv2.imread(ori_path + base_name, cv2.IMREAD_UNCHANGED)
        result[i] = np.sum(np.linalg.norm(down_img - base_img))
        add_dict(mapping, base_name, img)
    return result, mapping


def get_srim_results(root_path, ori_path, scale=8):
    img_list = sorted(glob.glob(root_path))
    data_length = len(img_list)
    result = np.empty((data_length))
    print("------", data_length)
    mapping = {}

    for i, v in enumerate(img_list):
        filename = v.split('/')[-1]
        # print(filename)
        base_name_index = filename.rindex("_")
        base_name = filename[:base_name_index]
        # print(base_name, base_name_index)
        # img = cv2.imread(v, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(v)
        # down_img = downsample(img)
        down_img = downsample_scale(img, scale)
        # print(ori_path + base_name + ".JPEG")
        base_img = cv2.imread(ori_path + base_name + ".JPEG")
        # print(np.mean(base_img), np.mean(down_img))
        if len(base_img.shape) < 3:
            base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        down_base = downsample_scale(base_img, scale)
        result[i] = np.sum(np.linalg.norm(down_img - down_base))
        # print(result[i], down_img.shape, down_base.shape)
        # print(a)
        add_dict(mapping, base_name, img)
    return result, mapping


def compute_weight(diff, sigma):
    return np.exp(-diff**2 / (2*(sigma**2)))


def compute_mean(mapping):
    result = {}
    for key, val in mapping.items():
        all_img = np.stack(val, axis=0)
        result[key] = np.mean(all_img, axis=0)
    return result


def get_caffe_weighted(root_path, dists, sigma, mean_mapping):
    img_list = sorted(glob.glob(root_path))
    total_var_result = 0
    count = 0
    for i, v in enumerate(img_list):
        filename = v.split('/')[-1]
        base_name_index = filename.index("_", 4)
        base_name = filename[(base_name_index + 1):]
        # img = cv2.imread(v, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(v)
        cur_weight = compute_weight(dists[i], sigma)
        if i % 50 == 49:
            print("SRIM Avg weight for %s is %f" % (base_name, count / 50))
            count = 0
        else:
            count += cur_weight
        # print("SRIM Cur Dist: %f, with weight: %f" % (dists[i], cur_weight))
        total_var_result += cur_weight * np.sum(np.linalg.norm(img - mean_mapping[base_name]))

    return total_var_result / len(img_list)
# def get_weighted_var()


def get_srim_weighted(root_path, dists, sigma, mean_mapping):
    img_list = sorted(glob.glob(root_path))
    total_var_result = 0
    count = 0
    for i, v in enumerate(img_list):
        filename = v.split('/')[-1]
        base_name_index = filename.rindex("_")
        base_name = filename[:base_name_index]
        # img = cv2.imread(v, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(v)
        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cur_weight = compute_weight(dists[i], sigma)
        if i % 50 == 49:
            print("SRIM Avg weight for %s is %f" % (base_name, count / 50))
            count = 0
        else:
            count += cur_weight
        # print("SRIM Cur Dist: %f, with weight: %f" % (dists[i], cur_weight))
        total_var_result += cur_weight * np.sum(np.linalg.norm(img - mean_mapping[base_name]))

    return total_var_result / len(img_list)


def get_bgan_weighted(databank, dist_bank, sigma, mean_mapping):
    total_var_result = 0
    for dname, imgs in databank.items():
        count = 0
        for i, img in enumerate(imgs):
            cur_weight = compute_weight(dist_bank[dname][i], sigma)
            count += cur_weight
            # print("BGAN Cur Dist: %f, with weight: %f" % (dist_bank[dname][i], cur_weight))
            total_var_result += cur_weight * np.sum(np.linalg.norm(img - mean_mapping[dname]))
        print("BGAN Avg weight for %s is %f" % (dname, count / 50))

    return total_var_result


# options
opt = TestOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

# # create dataset
# dataset = create_dataset(opt)

parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, help='Path to option JSON file.')
# print(parser.parse_known_args())
data_opt = option.parse((parser.parse_known_args())[0].opt, is_train=True)
data_opt = option.dict_to_nonedict(data_opt)  # Convert to NoneDict, which return None for missing key.
# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(data_opt['datasets'].items()):
    if phase == 'val':
        test_set = _create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        print('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

# sample random z
if opt.sync:
    z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)


# ========== New version ===========
# total_var = 0
#
# root_path = "/home/nio/naboo/SRIM/results/RRDB_Code_69_x8_Bird/n01531178_test/*"
# ori_path = "/home/nio/naboo/data/n01531178_test/"
#
# # caffe_dists, caffe_mapping = get_caffe_results(root_path, ori_path)
# #
# # caffe_mean = compute_mean(caffe_mapping)
#
#
# caffe_dists, caffe_mapping = get_srim_results(root_path, ori_path, scale=8)
#
# caffe_mean = compute_mean(caffe_mapping)
#
#
# print("SRIM dist max: %f, min: %f, std: %f" % (caffe_dists.max(), caffe_dists.min(), np.std(caffe_dists)))
#
# gen_samples = np.empty((opt.num_test * opt.n_samples, 3, 256, 256))
#
# databank = {}
# bgan_dist_bank = {}
# mean_bgan = {}
#
# bgan_dists = np.empty((opt.num_test * opt.n_samples))
# # first stage
# for i, data in enumerate(islice(test_loader, opt.num_test)):
#     model.set_input(data)
#     print('process input image %3.3d/%3.3d' % (i, opt.num_test))
#     if not opt.sync:
#         z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
#
#     cur_all_data = np.empty((opt.n_samples, 3, 256, 256))
#     for nn in range(opt.n_samples):
#         real_A, fake_B, real_B = model.test(z_samples[[nn]], encode=False)
#         # print(real_A.size(), fake_B.size())
#         # print(np.transpose(torch.reshape(real_A, real_A.size()[1:]), [2, 0, 1]))
#         real_A = torch.reshape(real_A, real_A.size()[1:]).cpu().numpy()
#         fake_B = torch.reshape(fake_B, fake_B.size()[1:]).cpu().numpy()
#         real_A = (real_A + 1.) * 255. / 2
#         fake_B = (fake_B + 1.) * 255. / 2
#         add_dict(databank, data["A_paths"][0], fake_B)
#         cur_all_data[nn] = fake_B
#         # print("-=-=-", real_A.shape, np.transpose(real_A, [1, 2, 0]).shape)
#
#         # down_a = downsample(np.transpose(real_A, [1, 2, 0]))
#         # down_b = downsample(np.transpose(fake_B, [1, 2, 0]))
#         # down_a = downsample_scale(np.transpose(real_A, [1, 2, 0]), scale=8)
#         down_a = np.transpose(real_A, [1, 2, 0])
#         down_b = downsample_scale(np.transpose(fake_B, [1, 2, 0]), scale=8)
#
#         # print("====", down_a.max(), down_a.min())
#         # print(down_a.max(), down_a.min())
#         # bgan_dists[i] = np.sum(np.linalg.norm(fake_B - real_A))
#         bgan_dists[(i*opt.n_samples + nn)] = np.sum(np.linalg.norm(down_a - down_b))
#         add_dict(bgan_dist_bank, data["A_paths"][0], bgan_dists[i])
#     mean_bgan[data["A_paths"][0]] = np.mean(cur_all_data, axis=0)
#
# print("BGAN dist max: %f, min: %f, std: %f" % (bgan_dists.max(), bgan_dists.min(), np.std(bgan_dists)))
#
#
# # compute sigma
# all_dists = np.concatenate((caffe_dists, bgan_dists))
# print(caffe_dists.shape, bgan_dists.shape, all_dists.shape)
# sigma = np.std(all_dists)
#
#
# print(all_dists.max(), all_dists.min(), sigma)
#
# # compute caffe score
# # caffe_score = get_caffe_weighted(root_path, caffe_dists, sigma, caffe_mean)
# caffe_score = get_srim_weighted(root_path, caffe_dists, sigma, caffe_mean)
# print("SRIM score: %f"%caffe_score)
#
# bgan_score = get_bgan_weighted(databank, bgan_dist_bank, sigma, mean_bgan) / (opt.num_test * opt.n_samples)
# print("BGAN score: %f"%bgan_score)
#
# webpage.save()
#
# print("======Totoal average variance: %f" % (total_var / opt.num_test))




# test stage
# Original =================

for i, data in enumerate(islice(test_loader, opt.num_test)):
    model.set_input(data)
    print('process input image %3.3d/%3.3d' % (i, opt.num_test))
    if not opt.sync:
        z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
    cur_imgs = np.empty((opt.n_samples, 3, 256, 256))
    for nn in range(opt.n_samples + 1):
        encode = nn == 0 and not opt.no_encode
        real_A, fake_B, real_B = model.test(z_samples[[nn]], encode=encode)
        if nn == 0:
            images = [real_A, real_B, fake_B]
            names = ['input', 'ground truth', 'encoded']
        else:
            images.append(fake_B)
            names.append('random_sample%2.2d' % nn)

        # print(fake_B.shape)
        if nn > 0:
            cur_imgs[(nn-1):nn] = fake_B

        # print(a)
    # cur_var = compute_var(cur_imgs)
    # total_var += cur_var
    # print("Current Var: %f" % cur_var)

    img_path = 'input_%3.3d' % i
    save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)

webpage.save()

