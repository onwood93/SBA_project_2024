import os
import math
import numpy as np
import random
import json
import pathlib
import torch
from torch.utils.data import Dataset
import utils.Configs as config
import torchvision.transforms.functional as ttF
import tqdm
import time

CROP_MARGIN = 50
CROP_HEIGHT = 200
CROP_WIDTH = 100
ORIG_HEIGHT = 1080
ORIG_WIDTH = 1920


def load_data_dir(data_dir):
    all_data_dir_list = []
    action_dir_dic = {}
    label_dic = {}

    for file_path in pathlib.Path(data_dir).glob("*/*.json"):
        action = file_path.name.split('_')[0][file_path.name.split('_')[0].find('A'):]
        all_data_dir_list.append(str(file_path))
        if action in action_dir_dic.keys():
            action_dir_dic[action].append(str(file_path))
        else:
            action_dir_dic[action] = [str(file_path)]
    
    for i, key in enumerate(action_dir_dic.keys()):
        label_dic[key] = i

    return all_data_dir_list, action_dir_dic, label_dic


# 컨피던스 정보 확인해야 함
class MyDataset(Dataset):
    # def __init__(self, data_dir='', num_semi_positives=10):
    #     super().__init__()
    #     self.h = config.H
    #     self.w = config.W
    #     self.data_dir = data_dir
    #     self.num_semi_positives = num_semi_positives
    #     # self.return_heatmap = return_heatmap

    #     # 데이터 경로 전체 리스트 & action별 정리된 딕셔너리
    #     self.all_data_dir_list, self.action_dir_dic, self.label_dic = load_data_dir(self.data_dir)

    # def __len__(self):
    #     return len(self.all_data_dir_list)
    
    def extract_adjusted_kps_from_a_json(self, origin_vid_dir):
        with open(origin_vid_dir, "r") as f:
            origin_keypoints = json.load(f)
            
        keypoints = np.array(origin_keypoints['annotation'], dtype=np.float32)
        crop_x1 = np.clip(keypoints[:, :, 1].min(axis=1) - CROP_MARGIN, 0, 1920)[:, None]
        crop_y1 = np.clip(keypoints[:, :, 0].min(axis=1) - CROP_MARGIN, 0, 1080)[:, None]
        crop_x2 = np.clip(keypoints[:, :, 1].max(axis=1) + CROP_MARGIN, 0, 1920)[:, None]
        crop_y2 = np.clip(keypoints[:, :, 0].max(axis=1) + CROP_MARGIN, 0, 1080)[:, None]
        
        keypoints = np.stack([
            (keypoints[..., 0] - crop_y1) * (CROP_HEIGHT/(crop_y2 - crop_y1)),
            (keypoints[..., 1] - crop_x1) * (CROP_WIDTH/(crop_x2 - crop_x1)),
        ], axis=-1)
        
        # (-1, 1) 범위로 normalize
        keypoints = np.stack([
            (keypoints[..., 0] - CROP_HEIGHT//2)/(CROP_HEIGHT//2),
            (keypoints[..., 1] - CROP_WIDTH//2)/(CROP_WIDTH//2),
        ], axis=-1)
        
        return keypoints

#     def extract_adjusted_kps_from_a_json(self, origin_vid_dir):
#         with open(origin_vid_dir, "r") as f:
#             origin_keypoints = json.load(f)
            
#         print(origin_vid_dir)
#         keypoints = np.array(origin_keypoints['annotation'])
#         return keypoints

    # C0~4 vector 계산
    def normalize_vector(self, p1, p2):
        # vector = p2 - p1
        vector = p1 - p2
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        normalized_vector = vector / norm
        return normalized_vector

    # anchor preprocessing bp별 분할 작업(C0-torso, C1-ra, C2-la, C3-rl, C4-ll)
    def compute_body_parts(self, anchor_keypoints):
        p0 = anchor_keypoints[:, 0, :]   # Nose
        p5 = anchor_keypoints[:, 5, :]   # L shoulder
        p6 = anchor_keypoints[:, 6, :]   # R shoulder
        p7 = anchor_keypoints[:, 7, :]   # L elbow
        p8 = anchor_keypoints[:, 8, :]   # R elbow
        p9 = anchor_keypoints[:, 9, :]   # L wrist
        p10 = anchor_keypoints[:, 10, :] # R wrist
        p11 = anchor_keypoints[:, 11, :] # L hip
        p12 = anchor_keypoints[:, 12, :] # R hip
        p13 = anchor_keypoints[:, 13, :] # L knee
        p14 = anchor_keypoints[:, 14, :] # R knee
        p15 = anchor_keypoints[:, 15, :] # L ankle
        p16 = anchor_keypoints[:, 16, :] # R ankle

        neck = (p5 + p6) / 2
        mid_hip = (p11 + p12) / 2

        # 방향 확인하고 수정
        c0 = np.concatenate([
            self.normalize_vector(p0, neck),
            self.normalize_vector(p5, neck),
            self.normalize_vector(p6, neck),
            self.normalize_vector(mid_hip, neck),
            self.normalize_vector(p11, mid_hip),
            self.normalize_vector(p12, mid_hip)
        ], axis=1)

        c1 = np.concatenate([
            self.normalize_vector(p7, p5),
            self.normalize_vector(p9, p7)
        ], axis=1)

        c2 = np.concatenate([
            self.normalize_vector(p8, p6),
            self.normalize_vector(p10, p8)
        ], axis=1)

        c3 = np.concatenate([
            self.normalize_vector(p13, p11),
            self.normalize_vector(p15, p13)
        ], axis=1)

        c4 = np.concatenate([
            self.normalize_vector(p14, p12),
            self.normalize_vector(p16, p14)
        ], axis=1)


        return np.expand_dims(c0, axis=0),np.expand_dims(c1, axis=0), np.expand_dims(c2, axis=0), np.expand_dims(c3, axis=0), np.expand_dims(c4, axis=0)

    # augmentation 랜덤하게 값 지정(값의 범위는 테스트를 통해 조정해야함)
    def add_aug_to_anchor(self, anchor_keypoints, aug_range=[-0.05, 0.05], augmentation=True):
        aug = np.random.uniform(aug_range[0], aug_range[1], size=anchor_keypoints.shape)
        aug_keypoints = anchor_keypoints + aug
        # aug_keypoints_list = self.compute_body_parts(aug_keypoints)

        return aug_keypoints

    def compute_heatmap(self, keypoints, std_x=0.1, std_y=0.05):
        h = np.linspace(-1, 1, CROP_HEIGHT, dtype=np.float32)[None, None, :]
        w = np.linspace(-1, 1, CROP_WIDTH, dtype=np.float32)[None, None, :]
        
        log_y_gaussian = -0.5 * ((h - keypoints[..., 0:1])/std_y)**2 # num_frames, num_keypoints, CROP_HEIGHT
        log_x_gaussian = -0.5 * ((w - keypoints[..., 1:2])/std_x)**2 # num_frames, num_keypoints, CROP_WIDTH
        log_gaussian = log_x_gaussian[:, :, None, :] + log_y_gaussian[:, :, :, None]
        gaussian = np.exp(log_gaussian)
        heatmaps = gaussian / gaussian.max(axis=(2, 3), keepdims=True) * 2 - 1

        return heatmaps
        # (frames, 17, 270, 480)

#     def compute_heatmap(self, keypoints, std=5, rate=0.25):
#         y_points = (keypoints[:, :,[0]] * rate).reshape(-1,1) # 561, 1
#         x_points = (keypoints[:, :,[1]] * rate).reshape(-1,1) # 561, 1

#         H = np.arange(self.h)[None, :] # 1, 270
#         W = np.arange(self.w)[None, :] # 1, 480

#         # H - y_points: (561, 270)
#         log_y_gauss = -0.5 * ((H - y_points)/std)**2 # (561, 270)
#         log_x_gauss = -0.5 * ((W - x_points)/std)**2 # (561, 480)
#         log_gauss = log_x_gauss[:, None, :] + log_y_gauss[:, :, None]
#         gauss = np.exp(log_gauss)
#         heatmaps = (gauss / gauss.max(axis=(-1, -2), keepdims=True)) * 2 - 1

#         heatmaps = heatmaps.reshape(-1, 17, self.h, self.w)

#         #-----------------------------------------------
#         # points = keypoints.reshape(-1,2) * rate

#         # x_points = points[:,[0]]
#         # y_points = points[:,[1]]

#         # # H = np.tile(np.arange(h), h).reshape(h,h).T
#         # # W = np.tile(np.arange(reduced_w), reduced_w).reshape(reduced_w,reduced_w)
#         # # H = np.tile(np.arange(h)[:, None], (h, w))
#         # # W = np.tile(np.arange(w)[None, :], (h, w))
#         # H = np.tile(np.arange(self.h)[:, None], self.w)
#         # W = np.tile(np.arange(self.w)[:, None], self.h).T

#         # heatmap_list = []
#         # for i in range(len(x_points)):
#         #     x = x_points[i]
#         #     y = y_points[i]
#         #     # x_gauss = 1/(np.sqrt(2*np.pi)*std) * np.exp(-0.5*((H-x)/std)**2)
#         #     # y_gauss = 1/(np.sqrt(2*np.pi)*std) * np.exp(-0.5*((W-y)/std)**2)
#         #     y_gauss = np.exp(-0.5*((H-x)/std)**2)
#         #     x_gauss = np.exp(-0.5*((W-y)/std)**2)
#         #     frame = (x_gauss * y_gauss)
#         #     frame = (frame / frame.max()) * 2 - 1
#         #     heatmap_list.append(frame)
#         # heatmaps = np.array(heatmap_list).reshape(-1, 17, self.h, self.w)

#         return heatmaps
#         # (frames, 17, 270, 480)
    
    def compute_optical_flow(self, keypoints, heatmaps, rate=0.25):
        flow = keypoints[1:] - keypoints[:-1]

        opt_flow = heatmaps[1:, :, :, :, None] * flow[:, :, None, None, :] # (num_of_frames, num_of_keypoints, H, W, 2)
        F, K, H, W, _ = opt_flow.shape
        opt_flow = opt_flow.transpose(0, 1, 4, 2, 3).reshape(-1, K*2, H, W)
        return opt_flow
    
#     def compute_optical_flow(self, keypoints, heatmaps, rate=0.25):
#         reduced_keypoints = keypoints * rate

#         # flow_x = []
#         # flow_y = []
#         # for i in range(1, len(reduced_keypoints)):
#         #     flow_x.append((reduced_keypoints[i] - reduced_keypoints[i-1])[:,[0]])
#         #     flow_y.append((reduced_keypoints[i] - reduced_keypoints[i-1])[:,[1]])
#         flow = reduced_keypoints[1:] - reduced_keypoints[:-1]
#         # flow_x = flow[:,:,[0]]
#         # flow_y = flow[:,:,[1]]
#         # hw_zeros = torch.zeros(size=(self.h, self.w)) # 빈 판
#         # flowmap = torch.zeros((len(flow), 2, self.h, self.w))
#         # for i in range(len(flow)):
#         #     flowmap[i][0] = flowmap[i][0] + flow[i][0]
#         #     flowmap[i][1] = flowmap[i][1] + flow[i][1]

#         # heatmaps -> (num_of_frames, num_of_keypoints, H, W)
#         # flow -> (num_of_frames - 1, num_of_keypoints, 2)

#         # tmp_x = ((hw_zeros + flow_x[i-1][j]) / 270) * 2 - 1
#         # tmp_y = ((hw_zeros + flow_y[i-1][j]) / 480) * 2 - 1
#         flow = np.stack([
#             (flow[..., 0] / 270) * 2 - 1,
#             (flow[..., 1] / 480) * 2 - 1
#         ], axis=-1)

#         # opt_flow_x.append(tmp_x * heatmap[j])
#         # opt_flow_y.append(tmp_y * heatmap[j])
#         opt_flow = heatmaps[1:, :, :, :, None] * flow[:, :, None, None, :] # (num_of_frames, num_of_keypoints, H, W, 2)
#         F, K, H, W, _ = opt_flow.shape
#         opt_flow = opt_flow.transpose(0, 1, 4, 2, 3).reshape(-1, K*2, H, W)
#         return opt_flow

        # opt_flow_x = []
        # opt_flow_y = []
        # for i in range(1,len(heatmaps)):
        #     heatmap = heatmaps[i]
        #     for j in range(len(heatmap)):
        #         tmp_x = ((hw_zeros + flow_x[i-1][j]) / 270) * 2 - 1
        #         tmp_y = ((hw_zeros + flow_y[i-1][j]) / 480) * 2 - 1
        #         opt_flow_x.append(tmp_x * heatmap[j])
        #         opt_flow_y.append(tmp_y * heatmap[j])

        # reshape_opt_flow_x = (np.array(opt_flow_x)).reshape(-1,17,self.h,self.w)
        # reshape_opt_flow_y = (np.array(opt_flow_y)).reshape(-1,17,self.h,self.w)

        # return np.concatenate((reshape_opt_flow_x,reshape_opt_flow_y),axis=1)
    
    # def compute_optical_flow(self, keypoints, heatmaps, rate=0.25):
    #     reduced_keypoints = keypoints * rate

    #     # flow_x = []
    #     # flow_y = []
    #     # for i in range(1, len(reduced_keypoints)):
    #     #     flow_x.append((reduced_keypoints[i] - reduced_keypoints[i-1])[:,[0]])
    #     #     flow_y.append((reduced_keypoints[i] - reduced_keypoints[i-1])[:,[1]])
    #     flow = reduced_keypoints[1:] - reduced_keypoints[:-1]
    #     flow_x = flow[:,:,[0]]
    #     flow_y = flow[:,:,[1]]
    #     hw_zeros = torch.zeros(size=(self.h, self.w)) # 빈 판
    #     # flowmap = torch.zeros((len(flow), 2, self.h, self.w))
    #     # for i in range(len(flow)):
    #     #     flowmap[i][0] = flowmap[i][0] + flow[i][0]
    #     #     flowmap[i][1] = flowmap[i][1] + flow[i][1]

    #     opt_flow_x = []
    #     opt_flow_y = []
    #     for i in range(1,len(heatmaps)):
    #         heatmap = heatmaps[i]
    #         for j in range(len(heatmap)):
    #             tmp_x = ((hw_zeros + flow_x[i-1][j]) / 270) * 2 - 1
    #             tmp_y = ((hw_zeros + flow_y[i-1][j]) / 480) * 2 - 1
    #             opt_flow_x.append(tmp_x * heatmap[j])
    #             opt_flow_y.append(tmp_y * heatmap[j])

    #     reshape_opt_flow_x = (np.array(opt_flow_x)).reshape(-1,17,self.h,self.w)
    #     reshape_opt_flow_y = (np.array(opt_flow_y)).reshape(-1,17,self.h,self.w)

    #     return np.concatenate((reshape_opt_flow_x,reshape_opt_flow_y),axis=1)
    #     # return np.concatenate((opt_flow_x, opt_flow_y),axis=1)
    #     # (frames-1, 34, 270, 480)
    
    def adjusting_frame(self, origin_keypoint):
        if len(origin_keypoint) == 33:
            return origin_keypoint
        
        elif len(origin_keypoint) > 33:
            linspace_num = np.linspace(0, len(origin_keypoint), 33)# 적어도 sorting해줄 것 / interpolation or np.linspace로 추출하는 방법 권장
            linspace_origin_keypoint = [origin_keypoint[int(i)] for i in linspace_num]
            return np.array(linspace_origin_keypoint)

        elif len(origin_keypoint) < 33:
            num_repet = math.ceil((33 - len(origin_keypoint)) / len(origin_keypoint) + 1)
            extended_origin = []
            for i in range(num_repet):
                if i % 2 == 0:
                    tmp_origin = origin_keypoint
                elif i % 2 == 1:
                    tmp_origin = origin_keypoint[::-1]
                extended_origin.append(tmp_origin)
            extended_origin = np.array(extended_origin).reshape(-1,17,2)[0:33]
            return extended_origin

    def expanding_sp_dataset(self, sp_keypoints):
        # 프레임이 가장 많은 개수 찾기
        num_sec_dim = []
        for i in range(len(sp_keypoints)):
            num_sec_dim.append(len(sp_keypoints[i]))

        max_frame = max(num_sec_dim)

        # reflect일 경우, replicate일 경우
        ran_increase = random.choice(['reflect', 'replicate'])
        increased_sp_keypoints=[]
        for i in range(len(sp_keypoints)):
            frame = sp_keypoints[i]
            if len(frame) == 0:
                print("시퀀스가 0인게 있다")
            if ran_increase == 'reflect':
                if num_sec_dim[i] == max_frame:
                    increased_sp_keypoints.append(frame)
                else:
                    tmp_reflect_frames = np.concatenate([frame[::-1],frame,frame[::-1]], axis=0)
                    ref_cnt = len(frame) * 3
                    multi_cnt = 0
                    while True:
                        # print(cnt, multi_cnt)
                        if ref_cnt > max_frame:
                            break
                        tmp_reflect_frames = np.concatenate([tmp_reflect_frames, frame, frame[::-1]], axis=0)
                        ref_cnt += len(frame)*2
                        multi_cnt += 1
                    
                    if max_frame % 2 == 0:
                        reflected = tmp_reflect_frames[(ref_cnt//2)-(max_frame//2):(ref_cnt//2)+(max_frame//2)]
                        increased_sp_keypoints.append(reflected)
                    else:
                        reflected = tmp_reflect_frames[(ref_cnt//2)-(max_frame//2):(ref_cnt//2)+(max_frame//2)+1]
                        increased_sp_keypoints.append(reflected)
            
            elif ran_increase == 'replicate':
                if num_sec_dim[i] == max_frame:
                    increased_sp_keypoints.append(frame)
                else:
                    rep_cnt = len(frame) * 3
                    add_cnt = len(frame)
                    while True:
                        if rep_cnt > max_frame:
                            break
                        rep_cnt += len(frame)
                        add_cnt += len(frame)
                    
                    tmp_replicate_frames = np.concatenate([np.tile(frame[:1],[add_cnt,1,1]), frame, np.tile(frame[-1:],[add_cnt,1,1])], axis=0)
                    if max_frame % 2 == 0:
                        replicated = tmp_replicate_frames[(rep_cnt//2)-(max_frame//2):(rep_cnt//2)+(max_frame//2)]
                        increased_sp_keypoints.append(replicated)
                    else:
                        replicated = tmp_replicate_frames[(rep_cnt//2)-(max_frame//2):(rep_cnt//2)+(max_frame//2)+1]
                        increased_sp_keypoints.append(replicated)
        
        return increased_sp_keypoints
        
    def semi_positives_maker(self):
        # a의 shape = (frame, -1)
        # reflection
        # reflected = np.concatenate([frame[::-1], frame, frame[::-1]], axis=0)
        # reflected = reflected[start:end]

        # replicated = np.concatenate([np.tile(frame[:1], [N, 1]), frame,  np.tile(frame[-1:], [N, 1])], axis=0)
        # frame[:1] -> (1, 17) 
        # np.tile(frame[:1], [N, 1]) -> (N, 17)

        # # 랜덤으로 선택된 anchor의 action과 동일한 semi-positives 선정
        # 같은 동작에서 10개를 뽑는 걸로 생각하면 됨
        sp_keypoints = []
        semi_positive_dirs = random.sample(self.action_dir_dic[self.action], self.num_semi_positives)
        # print("sp: ", semi_positive_dirs)

        for p in semi_positive_dirs:
            a_semi_positive_keypoints = self.extract_adjusted_kps_from_a_json(p)

            sp_keypoints.append(a_semi_positive_keypoints)

        return np.array(sp_keypoints)
    
    def make_anchor_set(self, all_data_dir_list):
        anchor_set_list = []
        for i in tqdm.tqdm(range(len(all_data_dir_list))):
            origin_anchor_keypoints = self.extract_adjusted_kps_from_a_json(all_data_dir_list[i])
            origin_heatmaps = self.compute_heatmap(origin_anchor_keypoints)
            origin_flow = self.compute_optical_flow(origin_anchor_keypoints, origin_heatmaps)
            resized_origin_heatmap, resized_origin_flow = self.crop_keypoints(origin_anchor_keypoints, origin_heatmaps, origin_flow)
            norm_adjusted_keypoints = self.keypoints_normalize(origin_anchor_keypoints)
            anchor_set_list.append([all_data_dir_list[i], norm_adjusted_keypoints[1:], resized_origin_heatmap[1:], resized_origin_flow])
        return anchor_set_list
    
    def __init__(self, data_dir='', num_semi_positives=10):
        super().__init__()
        self.h = config.H
        self.w = config.W
        self.data_dir = data_dir
        self.num_semi_positives = num_semi_positives
        # self.return_heatmap = return_heatmap

        # 데이터 경로 전체 리스트 & action별 정리된 딕셔너리
        self.all_data_dir_list, self.action_dir_dic, self.label_dic = load_data_dir(self.data_dir)
        # self.anchor_set_list = self.make_anchor_set(self.all_data_dir_list)

    def __len__(self):
        return len(self.all_data_dir_list)
    
    def __getitem__(self, index):
        """
        index 번째 데이터 가져오고 augmentation 해서 anchor, anchor_aug, semi_positives들 만드는 코드
        anchor: 임의의 비디오 1개 (num_of_frames, )
        anchor_aug: anchor에서 augmentation 한 것 (augmentation된 비디오 1개)
        semi_positives: anchor와 같은 동작이지만, anchor가 아닌 다른 비디오 num_semi_positives개
        """

        # SHAPE
        # anchor_c0: (1, number of frames, 12)
        # anchor_c1 ~ anchor_c4: (1, number of frames, 4)
        # anchor_aug_c0: (1, number of frames, 12)
        # anchor_aug_c1 ~ anchor_c4: (1, number of frames, 4)
        # semi_positives_c0: (num_semi_positives, number of frames, 12)
        # semi_positives_c1 ~ anchor_c4: (num_semi_positives, number of frames, 4)

        # {
        #    "anchor_c0": anchor_c0,
        #    "anchor_aug_c0": anchor_aug_c0,
        #    "semi_positives_c0": semi_positives_c0,

        #    "anchor_c1": anchor_c1,
        #    "anchor_aug_c1": anchor_aug_c1,
        #    "semi_positives_c1": semi_positives_c1,

        #     # ...
        # }
        # origin_vid_dir 랜덤하게 선정
        # self.random_vid_dir = random.choice(self.all_data_dir_list)
        
        start_time = time.time()
        
        #------------------------------------------------------------------
        self.index_vid_dir = self.all_data_dir_list[index]
        # print(index, self.index_vid_dir)
        # chosen_anchor = self.anchor_set_list[index]
        # chosen_anchor shape => [all_data_dir_list[i], norm_adjusted_keypoints[1:], resized_origin_heatmap[1:], resized_origin_flow]
        # key_fn = chosen_anchor[0].split('/')[-1].split('_')[0]
        key_fn = self.index_vid_dir.split('/')[-1].split('_')[0]
        self.action = key_fn[key_fn.find('A'):]

        input_data = {}
        # 위의 형식대로 input_data에 집어넣기?
        #------------------------------------------------------------------
        origin_anchor_keypoints = self.extract_adjusted_kps_from_a_json(self.index_vid_dir)
        #------------------------------------------------------------------
        # print("1, extract_adjusted_kps_from_a_json", time.time() - start_time)

        # anchor_keypoints = self.compute_body_parts(origin_anchor_keypoints) # 5 * (frames, 12 or 4)

        # # aug_anchor_keypoints = self.extract_keypoints_from_a_json(self.index_vid_dir) # (frames, 17, 2)
        # aug_keypoints = self.add_aug_to_anchor(origin_anchor_keypoints) # 5 * (frames, 12 or 4)

        # sp_10 = self.semi_positives_maker() # (10*frames,17,2)
        # sp_10_keypoints = self.compute_body_parts(sp_10) # 5 * (10*frames, 12 or 4)

        start_time = time.time()
        # adjusted_keypoints = self.adjusting_frame(origin_anchor_keypoints)
        #------------------------------------------------------------------
        origin_heatmaps = self.compute_heatmap(origin_anchor_keypoints)
        origin_flow = self.compute_optical_flow(origin_anchor_keypoints, origin_heatmaps)
        # print(origin_heatmaps.shape, origin_flow.shape)
        #------------------------------------------------------------------
        # print("2, compute_heatmap, compute_optical_flow", time.time() - start_time)
       
        # aug_keypoints = self.add_aug_to_anchor(adjusted_keypoints)
        # aug_heatmaps = self.compute_heatmap(aug_keypoints)
        # aug_flow = self.compute_optical_flow(aug_keypoints, aug_heatmaps)
        # resized_aug_heatmap, resized_aug_flow = self.crop_keypoints(aug_keypoints, aug_heatmaps, aug_flow)

        start_time = time.time()
        #------------------------------------------------------------------
        sp_10 = self.semi_positives_maker()
        # print("3, semi_positives_maker", time.time() - start_time)

        sp_heatmaps = list(map(lambda i: self.compute_heatmap(sp_10[i]), range(len(sp_10))))
        sp_flow = list(map(lambda i: self.compute_optical_flow(sp_10[i], sp_heatmaps[i]), range(len(sp_10))))
        
        # sp_heatmaps = []
        # sp_flow = []
        # for sp_1 in sp_10:
        #     tmp_sp_heatmap = self.compute_heatmap(sp_1)
        #     tmp_sp_flow = self.compute_optical_flow(sp_1, tmp_sp_heatmap)
        #     sp_heatmaps.append(tmp_sp_heatmap)
        #     sp_flow.append(tmp_sp_flow)
        #------------------------------------------------------------------
        # print("4, semi_positives_maker", time.time() - start_time)

        input_data['anchor'] = torch.tensor(origin_anchor_keypoints[1:], dtype=torch.float32) # (32, 17, 2)
        input_data['anchor_heatmap'] = torch.tensor(origin_heatmaps[1:]) # (32, 17, 200, 100)
        input_data['anchor_flow'] = torch.tensor(origin_flow) # (32, 34, 200, 100)

        #################################################################
        sp_heatmaps = torch.tensor(np.stack(sp_heatmaps, 0)).type(torch.float32)
        sp_flow = torch.tensor(np.stack(sp_flow, 0)).type(torch.float32)
        
        input_data['semi_positives'] = torch.tensor(sp_10[:,1:,:,:], dtype=torch.float32) # (num_semi_positives, 32, 17, 2)
        input_data['sp_heatmap'] = sp_heatmaps[:,1:,:,:,:] # (num_semi_positives, 32, 17, 270, 480)
        input_data['sp_flow'] = sp_flow # (num_semi_positivest, 32, 34, 270, 480)

        input_data['class'] = torch.tensor(self.label_dic[self.action])
        ####################################################################
        
        return input_data


# class MyDataset2(MyDataset):

#     def __init__(self, data_dir):
#         super().__init__(data_dir=data_dir)

#     def _compute_heatmap(self, keypoints):
#         return None
    
#     def __getitem__(self, index):
#         data_item = super().__getitem__(index)
#         anchor_c0 = data_item["anchor_c0"]
#         anchor_c0_heatmap = self._compute_heatmap(anchor_c0)

#         return {
#             **data_item,
#             "anchor_c0_heatmap": anchor_c0_heatmap,
#             # ...
#         }

