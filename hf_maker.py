import tqdm
import numpy as np
import json
import pathlib
import torch
# import torchvision.transforms.functional as ttF


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

def extract_adjusted_kps_from_a_json(origin_vid_dir):
    with open(origin_vid_dir, "r") as f:
        origin_keypoints = json.load(f)

    return np.array(origin_keypoints['annotation'])

def compute_heatmap(keypoints, std=5):
        h = 200
        w = 100

        y_points = (keypoints[:, :,[0]]).reshape(-1,1) # 561, 1
        x_points = (keypoints[:, :,[1]]).reshape(-1,1) # 561, 1

        H = np.arange(h)[None, :] # 1, 200
        W = np.arange(w)[None, :] # 1, 100

        # H - y_points: (561, 200)
        log_y_gauss = -0.5 * ((H - y_points)/std)**2 # (561, 200)
        log_x_gauss = -0.5 * ((W - x_points)/std)**2 # (561, 100)
        log_gauss = log_x_gauss[:, None, :] + log_y_gauss[:, :, None]
        gauss = np.exp(log_gauss)
        heatmaps = (gauss / gauss.max(axis=(-1, -2), keepdims=True)) * 2 - 1

        heatmaps = heatmaps.reshape(-1, 17, h, w)

        return heatmaps
        # (frames, 17, 200, 100)
    
def compute_optical_flow(keypoints, heatmaps):
    # reduced_keypoints = keypoints
    flow = keypoints[1:] - keypoints[:-1]
    
    flow = np.stack([
        (flow[..., 0] / 200) * 2 - 1,
        (flow[..., 1] / 100) * 2 - 1
    ], axis=-1)

    opt_flow = heatmaps[1:, :, :, :, None] * flow[:, :, None, None, :] # (num_of_frames, num_of_keypoints, H, W, 2)
    F, K, H, W, _ = opt_flow.shape
    opt_flow = opt_flow.transpose(0, 1, 4, 2, 3).reshape(-1, K*2, H, W)
    return opt_flow

def crop_keypoints(keypoints, margin=50):
        # tmp_origin = keypoints * 0.25
        origin_x_max = int(np.max(keypoints[:,:,0]))
        origin_y_max = int(np.max(keypoints[:,:,1]))
        origin_x_min = int(np.min(keypoints[:,:,0]))
        origin_y_min = int(np.min(keypoints[:,:,1]))
        # print(origin_x_max, origin_x_min, origin_y_max, origin_y_min)

        x_min_margin = max(origin_x_min-margin, 0)
        y_min_margin = max(origin_y_min-margin, 0)

        keypoints[:,:,0] = (keypoints[:,:,0] - (x_min_margin)) / (origin_x_max+margin) * 200
        keypoints[:,:,1] = (keypoints[:,:,1] - (y_min_margin)) / (origin_y_max+margin) * 100

        # min - margin > 0 처리하면 될 것 같음
        # crop_heatmap = torch.tensor(heatmaps[:,:,x_min_margin:origin_x_max+margin,y_min_margin:origin_y_max+margin], dtype=torch.float32)
        # resized_crop_heatmap = ttF.resize(crop_heatmap, (200,100))
        
        # crop_flow = torch.tensor(flows[:,:,x_min_margin:origin_x_max+margin,y_min_margin:origin_y_max+margin], dtype=torch.float32)
        # resized_crop_flow = ttF.resize(crop_flow, (200,100))

        return keypoints
    
def keypoints_normalize(keypoints):

    if len(keypoints.shape) == 4:
        # print(keypoints.shape)
        keypoints[:,:,:,[0]] = (keypoints[:,:,:,[0]]/200)*2-1
        keypoints[:,:,:,[1]] = (keypoints[:,:,:,[1]]/100)*2-1
        # normalized_keypoints = np.concatenate([(keypoints[:,:,:,[0]]/1080)*2-1, (keypoints[:,:,:,[1]]/1920)*2-1], axis = 3)
        
    else:
        # print(keypoints.shape)
        # print(keypoints[:,:,[0]].shape)
        # normalized_keypoints = np.concatenate([(keypoints[:,:,[0]]/1080)*2-1, (keypoints[:,:,[1]]/1920)*2-1], axis = 2)
        keypoints[:,:,[0]] = (keypoints[:,:,[0]]/200)*2-1
        keypoints[:,:,[1]] = (keypoints[:,:,[1]]/100)*2-1

    return keypoints

# data_dir = 'adjusted_100_dataset'
# total_dir,_,label_dic = load_data_dir(data_dir)
# total_dir = total_dir[:3650]

# for i in tqdm.tqdm(range(len(total_dir))):
#     extracted = extract_adjusted_kps_from_a_json(total_dir[i])
#     resized_keypoints = crop_keypoints(extracted)

#     # adjusted = adjusting_frame(extracted)
#     heatmap = compute_heatmap(resized_keypoints)
#     flow = compute_optical_flow(resized_keypoints, heatmap)

#     norm_extracted = keypoints_normalize(resized_keypoints)

#     key_fn = total_dir[i].split('/')[-1].split('_')[0]
#     action = key_fn[key_fn.find('A'):]
#     action_label = label_dic[action]
    
#     save_file_path = total_dir[i].replace('adjusted_100_dataset','/data/onwood/ahf_100').replace('.json','.npz')

#     np_dic = {'file_name':save_file_path,
#                 'anchor':norm_extracted.astype(np.float32),
#                 'heatmap':heatmap.astype(np.float32),
#                 'flow':flow.astype(np.float32),
#                 'action':action_label}
                
#     # json_dic = {'annotation':adjusted.tolist()}
    
#     np.savez_compressed(save_file_path, **np_dic)