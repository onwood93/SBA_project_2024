import numpy as np
import torch
import torch.nn as nn
import sys
from tslearn.metrics import dtw_path
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

e_val = sys.float_info.epsilon
# motion_feature에서 Avg를 구해서 global motion(d, )로 만들어서 contra loss에 넣기

# def cE_loss(encoded, class_num):
    
#     return nn.cross_Entropy(encoded, class_num)

# def mse_loss(origin_anchor, decoded):

#     return nn.MSELoss(origin_anchor, decoded)

# def frame_matching_loss(anchor, sp):
#     batch = anchor.shape[0]
#     dtw_list = []
#     for i in range(len(anchor)):
#         tmp_dtw_path, tmp_dtw_dist = dtw_path(anchor[i], sp[i])
#         # tmp_dtw_dist = tmp_dtw[1]

#         tmp_anchor_seq = np.array(tmp_dtw_path)[:,0]
#         tmp_sp_seq = np.array(tmp_dtw_path)[:,1]

#         anchor_re_tensor = anchor[i][tmp_anchor_seq]
#         sp_re_tensor = sp[i][tmp_sp_seq]

#         tmp_loss = (1/(tmp_dtw_dist.detach()+e_val))*torch.norm(anchor_re_tensor - sp_re_tensor,dim=1)
#         tmp_sum_loss = torch.sum(tmp_loss)
#         dtw_list.append(tmp_sum_loss)

#     dtw_loss = sum(dtw_list)/batch
#     return dtw_loss

def frame_matching_loss(anchor, sp):
    anchor_a = anchor.detach().cpu().numpy()
    sp_a = sp.detach().cpu().numpy()
    batch = anchor.shape[0]
    dtw_list = []
    for i in range(len(anchor)):
        tmp_dtw_dist, tmp_dtw_path = fastdtw(anchor_a[i], sp_a[i], dist=euclidean)
        # tmp_dtw_dist = tmp_dtw[1]

        tmp_anchor_seq = np.array(tmp_dtw_path)[:,0]
        tmp_sp_seq = np.array(tmp_dtw_path)[:,1]

        anchor_re_tensor = anchor[i][tmp_anchor_seq]
        sp_re_tensor = sp[i][tmp_sp_seq]

        tmp_loss = (1/(tmp_dtw_dist+e_val))*torch.norm(anchor_re_tensor - sp_re_tensor,dim=1)
        tmp_sum_loss = torch.sum(tmp_loss)
        dtw_list.append(tmp_sum_loss)

    dtw_loss = sum(dtw_list)/batch
    return dtw_loss

# # MSE 사용
# def recons_loss(origin, target):
#     loss = nn.MSELoss()
#     output = loss(origin, target)

#     return output