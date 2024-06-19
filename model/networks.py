import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# 프레임 개수 고정(32개, uniform하게 33장 추출), mydataset에서 코드 추가 작성 필요, interpolation
# gpu에 올려서 샘플 동시에 몇 개까지 가능한지 확인(최소 5개는 되면 좋겠음)
# Encoder 3개 하나로 합치기
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.h_seq = nn.Sequential(nn.Conv2d(17, 16, kernel_size=11),
                                   # nn.LeakyReLU(0.2),
                                   nn.Tanh(),
                                   nn.BatchNorm2d(16),
                                   # nn.MaxPool2d((2,2)),
                                   nn.Conv2d(16, 8, kernel_size=7),
                                   # nn.LeakyReLU(0.2),
                                   nn.Tanh(),
                                   nn.BatchNorm2d(8),
                                   nn.MaxPool2d((2,2)),
                                   nn.Conv2d(8, 3, kernel_size=3))

                                    
        
        self.f_seq = nn.Sequential(nn.Conv2d(34, 16, kernel_size=11),
                                   # nn.LeakyReLU(0.2), 
                                   nn.Tanh(),
                                   nn.BatchNorm2d(16),
                                   # nn.MaxPool2d((2,2)),
                                   nn.Conv2d(16, 8, kernel_size=7),
                                   # nn.LeakyReLU(0.2),
                                   nn.Tanh(),
                                   nn.BatchNorm2d(8),
                                   nn.MaxPool2d((2,2)),
                                   nn.Conv2d(8, 3, kernel_size=3))
        
        self.seq_3D = nn.Sequential(nn.Conv3d(6,8,kernel_size=11),
                                    #  nn.LeakyReLU(0.2),
                                    nn.Tanh(),
                                    nn.BatchNorm3d(8),
                                    nn.Conv3d(8,12,kernel_size=9),
                                    # nn.LeakyReLU(0.2),
                                    nn.Tanh(),
                                    nn.BatchNorm3d(12),
                                    nn.Conv3d(12,16,kernel_size=7),
                                    # nn.LeakyReLU(0.2),
                                    nn.Tanh(),
                                    nn.BatchNorm3d(16),
                                    nn.Conv3d(16,24,kernel_size=5),
                                    # nn.LeakyReLU(0.2),
                                    nn.Tanh(),
                                    nn.BatchNorm3d(24),
                                    nn.Conv3d(24,32,kernel_size=3))

        self.lin_seq = nn.Sequential(nn.Linear(1200, 512),
                                     nn.ReLU(),
                                     nn.Linear(512,128),
                                     nn.ReLU(),
                                     nn.Linear(128,73))
        
    #     self.lin_for_cE = nn.Linear(128,73)
        
    # def for_cE(self, input):
    #     output = self.lin_for_cE(input)
    #     return output
    
    def forward(self, h_feat, f_feat):
        if len(h_feat.shape)==6:
            num_semi = h_feat.shape[1]
        else:
            num_semi = 1
        batch = h_feat.shape[0]
        H = h_feat.shape[-2]
        W = h_feat.shape[-1]
        # print(batch, H, W)
        h_feat = self.h_seq(h_feat.reshape(-1,17,H,W))
        # print("h_feat: ", h_feat.shape)
        f_feat = self.f_seq(f_feat.reshape(-1,34,H,W))
        # print("f_feat: ", f_feat.shape)

        hf_feat = torch.cat((h_feat, f_feat), dim=1).reshape(batch*num_semi, 32, 6, 90, 40).transpose(2,1)
        # print("hf_feat: ", hf_feat.shape)
    
        motion_feature = self.seq_3D(hf_feat).reshape(batch*num_semi,32,-1)
        # print("motion_feat_3D: ", motion_feature.shape)

        motion_feature = self.lin_seq(motion_feature)
        # print("motion_feat_linear: ", motion_feature.shape)

        return motion_feature

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.linears = nn.ModuleList([nn.Linear(128, 2) for i in range(17)])
        # 프레임 개수만큼 벡터 생성 (프레임 하나로 만들어지면 정확성이 떨어지지 않을까해서)
        # 그렇다면 sequential - linear acti linear 방식
        # self.linears_1 = nn.ModuleList([nn.Linear(128, 32) for i in range(17)])
        # self.linears_2 = nn.ModuleList([nn.Linear(32, 2) for i in range(17)])

        # self.linears_1 = nn.Linear(128, 32 * 17)
        # self.linears_2 = nn.Linear(32, 2 * 17)

        self.linears = nn.Sequential(nn.Linear(73, 32 * 17),
                                     nn.ReLU(),
                                     nn.Linear(32 * 17, 2 * 17))
        # 1번은 분해하는 역할 2번은 압축시켜주는 역할

    # forward도 수정 필요
    def forward(self, x):
        batch = x.shape[0]
        x = self.linears(x)
        # print("decoder: ", x.shape)

        return x.reshape(batch,32,17,2)

# class heatmap_Encoder(nn.Module):
#     def __init__(self):
#         super(heatmap_Encoder, self).__init__()
#         self.h_seq = nn.Sequential(nn.Conv2d(17, 16, kernel_size=11),
#                                     nn.LeakyReLU(0.2),
#                                     nn.BatchNorm2d(16),
#                                     # nn.MaxPool2d((2,2)),
#                                     nn.Conv2d(16, 8, kernel_size=7),
#                                     nn.LeakyReLU(0.2),
#                                     nn.BatchNorm2d(8),
#                                     nn.MaxPool2d((2,2)),
#                                     nn.Conv2d(8, 3, kernel_size=3))

#     def forward(self, h_feat):
#         h_feat = self.h_seq(h_feat)

#         return h_feat

# class flow_Encoder(nn.Module):
#     def __init__(self):
#         super(flow_Encoder, self).__init__()
#         self.f_seq = nn.Sequential(nn.Conv2d(34, 16, kernel_size=11),
#                                     nn.LeakyReLU(0.2),
#                                     nn.BatchNorm2d(16),
#                                     # nn.MaxPool2d((2,2)),
#                                     nn.Conv2d(16, 8, kernel_size=7),
#                                     nn.LeakyReLU(0.2),
#                                     nn.BatchNorm2d(8),
#                                     nn.MaxPool2d((2,2)),
#                                     nn.Conv2d(8, 3, kernel_size=3))

#     def forward(self, f_feat):
#         f_feat = self.f_seq(f_feat)

#         return f_feat

# class encoder_3D(nn.Module):
#     def __init__(self):
#         super(encoder_3D, self).__init__()
#         self.e3D_seq = nn.Sequential(nn.Conv3d(6,8,kernel_size=11),
#                                      nn.LeakyReLU(0.2),
#                                      nn.BatchNorm3d(8),
#                                      nn.Conv3d(8,12,kernel_size=9),
#                                      nn.LeakyReLU(0.2),
#                                      nn.BatchNorm3d(12),
#                                      nn.Conv3d(12,16,kernel_size=7),
#                                      nn.LeakyReLU(0.2),
#                                      nn.BatchNorm3d(16),
#                                      nn.Conv3d(16,24,kernel_size=5),
#                                      nn.LeakyReLU(0.2),
#                                      nn.BatchNorm3d(24),
#                                      nn.Conv3d(24,32,kernel_size=3))

#         self.lin_seq = nn.Sequential(nn.Linear(38000, 512),
#                                      nn.ReLU(),
#                                      nn.Linear(512,128))
#         # self.linear = nn.Linear(38000, 128)
#         # self.lin_seq = nn.Sequential(nn.Linear(4920, 512),
#         #                              nn.Linear(512,128))
        
#     def forward(self, motion_feature):
#         motion_feature = self.e3D_seq(motion_feature).reshape(32,-1)
#         motion_feature = self.lin_seq(motion_feature)

#         return motion_feature
    
# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         # self.linears = nn.ModuleList([nn.Linear(128, 2) for i in range(17)])
#         # 프레임 개수만큼 벡터 생성 (프레임 하나로 만들어지면 정확성이 떨어지지 않을까해서)
#         # 그렇다면 sequential - linear acti linear 방식
#         # self.linears_1 = nn.ModuleList([nn.Linear(128, 32) for i in range(17)])
#         # self.linears_2 = nn.ModuleList([nn.Linear(32, 2) for i in range(17)])

#         # self.linears_1 = nn.Linear(128, 32 * 17)
#         # self.linears_2 = nn.Linear(32, 2 * 17)

#         self.linears = nn.Sequential(nn.Linear(128, 32 * 17),
#                                      nn.ReLU(),
#                                      nn.Linear(32 * 17, 2 * 17))
#         # 1번은 분해하는 역할 2번은 압축시켜주는 역할

#     # forward도 수정 필요
#     def forward(self, x):
#         x_list = []
#         for i in range(len(x)):
#             for j in range(17):
#                 # tmp_x = self.linears[j](x[i])
#                 tmp_x = self.linears_1[j](x[i])
#                 tmp_x = self.linears_2[j](tmp_x)
#                 x_list.append(tmp_x)

#         return torch.stack(x_list, dim=0).reshape(32,17,2)