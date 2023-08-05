import cv2
import sys
import torch
import torch.nn.functional as F
import numpy as np
from real_sense_stereo_camera import (RealSenseStereoCamera)
from stereo_calibration import StereoCalibration
from patchnetvlad import PatchNetVLAD
from torchvision import models
from database import Database
from patch_matcher import PatchMatcher
from local_matcher import calc_keypoint_centers_from_patches 
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def concat_imgs(images,shape,size,scores,i,type):
    #shape is Grid shape
    width,height = size # Image size
    image = Image.new('RGB',(width*shape[1],height*shape[0]))
    index = 0
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width*col, height*row
            img = images[index]#.convert('RGB')
            img = img.resize(size)
            image.paste(img,offset)
            index += 1

    image.save(f"imgs/{type}_img{i}_s{scores}.png",'png')
    print(f"{type} scores",scores)

def get_backend():
    enc_dim = 512
    enc = models.vgg16(pretrained=True)

    layers = list(enc.features.children())[:-2]
    # only train conv5_1, conv5_2, and conv5_3 (leave rest same as Imagenet trained weights)
    for layer in layers[:-5]:
        for p in layer.parameters():
            p.requires_grad = False
    enc = torch.nn.Sequential(*layers)
    return enc_dim, enc

class Flatten(torch.nn.Module):
    def forward(self, input_data):
        return input_data.view(input_data.size(0), -1)

class L2Norm(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input_data):
        return F.normalize(input_data, p=2, dim=self.dim)
    
def apply_patch_weights(input_scores, num_patches, patch_weights):
    output_score = 0
    if len(patch_weights) != num_patches:
        raise ValueError('The number of patch weights must equal the number of patches used')

    for i in range(num_patches):
        output_score = output_score + (patch_weights[i] * input_scores[i])
    return output_score

def get_pca_encoding(model, vlad_encoding):
    pca_encoding = model.WPCA(vlad_encoding.unsqueeze(-1).unsqueeze(-1))
    return pca_encoding

def match_two(model, img1, img2, device, config):
    pool_size = int(config['global_params']['num_pcs'])

    model.eval()

    input_data = torch.cat((img1.to(device), img2.to(device)), 0)

    with torch.no_grad():
        vlad_local, _ = model(input_data)

        local_feats_one = []
        local_feats_two = []
        for this_local in vlad_local:
            this_local_feats = get_pca_encoding(model, this_local.permute(2, 0, 1).reshape(-1, this_local.size(1))). \
                reshape(this_local.size(2), this_local.size(0), pool_size).permute(1, 2, 0)
            
            local_feats_one.append(torch.transpose(this_local_feats[0, :, :], 0, 1))
            local_feats_two.append(this_local_feats[1, :, :])

    patch_sizes = [int(s) for s in config['global_params']['patch_sizes'].split(",")]
    strides = [int(s) for s in config['global_params']['strides'].split(",")]
    patch_weights = np.array(config['feature_match']['patchWeights2Use'].split(",")).astype(float)

    all_keypoints = []
    all_indices = []

    for patch_size, stride in zip(patch_sizes, strides):
        # we currently only provide support for square patches, but this can be easily modified for future works
        keypoints, indices = calc_keypoint_centers_from_patches(config['feature_match'], patch_size, patch_size, stride, stride)
        all_keypoints.append(keypoints)
        all_indices.append(indices)

    matcher = PatchMatcher(config['feature_match']['matcher'], patch_sizes, strides, all_keypoints,
                           all_indices)

    scores, inlier_keypoints_one, inlier_keypoints_two = matcher.match(local_feats_one, local_feats_two)
    score = -apply_patch_weights(scores, len(patch_sizes), patch_weights)

    #print(f"Similarity score between the two images is: {score:.5f}. Larger scores indicate better matches.")
    return score

class CustomNetVLAD(torch.nn.Module):
        def __init__(self, encoder, net_vlad, num_pcs, netvlad_output_dim):
            super(CustomNetVLAD, self).__init__()
            self.encoder = encoder
            self.pool = net_vlad
            self.num_pcs = num_pcs
            self.netvlad_output_dim = netvlad_output_dim
            
            pca_conv = torch.nn.Conv2d(netvlad_output_dim, num_pcs, kernel_size=(1, 1), stride=1, padding=0)
            self.WPCA = torch.nn.Sequential(pca_conv, Flatten(), L2Norm(dim=-1))

        def forward(self, x):
            encoded = self.encoder(x)
            pooled = self.pool(encoded)
            return pooled

def main(cam=None, calibration=None):
    query_imgs = []
    all_scores = []
    all_frames = []

    encoder_dim, encoder = get_backend()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {'global_params' : {'patch_sizes' : "5", 
                                 'strides' : "1", 
                                 'num_pcs' : 512, 
                                 'num_clusters' : 64},
              'feature_match' : {'patchWeights2Use' : "1",
                                 'matcher' : 'spatialApproximator',
                                 'imageresizeH' : 480,
                                 'imageresizeW' : 640}}
    
    data_transform = transforms.Compose([ 
            transforms.Resize(224,antialias=True),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Instantiate the model
    num_clusters = config['global_params']['num_clusters']
    net_vlad = PatchNetVLAD(num_clusters=num_clusters, dim=encoder_dim)
    model = CustomNetVLAD(encoder, net_vlad, num_pcs=config['global_params']['num_pcs'], netvlad_output_dim=encoder_dim * num_clusters)
    
    # Load the state dictionary
    checkpoint = torch.load('pretrained_model/pitts_WPCA512.pth.tar', map_location='cpu')
    print(checkpoint['state_dict'].keys())
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    #load video database
    database_object = Database('./demo/ifi_rundt2_Trim2.mp4', model)
    database_lst = database_object.database
    print("Length of database: ", database_object.N)
    save_index = 0   

    #Processing rgb frame
    if cam is None:
        rgb_frame = Image.open('./data/random1.jpg')#.resize((config['feature_match']['imageresizeW'],config['feature_match']['imageresizeH']))
    

        best_match, max_score, score_lst, frames_lst = compare_to_database(rgb_frame, database_lst, config, device, data_transform, model)

        print("s",score_lst)
        print("f",len(frames_lst))

        query_imgs.append(frames_lst[-1])
        all_scores.append(score_lst[:-1])
        all_frames.append(frames_lst[:-1])

        best_matches = [frames_lst[-1], frames_lst[0], frames_lst[1], frames_lst[2]]
        worst_matches = [frames_lst[-1], frames_lst[-2], frames_lst[-3], frames_lst[-4]]
        shape_best = (1,len(best_matches))
        shape_worst = (1,len(worst_matches))
        size = (config['feature_match']['imageresizeW'],config['feature_match']['imageresizeH']) #frames_lst[0].size

        concat_imgs(best_matches,shape_best,size,score_lst[:3],save_index,"best")
        concat_imgs(worst_matches,shape_worst,size,score_lst[len(score_lst)-4:-1],save_index,"worst")
        save_index += 1

    else:
        window1_name = "window_1"
        cv2.namedWindow(window1_name, cv2.WINDOW_NORMAL)

        # MAIN LOOP
        while True:
            # Read next frame
            bgr_frame = cam.get_stereo_pair_rgb().left

            key = cv2.waitKey(1)

            # React to keyboard commands.
            if key == ord('s'):
                print("Saving location")


                #Processing rgb frame
                best_match, max_score, score_lst, frames_lst = compare_to_database(bgr_frame, database_lst, config, device, data_transform, model,"bgr")

                print("s",score_lst)
                print("f",len(frames_lst))

                query_imgs.append(frames_lst[-1])
                all_scores.append(score_lst[:-1])
                all_frames.append(frames_lst[:-1])

                best_matches = [frames_lst[-1], frames_lst[0], frames_lst[1], frames_lst[2]]
                worst_matches = [frames_lst[-1], frames_lst[-2], frames_lst[-3], frames_lst[-4]]
                shape_best = (1,len(best_matches))
                shape_worst = (1,len(worst_matches))
                size = frames_lst[-1].size #(config['feature_match']['imageresizeW'],config['feature_match']['imageresizeH'])

                concat_imgs(best_matches,shape_best,size,score_lst[:3],save_index,"best")
                concat_imgs(worst_matches,shape_worst,size,score_lst[len(score_lst)-4:-1],save_index,"worst")
                save_index += 1

                #print("Recognized this location: ", pos_match, "with score", max_score)

            elif key == ord('q'):
                print("Quit")
                break

            # Show matches
            cv2.imshow(window1_name, bgr_frame)

def compare_to_database(frame1, database, config, device, data_transform, model, type="rgb"):
    """
    Args:
        frame1 : RGB numpy image matrix 
        database : A list of BRG numpy image matrix

    Returns:
        similarity score : float
    """
    print("Compare")
    
    im_one_pil = frame1
    if type == "bgr":
        im_one_pil = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    img1_transformed = data_transform(im_one_pil).unsqueeze(0)

    max_score = -1
    best_match = None

    score_lst = [-1]
    frames_lst = [im_one_pil]

    for frame2 in database:
        im_two_pil = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))

        img2_transformed = data_transform(im_two_pil).unsqueeze(0)
        score = match_two(model, img1_transformed, img2_transformed, device, config)
        print(f"{score:.5f}")

        score_lst.append(score)
        frames_lst.append(im_two_pil)

        if score >= max_score:
            max_score = score
            best_match = im_two_pil
    
    #print(frames_lst)
    #score_lst, frames_lst = zip(*sorted(zip(score_lst, frames_lst),reverse=True))
    sorted_indices = sorted(range(len(score_lst)), key=lambda k: score_lst[k], reverse=True)
    sorted_score_lst = [score_lst[i] for i in sorted_indices]
    sorted_frames_lst = [frames_lst[i] for i in sorted_indices]

    return best_match, max_score, sorted_score_lst, sorted_frames_lst

def realsense():
    cam = RealSenseStereoCamera()
    calibration = StereoCalibration.from_realsense(cam)
    return cam, calibration
   
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "2":
        main()
    else:
        source = realsense()
        main(*source)
