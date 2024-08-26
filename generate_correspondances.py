
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import os
from modules.utils import load_correspondences, visualize_correspondences
from scipy.spatial.distance import cdist

# FEATURE DETECTORS
def orb_detect(img, num_k, show=False):
    orb = cv2.ORB_create(nfeatures=num_k)
    keypoints, descriptors = orb.detectAndCompute(img, None)

    if show:
        img_with_keypoints = img.copy()
        for kp in keypoints:
            x, y = kp.pt
            cv2.circle(img_with_keypoints, (int(x), int(y)), 10, (0, 0, 255), -1)  
        img_with_keypoints_rgb = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)
        plt.imshow(img_with_keypoints_rgb)
        plt.axis('off')
        plt.show()
    return keypoints, descriptors

def sift_detect(img, num_k, show=False):
    sift = cv2.SIFT_create(nfeatures=num_k)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    
    if show:
        img_with_keypoints = img.copy()
        for kp in keypoints:
            x, y = kp.pt
            cv2.circle(img_with_keypoints, (int(x), int(y)), 10, (0, 0, 255), -1)  

        img_with_keypoints_rgb = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)
        plt.imshow(img_with_keypoints_rgb)
        plt.axis('off')  # Turn off axis
        plt.show()
    return keypoints, descriptors

# MATCH KEYPOINTS
def get_keypoints_and_descriptors(img, method, num_kp):
    if method == 'SIFT':
        return sift_detect(img, num_kp)
    elif method == 'ORB':
        return orb_detect(img, num_kp)
    else:
        raise ValueError("method not available")
    
def get_matching_keypoints(keypoints1, descriptors1, keypoints2, descriptors2, lowe_ratio=0.7, lowe_min_dist=150, method='SIFT'):
    if method == 'ORB':
        metric = 'hamming'
    else:
        metric = 'euclidean'
        
    matched_keypoints = []
    distances12 = cdist(descriptors1, descriptors2, metric=metric)
    distances21 = cdist(descriptors2, descriptors1, metric=metric)

    for kp_idx in range(len(descriptors1)):
        sorted_indices12 = np.argsort(distances12[kp_idx])
        match_kp_idx = sorted_indices12[0]

        # Calculate the Lowe's ratio
        best_distance = distances12[kp_idx, match_kp_idx]
        second_best_distance = distances12[kp_idx, sorted_indices12[1]]
        ratio = best_distance / second_best_distance

        # Apply Lowe's tests
        if best_distance < lowe_min_dist and ratio < lowe_ratio:
            # Check backward and forward correspondance
            sorted_indices21 = np.argsort(distances21[match_kp_idx])
            back_match_kp = sorted_indices21[0]

            if kp_idx == back_match_kp:
                # Append the corresponding keypoints to matched_keypoints
                matched_keypoints.append((keypoints1[kp_idx], keypoints2[match_kp_idx]))
    
    return matched_keypoints

# GENETARE IMG CORRESPONDANCES
def generate_img_correspondances(img_paths:list, method:str, num_keypoints:int, output_folder:str, lowe_ratio=0.7, lowe_min_dist=150):
    
    #Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx1, img_path1 in tqdm(enumerate(img_paths), desc="Processing images", total=len(img_paths)):
        # The images are ordered in the folder, images far away in the folder are very different, 
        # almost no correspondance should be found. Therefore we discard all these images.
        for idx2 in range(idx1+1, min(idx1+6, len(img_paths))):
            
            #Get keypoints
            img_path2 = img_paths[idx2]
            img1 = cv2.imread(img_path1)
            img2 = cv2.imread(img_path2)
            keypoints1, descriptors1 = get_keypoints_and_descriptors(img1, method, num_keypoints)
            keypoints2, descriptors2 = get_keypoints_and_descriptors(img2, method, num_keypoints)
            
            #Get correct matches
            match_kp = get_matching_keypoints(keypoints1, descriptors1, keypoints2, descriptors2, method=method, lowe_ratio=lowe_ratio, lowe_min_dist=lowe_min_dist)
            
            #Generate the output file with the correspondences
            if len(match_kp) > 4:
                filename = f"{output_folder}/{img_path1.split('/')[-1].replace('.jpg', '')}_{img_path2.split('/')[-1].replace('.jpg', '')}.txt"
                with open(filename, 'w') as f:
                    for kp1, kp2 in match_kp:
                        f.write(f"{kp1.pt[0]} {kp1.pt[1]} {kp2.pt[0]} {kp2.pt[1]}\n")

# CLEAN THE CORRESPONDANCES
def clean_correspondances(correspondences, max_length=200, sim_threshold=0.9):
    vectors = correspondences[:, 1] - correspondences[:, 0]
    #Idea: Correspondances with big movement indicate far aways objects or bad correspondances. We can delete those.
    lengths = np.linalg.norm(vectors, axis=1)
    
    #Second idea: If a correspondance moves very different compared to its neighbours, it is probably a bad correspondance. (Smooth flow hypothesis)
    #Get top 3 closest points, with KNN, if its similarity is high
    first_points = correspondences[:,0,:]
    distances = cdist(first_points, first_points)
    sorted_indices = np.argsort(distances, axis=1)

    del_idxs = []
    for i, nearest_indices in enumerate(sorted_indices):
        dif_neighbours = 0
        for j, neighbor_idx in enumerate(nearest_indices[1:4]):
            sim = np.dot(vectors[i], vectors[neighbor_idx])/(lengths[i]*lengths[neighbor_idx])
            if sim < sim_threshold:
                dif_neighbours += 1
        #If two of its three neigbours have different direction, its probably an incorrect match.
        if dif_neighbours > 1:
            del_idxs.append(i)

    filtered_correspondences = np.delete(correspondences, del_idxs, axis=0)
    #Idea: Correspondances with big movement indicate far aways objects or bad correspondances. We can delete those.
    vectors = filtered_correspondences[:, 1] - filtered_correspondences[:, 0]
    lengths = np.linalg.norm(vectors, axis=1)
    filtered_correspondences = filtered_correspondences[lengths <= max_length]
    return filtered_correspondences

#SHOULD GO TO UTILS, NOT YET IN UTILS TO AVOID MERGE PROBLEMS
def get_all_imgs_paths_in_folder(folder_path):
    imgs = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            imgs.append(os.path.join(folder_path, file))
    return sorted(imgs)

def save_correspondences(file_path: str, correspondences):
    with open(file_path, 'w') as file:
        for coords in correspondences:
            line = ' '.join(map(str, coords.flatten())) + '\n'
            file.write(line)
    
