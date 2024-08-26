import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_correspondences(file_path: str): 
    with open(file_path, 'r') as file:
        lines = file.readlines()
    correspondences = []

    for line in lines:
        values = list(map(float, line.split()))
        coords = np.array(values).reshape(2, 2)
        correspondences.append(coords)

    correspondences = np.array(correspondences)
    return correspondences

def visualize_correspondences(img1_path: str, img2_path: str, correspondences):
    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Draw correspondences
    for correspondence in correspondences:
        (x1, y1), (x2, y2) = correspondence
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        # Draw point on image 1
        cv2.circle(img1, pt1, 5, (0, 0, 255), -1)
        # Draw point on image 2
        cv2.circle(img2, pt2, 5, (0, 0, 255), -1)
        # Draw line between correspondences
        cv2.arrowedLine(img1, pt1, pt2, (255, 0, 0), 2)
        cv2.arrowedLine(img2, pt2, pt1, (255, 0, 0), 2)

    # Concatenate images horizontally for visualization
    vis = np.concatenate((img1, img2), axis=1)

    # Display the visualization inline
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()