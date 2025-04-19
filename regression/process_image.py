import cv2
import numpy as np
import pandas as pd
import os
from common import HISTORY, IMAGE_PATH, CSV_PATH


def noise(points, noise_scale=0.01):
    points = points.astype(np.float32)
    noise = np.random.uniform(-noise_scale, noise_scale, points.shape)
    points += noise
    return points


columns = [f'Point(t-{i})' for i in range(HISTORY - 1, -1, -1)]
columns.append("Point(t+1)")
df = pd.DataFrame(columns=columns)

for filename in os.listdir(IMAGE_PATH):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image = cv2.imread(os.path.join(
            IMAGE_PATH, filename), cv2.IMREAD_GRAYSCALE)

        _, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = 255 - binary

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            # Assume the largest contour is the curve
            contour = max(contours, key=cv2.contourArea)
            points = contour.squeeze()
            """
            We squeeze the points because contour returns an array like this:
            array([
                    [[10, 15]],
                    [[11, 16]],
                    [[12, 17]],
                    ...])
            shape = (N, 1, 2)
            It's a 3D array with annoying double brackets, so we squeeze it into:
            array([
                [10, 15],
                [11, 16],
                [12, 17],
                ...])
            shape = (N, 2)
            Now we get a nice array of coordinates :)
            """
            # points = noise(points, noise_scale=0.01)

            for i in range(HISTORY, len(points) - 1):
                row_points = points[i-HISTORY:i+1]
                row = {
                    f'Point(t-{HISTORY-j-1})': (row_points[j][0], row_points[j][1]) for j in range(HISTORY)}
                row["Point(t+1)"] = (row_points[-1][0], row_points[-1][1])
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

df.to_csv(CSV_PATH, index=False)
