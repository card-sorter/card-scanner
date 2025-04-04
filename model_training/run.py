import cv2
from PIL import Image
import numpy as np

# The perspective transform code is form jsol/card-scanner. We wil write our own implemementation once everything works.

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("../runs/segment/train9/weights/best.pt")

# Run inference on 'bus.jpg'
results = model(["https://media.discordapp.net/attachments/693635180954517534/1218336179288801290/IMG_5591.jpg?ex=67f1103c&is=67efbebc&hm=322bef4a693478466a8e002880ff371f4966de5aa051bff5b21376a9e4c01cc2&=&format=webp&width=1141&height=856"])  # results list

def reorder_points(points):
    # Calculate centroids
    centroids = np.mean(points, axis=0)

    # Sort points based on distance from centroids
    points_sorted = sorted(points, key=lambda x: np.arctan2(x[0][1] - centroids[0][1], x[0][0] - centroids[0][0]))

    return np.array(points_sorted)

def get_card_dimensions(corners):
    # Calculate the width and height of the card
    width = np.linalg.norm(corners[0] - corners[1])
    height = np.linalg.norm(corners[1] - corners[2])
    return width, height

# Visualize the results
for i, r in enumerate(results):


    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")

    if r.masks:
        for c in r.masks.xy:
            contour = c.astype(np.int32)
            contour = contour.reshape(-1, 1, 2)
            epsilon = 0.1 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            print(approx)
            if len(approx) == 4:
                approx = reorder_points(approx)
                width, height = get_card_dimensions(approx)
                dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(approx.astype(np.float32), dst)

                warped = cv2.warpPerspective(r.orig_img, M, (int(width), int(height)))

                # Rotate the warped image to ensure portrait orientation
                if width > height:
                    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

                # Resize the warped image to fill the canvas without maintaining aspect ratio
                warped_stretched = cv2.resize(warped, (716, 1000))

                cv2.imshow("img", warped_stretched)
                cv2.waitKey(0)


