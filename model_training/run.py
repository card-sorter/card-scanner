import cv2
import numpy as np
from PIL import Image
from fontTools.misc.bezierTools import epsilon

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("../runs/segment/train7/weights/best.pt")
img = cv2.imread('PXL_20250318_150522703.jpg', cv2.IMREAD_COLOR_RGB)
# Run inference on 'bus.jpg'
results = model([img])  # results list

# Visualize the results
for i, r in enumerate(results):




    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    contours = r.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
    for contour in contours:
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(img, [contours], 0, (0, 0, 255), 4)
    # Show results to screen (in supported environments)
    cv2.imshow("image", img)
    cv2.waitKey(0)

    # Save results to disk
    r.save(filename=f"results{i}.jpg")