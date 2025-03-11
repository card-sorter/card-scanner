import numpy as np
import os, random, cv2
import math


def overlay_with_transform(background_path, overlay_path, output_path):
    background = cv2.imread(background_path)
    overlay_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

    if overlay_img.shape[2] == 3:
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2BGRA)
        overlay_img[:, :, 3] = 255

    h_background, w_background = background.shape[:2]
    h_overlay, w_overlay = overlay_img.shape[:2]

    if max(w_background, h_background) == h_background:
        max_w = w_background
        max_h = int(max_w*1.4)
    else:
        max_h = h_background
        max_w = int(max_h//1.4)

    scale = random.uniform(0.4, 0.8)

    max_w = int(max_w * scale)
    max_h = int(max_h * scale)

    loc_x = random.randint(0, w_background-max_w)
    loc_y = random.randint(0, h_background-max_h)

    tl = [random.randint(loc_x, max_w//3+loc_x), random.randint(loc_y, max_h//3+loc_y)]
    tr = [random.randint(max_w*2//3+loc_x, max_w+loc_x), random.randint(loc_y, max_h//3+loc_y)]
    bl = [random.randint(0+loc_x, max_w//3+loc_x), random.randint(max_h*2//3+loc_y, max_h+loc_y)]
    br = [random.randint(max_w*2//3+loc_x, max_w+loc_x), random.randint(max_h*2//3+loc_y, max_h+loc_y)]
    pts1 = np.float32([[0, 0], [w_overlay, 0], [0, h_overlay], [w_overlay, h_overlay]])
    pts2 = np.float32([tl, tr, bl, br])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(overlay_img, M, dsize=(w_background, h_background))

    alpha = dst[:, :, 3] / 255.0

    alpha_3channel = np.dstack([alpha, alpha, alpha])
    result = background.copy()
    for c in range(3):
        result[:, :, c] = (1 - alpha_3channel[:, :, c]) * background[:, :, c] + \
                          alpha_3channel[:, :, c] * dst[:, :, c]


    cv2.imwrite(output_path, result)


# Example usage
if __name__ == "__main__":
    for i in range(20):
        overlay_with_transform(
            "./training_data/background/"+random.choice(os.listdir("./training_data/background")),
            "./images/"+random.choice(os.listdir("./images")),
            "./training_data/gen/%s.jpg" % i
        )