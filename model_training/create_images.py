import numpy as np
import os, random, cv2
import math
from multiprocessing import Pool
import tqdm

images = os.listdir("C:/large_dataset/background_training")
cards = os.listdir("./images")
output = "C:/large_dataset/train/"

# from https://stackoverflow.com/a/75391691
def chain_affine_transformation_mats(M0, M1):
    """
    Chaining affine transformations given by M0 and M1 matrices.
    M0 - 2x3 matrix applying the first affine transformation (e.g rotation).
    M1 - 2x3 matrix applying the second affine transformation (e.g translation).
    The method returns M - 2x3 matrix that chains the two transformations M0 and M1 (e.g rotation then translation in a single matrix).
    """
    T0 = np.vstack((M0, np.array([0, 0, 1])))  # Add row [0, 0, 1] to the bottom of M0 ([0, 0, 1] applies last row of eye matrix), T0 is 3x3 matrix.
    T1 = np.vstack((M1, np.array([0, 0, 1])))  # Add row [0, 0, 1] to the bottom of M1.
    T = T1 @ T0  # Chain transformations T0 and T1 using matrix multiplication.
    M = T[0:2, :]  # Remove the last row from T (the last row of affine transformations is always [0, 0, 1] and OpenCV conversion is omitting the last row).
    return M

def point_position(point, M, x_dim, y_dim):
    return [(M[0][0] * point[0] + M[0][1] * point[1] + M[0][2])/x_dim, (M[1][0] * point[0] + M[1][1] * point[1] + M[1][2])/y_dim]


def overlay_with_transform(background, overlay_img, text):
    if overlay_img.shape[2] == 3:
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2BGRA)
        overlay_img[:, :, 3] = 255

    h_background, w_background = background.shape[:2]
    h_overlay, w_overlay = overlay_img.shape[:2]

    if max(w_background//1.4, h_background) == h_background:
        max_w = w_background//1.4
        max_h = w_background
    else:
        max_h = h_background
        max_w = int(max_h//1.4)

    scale = random.uniform(0.3, 0.4)

    max_w = int(max_w * scale)
    max_h = int(max_h * scale)

    tl = (random.randint(0, max_w//4), random.randint(0, max_h//4))
    tr = (random.randint(max_w*3//4, max_w), random.randint(0, max_h//4))
    bl = (random.randint(0, max_w//4), random.randint(max_h*3//4, max_h))
    br = (random.randint(max_w*3//4, max_w), random.randint(max_h*3//4, max_h))
    pts1 = np.float32([[0, 0], [w_overlay, 0], [0, h_overlay], [w_overlay, h_overlay]])
    pts2 = np.float32([tl, tr, bl, br])

    extents = [[min(tl[0], bl[0]), min(tl[1], tr[1])],
               [max(tr[0], br[0]), max(bl[1], br[1])]]
    v_len = extents[1][1] - extents[0][1]
    h_len = extents[1][0] - extents[0][0]
    center = [(extents[0][0]+extents[1][0])/2, (extents[0][1]+extents[1][1])/2]

    radius = int(math.sqrt(v_len*v_len + h_len*h_len)/2)
    angle = random.uniform(0, 360)
    x_loc = random.randint(radius, w_background-radius)-center[0]
    y_loc = random.randint(radius, h_background-radius)-center[1]


    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(overlay_img, M, (w_background, h_background))

    trans_M = np.float32([[1, 0, x_loc],[0 ,1, y_loc]])
    rot_M = cv2.getRotationMatrix2D(center, angle, 1)
    M = chain_affine_transformation_mats(rot_M, trans_M)

    dst = cv2.warpAffine(dst, M, (w_background, h_background))

    alpha = dst[:, :, 3] / 255.0

    alpha_3channel = np.dstack([alpha, alpha, alpha])
    result = background.copy()
    for c in range(3):
        result[:, :, c] = (1 - alpha_3channel[:, :, c]) * background[:, :, c] + \
                          alpha_3channel[:, :, c] * dst[:, :, c]

    points = point_position(tl, M, w_background, h_background) + point_position(tr, M, w_background, h_background) + point_position(br, M, w_background, h_background) + point_position(bl, M, w_background, h_background)

    text.write("0 ")
    for p in points:
        text.write("%.3f " % p)
    text.write('\n')
    return result


def loader(vals):
    background = cv2.imread(vals[0])
    text_path = output + vals[1] + ".txt"
    with open(text_path, "w") as text:
        for _ in range(random.randint(2, 4)):
            card = "./images/"+random.choice(cards)
            overlay_img = cv2.imread(card, cv2.IMREAD_UNCHANGED)
            try:
                background = overlay_with_transform(background, overlay_img, text)
            except:
                break
    cv2.imwrite(output + vals[1] + ".jpg", background)


if __name__ == "__main__":
    idx = 0
    todo = []
    random.shuffle(images)
    for i in images:
        todo.append((
            "C:/large_dataset/background_training/"+i,
            str(idx))
        )
        idx+=1
        if idx > 70000:
            break
    p = Pool(6)
    for _ in tqdm.tqdm(p.imap_unordered(loader, todo), total=len(todo)):
        pass