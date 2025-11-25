import numpy as np
import cv2
import math

def cylindrical_projection(image, center, angle_degrees, curvature):

    angle = np.radians(angle_degrees)
    
    height, width = image.shape[:2]
    
    projected = np.zeros_like(image)
    
    focal_length = width / (2 * curvature) if curvature > 0 else width
    
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    cx, cy = center
    
    for y in range(height):
        for x in range(width):
            x_rot = (x - cx) * cos_angle - (y - cy) * sin_angle + cx
            y_rot = (x - cx) * sin_angle + (y - cy) * cos_angle + cy
            
            theta = (x_rot - cx) / focal_length
            h = (y_rot - cy) / focal_length
            
            x_src = focal_length * np.tan(theta) + cx
            y_src = focal_length * h / np.cos(theta) + cy
            
            if 0 <= x_src < width and 0 <= y_src < height:
                x1, y1 = int(x_src), int(y_src)
                x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)
                
                wx = x_src - x1
                wy = y_src - y1
                
                if len(image.shape) == 3:  
                    for channel in range(3):
                        top = (1 - wx) * image[y1, x1, channel] + wx * image[y1, x2, channel]
                        bottom = (1 - wx) * image[y2, x1, channel] + wx * image[y2, x2, channel]
                        projected[y, x, channel] = (1 - wy) * top + wy * bottom
                else:  # grayscale image
                    top = (1 - wx) * image[y1, x1] + wx * image[y1, x2]
                    bottom = (1 - wx) * image[y2, x1] + wx * image[y2, x2]
                    projected[y, x] = (1 - wy) * top + wy * bottom
    
    return projected

def cylindrical_projection_fast(image, center, angle_degrees, curvature):
  
    height, width = image.shape[:2]
    
    focal_length = width / (2 * curvature) if curvature > 0 else width
    
    angle = np.radians(angle_degrees)
    
    x_dst, y_dst = np.meshgrid(np.arange(width), np.arange(height))
    
    cx, cy = center
    x_rot = (x_dst - cx) * np.cos(angle) - (y_dst - cy) * np.sin(angle) + cx
    y_rot = (x_dst - cx) * np.sin(angle) + (y_dst - cy) * np.cos(angle) + cy
    
    theta = (x_rot - cx) / focal_length
    h = (y_rot - cy) / focal_length
    
    x_src = focal_length * np.tan(theta) + cx
    y_src = focal_length * h / np.cos(theta) + cy
    
    map_x = x_src.astype(np.float32)
    map_y = y_src.astype(np.float32)
    
    projected = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    
    return projected

if __name__ == "__main__":
    # image loader
    image = cv2.imread('tau.jpg')
    
    # parameters
    center = (image.shape[1] // 2, image.shape[0] // 2)  # Image center
    angle_degrees = 0  
    curvature = 0.5  
    
    result = cylindrical_projection_fast(image, center, angle_degrees, curvature)
    
    cv2.imshow('Original', image)
    cv2.imshow('Cylindrical Projection', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()