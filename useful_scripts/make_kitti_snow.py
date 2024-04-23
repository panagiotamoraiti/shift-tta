import os
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage import map_coordinates
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import ctypes

class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)
        
def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]

def snow(x, severity=1):
    c = [(0.1,0.2,1,0.6,8,3,0.95),
         (0.1,0.2,1,0.5,10,4,0.9),
         (0.15,0.3,1.75,0.55,10,4,0.9),
         (0.25,0.3,2.25,0.6,12,6,0.85),
         (0.3,0.3,1.25,0.65,14,12,0.8)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    #print("XSHAPE ", x.shape[0])
    #print("XSHAPE ", x.shape)
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome
    #snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = Image.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))
    snow_layer = cv2.imdecode(np.frombuffer(snow_layer.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    m = np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(x.shape[0], x.shape[1], 1) * 1.5 + 0.5)
    x = c[6] * x + (1 - c[6]) * m
    
    #print("XSHAPE ", x.shape)
    #print("snow_layerSHAPE ", snow_layer.shape)
    r = np.rot90(snow_layer, k=2)
    #print("rot90SHAPE ", r.shape)
    ss = x + snow_layer + r
    s = np.clip(ss, 0, 1)
    snow_layer_resized_pil = Image.fromarray((s * 255).astype(np.uint8))
    return snow_layer_resized_pil

# Load KITTI dataset
# kitti_dataset_dir = "/home/panagiota/work/tta/shift-detection-tta/data/kitti/data_object/training/image_2"
kitti_dataset_dir = "/home/panagiota/work/tta/shift-detection-tta/data/kitti/raw_data/2011_09_26/2011_09_26_drive_0056_sync/image_02/data"

# out_dir = "/home/panagiota/work/tta/shift-detection-tta/data/kitti/"
out_dir = "/home/panagiota/work/tta/shift-detection-tta/data/kitti/weather_datasets/weather_kitti/raw_data/2011_09_26/2011_09_26_drive_0056_sync/image_02/data"

kitti_images = os.listdir(kitti_dataset_dir)

# Create a new directory to save corrupted images
corrupted_images_dir = os.path.join(out_dir, "snow")
os.makedirs(corrupted_images_dir, exist_ok=True)

print("Generating Images...")

for i in range(1, 6):
    # Create a new directory to save corrupted images
    c_images_dir = os.path.join(corrupted_images_dir, "severity_" + str(i))
    os.makedirs(c_images_dir, exist_ok=True)

    # Apply snow corruption to KITTI images and save as PNG
    for img_name in kitti_images:
        img_path = os.path.join(kitti_dataset_dir, img_name)
        img = Image.open(img_path)
        corrupted_img = snow(img, severity=i)  # You can adjust the severity as needed
        save_path = os.path.join(c_images_dir, img_name)
        corrupted_img.save(save_path, format="PNG")

