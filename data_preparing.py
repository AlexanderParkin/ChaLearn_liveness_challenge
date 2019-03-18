from PIL import Image
from tqdm import tqdm
import pandas as pd

def images_resizing(df):
    '''
    df - DataFrame with rgb, depth, ir columns
    '''
    for rgb_path, depth_path, ir_path in tqdm(zip(df.rgb.values, df.depth.values, df.ir.values),
                                         total=len(df)):
        rgb_img = Image.open(pp + rgb_path)
        depth_img = Image.open(pp + depth_path)
        ir_img = Image.open(pp + ir_path)

        ir_img = ir_img.resize(rgb_img.size,resample=3)
        ir_img.save(pp + ir_path.replace('.jpg', '_resized.jpg'))
        depth_img = depth_img.resize(rgb_img.size,resample=3)
        depth_img.save(pp + depth_path.replace('.jpg', '_resized.jpg'))
