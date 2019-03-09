import numbers
import random

from PIL import Image, ImageFilter
import numpy as np
import cv2

from torchvision.transforms import functional as F
from torchvision.transforms import transforms as tvtransforms

class CustomCenterCrop(tvtransforms.CenterCrop):
    def __call__(self, img_dict):
        """
        Args:
            img_dict (PIL Images dictionary with keys 'rgb', 'ir', 'depth'): Image to be cropped.
        Returns:
            PIL Images dictionary: Cropped images.
        """
        keys = ['rgb', 'ir', 'depth']
        for key in keys:
            img_dict[key] = F.center_crop(img_dict[key], self.size)
            
        return img_dict

class CustomCrop(object):
    def __init__(self, size, crop_index):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
        
        self.crop_index = crop_index
    
    def __call__(self, img_dict):
        keys = ['rgb', 'ir', 'depth']
        for k in keys:
            img = img_dict[k]
            w, h = img.size
            crop_h, crop_w = self.size
            if crop_w > w or crop_h > h:
                raise ValueError("Requested crop size {} is bigger than input size {}".format(self.size,
                                                                                              (h, w)))
            if self.crop_index == 0:
                img_dict[k] = F.center_crop(img, (crop_h, crop_w))
            elif self.crop_index == 1:
                img_dict[k] = img.crop((0, 0, crop_w, crop_h))
            elif self.crop_index == 2:
                img_dict[k] = img.crop((w - crop_w, 0, w, crop_h))
            elif self.crop_index == 3:
                img_dict[k] = img.crop((0, h - crop_h, crop_w, h))
            elif self.crop_index == 4:
                img_dict[k] = img.crop((w - crop_w, h - crop_h, w, h))
            else:
                raise ValueError("Requested crop index is not in range(5)")
        return img_dict
        
    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, crop_index={1})'.format(self.size, self.crop_index)

class CustomRotate(object):
    def __init__(self, rotate_angle=0):
        self.rotate_angle = rotate_angle
        
    def __call__(self, img_dict):
        keys = ['rgb', 'ir', 'depth']
        for k in keys:
            img_dict[k] = img_dict[k].rotate(self.rotate_angle)

        return img_dict
        
    def __repr__(self):
        return self.__class__.__name__ + '(rotate_angle={0})'.format(self.rotate_angle)

    
class CustomToTensor(tvtransforms.ToTensor):
    def __call__(self, pic_dict):
        keys = ['rgb', 'ir', 'depth']
        for key in keys:
            pic_dict[key] = F.to_tensor(pic_dict[key])   
        return pic_dict
    

class CustomNormalize(tvtransforms.Normalize):
    def __call__(self, tensor_dict):
        keys = ['rgb', 'ir', 'depth']
        for key in keys:
            tensor_dict[key] = F.normalize(tensor_dict[key], self.mean, self.std)
        return tensor_dict
    
class CustomResize(tvtransforms.Resize):
    def __call__(self, img_dict):
        keys = ['rgb', 'ir', 'depth']
        for key in keys:
            img_dict[key] = F.resize(img_dict[key], self.size, self.interpolation)
        return img_dict
    
class CustomPad(tvtransforms.Pad):
    def __call__(self, img_dict):
        keys = ['rgb', 'ir', 'depth']
        for key in keys:
            img_dict[key] = F.pad(img_dict[key], self.padding, self.fill, self.padding_mode)
        return img_dict
    
class CustomRandomCrop(tvtransforms.RandomCrop):
    def __call__(self, img_dict):
        keys = ['rgb', 'ir', 'depth']
        
        if self.padding > 0:
            for key in keys:
                img_dict[key] = F.pad(img_dict[key], 
                                      self.padding, 
                                      self.fill, 
                                      self.padding_mode)

        # pad the width if needed
        for key in keys:
            if self.pad_if_needed and img_dict[key].size[0] < self.size[1]:
                img_dict[key] = F.pad(img_dict[key], 
                                      (int((1 + self.size[1] - img_dict[key].size[0]) / 2), 0))
            # pad the height if needed
            if self.pad_if_needed and img_dict.size[1] < self.size[0]:
                img_dict[key] = F.pad(img_dict[key], 
                                      (0, int((1 + self.size[0] - img_dict[key].size[1]) / 2)))
        
        
        i, j, h, w = self.get_params(img_dict[keys[0]], self.size)
        for key in keys:
            img_dict[key] = F.crop(img_dict[key], i, j, h, w)
        
        return img_dict

class CustomRandomHorizontalFlip(tvtransforms.RandomHorizontalFlip):
    def __call__(self, img_dict):
        keys = ['rgb', 'ir', 'depth']
        if random.random() < self.p:
            for key in keys:
                img_dict[key] = F.hflip(img_dict[key])
        return img_dict

class CustomRandomResizedCrop(tvtransforms.RandomResizedCrop):
    def __call__(self, img_dict):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        keys = ['rgb', 'ir', 'depth']
        
        i, j, h, w = self.get_params(img_dict[keys[0]], self.scale, self.ratio)
        for key in keys:
            img_dict[key] = F.resized_crop(img_dict[key], i, j, h, w, 
                                           self.size, self.interpolation)
        return img_dict

class CustomColorJitter(tvtransforms.ColorJitter):
    def __call__(self, img_dict):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Color jittered image.
        """
        keys = ['rgb']
        
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        for key in keys:
            img_dict[key] = transform(img_dict[key])
        return img_dict
    
class CustomRandomRotation(tvtransforms.RandomRotation):
    def __call__(self, img_dict):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """
        keys = ['rgb', 'ir', 'depth']
        
        angle = self.get_params(self.degrees)
        for key in keys:
            img_dict[key] = F.rotate(img_dict[key], 
                                     angle, 
                                     self.resample, 
                                     self.expand, 
                                     self.center)
        return img_dict

class CustomRandomAffine(tvtransforms.RandomAffine):
    def __call__(self, img_dict):
        """
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: Affine transformed image.
        """
        keys = ['rgb','ir','depth']
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_dict[keys[0]].size)
        for key in keys:
            img_dict[key] = F.affine(img_dict[key], *ret, resample=self.resample, fillcolor=self.fillcolor)
        return img_dict

class CustomRandomGrayscale(tvtransforms.RandomGrayscale):
    def __call__(self, img_dict):
        if random.random() < self.p:
            keys = ['rgb']
            for key in keys:
                num_output_channels = 1 if img_dict[key].mode == 'L' else 3
                img_dict[key] = F.to_grayscale(img_dict[key], num_output_channels=num_output_channels)
        return img_dict

class AllElementsResize(object):
    def __init__(self, target_elem_idx=0, interpolation=2):
        self.target_index = target_elem_idx
        self.interpolation = interpolation
        
    def __call__(self, img_dict):
        keys = ['rgb', 'ir', 'depth']
        target_size = img_dict[keys[self.target_index]].size
        for key in keys:
            img_dict[key] = F.resize(img_dict[key], target_size, 
                                     interpolation=self.interpolation)
        return img_dict
    def __repr__(self):
        format_string = self.__class__.__name__ + '(target_elem_idx={0}, interpolation={1})'.format(self.target_index,
                                                                                                    self.interpolation)
        return format_string

class Facepart1Crop(object):
    def __init__(self, w_ratio, h_ratio):
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio
        
    def __call__(self, img_dict):
        keys = ['rgb', 'ir', 'depth']
        for key in keys:
            i, j, h, w = self.get_params(img_dict[key].size)
            img_dict[key] = F.crop(img_dict[key], i, j, h, w)
        
        return img_dict
    
    def get_params(self, img_size):
        new_w = round(self.w_ratio * img_size[0])
        new_h = round(self.h_ratio * img_size[1])
        return (0,0,new_h, new_w)

    def __repr__(self):
        return self.__class__.__name__ + '(w_ratio={0}, h_ratio={1})'.format(self.w_ratio, self.h_ratio)

class DepthNormalize(object):
    def __init__(self, norm_type='minmax'):
        self.norm_type = norm_type
        
    def __call__(self, img_dict):
        keys = ['depth']
        for key in keys:
            img_v = np.array(img_dict['depth'].convert('L'))

            min_v = img_v[img_v > 0].min()
            max_v = img_v.max()
            if max_v - min_v > 0:
                res = (img_v - min_v)/(max_v - min_v)
                res[img_v == 0] = 0
                img_dict[key] = Image.fromarray((res * 255).astype(np.uint8)).convert('RGB') 
        return img_dict
    
    def __repr__(self):
        return self.__class__.__name__ + '(norm_type={0})'.format(self.norm_type)

class MaskIntersection(object):
    def __init__(self, same_label=False, orgl_kernel_radius=5, augm_kernel_radius=10):
        self.orgl_kernel_radius = orgl_kernel_radius
        a, b = orgl_kernel_radius, orgl_kernel_radius
        n, r = 2 * orgl_kernel_radius + 1, orgl_kernel_radius
        y,x = np.ogrid[-a:n-a, -b:n-b]
        self.kernel_orgl = (x*x + y*y <= r*r).astype(np.uint8)
        
        self.augm_kernel_radius = augm_kernel_radius
        a, b = augm_kernel_radius, augm_kernel_radius
        n, r = 2 * augm_kernel_radius + 1, augm_kernel_radius
        y,x = np.ogrid[-a:n-a, -b:n-b]
        self.kernel_augm = (x*x + y*y <= r*r).astype(np.uint8)
        
        self.same_label = same_label
    
    def __call__(self, img_dict):
        data_get_func = img_dict['meta']['get_item_func']
        curr_idx = img_dict['meta']['idx']
        max_idx = img_dict['meta']['max_idx']
        
        other_idx = np.random.randint(0, max_idx)
        data4augm = data_get_func(other_idx)
        while (curr_idx == other_idx) or (self.same_label and data4augm['label'] != img_dict['label']):
            other_idx = np.random.randint(0, max_idx)
            data4augm = data_get_func(other_idx)
            
        depth4augm = data4augm['depth'].resize(img_dict['depth'].size)
        mask4augm = np.array(depth4augm.convert('L')) > 0
        mask4augm = cv2.morphologyEx(mask4augm.astype(np.uint8), 
                                     cv2.MORPH_OPEN, 
                                     self.kernel_orgl)
        
        mask = np.array(img_dict['depth'].convert('L')) > 0
        mask = cv2.morphologyEx(mask.astype(np.uint8), 
                                cv2.MORPH_OPEN, 
                                self.kernel_augm)
        mask = (mask == mask4augm) & (mask)
        mask = np.repeat(np.expand_dims(mask, 2), 3, axis=2)
        
        keys = ['depth', 'ir']
        for key in keys:
            np_img = np.array(img_dict[key]) * mask
            img_dict[key] = Image.fromarray(np_img)
        return img_dict
        
    def __repr__(self):
        return self.__class__.__name__ + '(same_label={0},'.format(self.same_label) + \
    'orgl_kernel_radius={0}, augm_kernel_radius={1})'.format(self.orgl_kernel_radius,
                                                             self.augm_kernel_radius)

class MergeItems(object):
    def __init__(self, same_label=False, p=0.5):
        self.p = p
        self.same_label = same_label
    
    def __call__(self, img_dict):
        
        if np.random.rand() < self.p:
            data_get_func = img_dict['meta']['get_item_func']
            curr_idx = img_dict['meta']['idx']
            max_idx = img_dict['meta']['max_idx']

            other_idx = np.random.randint(0, max_idx)
            data4augm = data_get_func(other_idx)
            while (curr_idx == other_idx) or (self.same_label and data4augm['label'] != img_dict['label']):
                other_idx = np.random.randint(0, max_idx)
                data4augm = data_get_func(other_idx)

            alpha = np.random.rand()

            keys = ['rgb', 'depth', 'ir']
            for key in keys:
                img_dict[key] = Image.blend(data4augm[key].resize(img_dict[key].size),
                                            img_dict[key],
                                            alpha=alpha)
            if not self.same_label:
                img_dict['label'] = alpha * img_dict['label'] + (1 - alpha) * data4augm['label']
    
        return img_dict
        
    def __repr__(self):
        return self.__class__.__name__ + '(same_label={0}, p={1})'.format(self.same_label, self.p)

class LabelSmoothing(object):
    def __init__(self, eps=0.1, p=0.5):
        self.p = p
        self.eps = eps
    
    def __call__(self, img_dict):
        if np.random.rand() < self.p:
            img_dict['label'] = np.abs(img_dict['label'] - self.eps)
        
        return img_dict
        
    def __repr__(self):
        return self.__class__.__name__ + '(eps={0}, p={1})'.format(self.eps, self.p)

class CustomGaussianBlur(object):
    """ Apply Gaussian blur to image with probability 0.5
    """
    def __init__(self,max_kernel_radius=3, p=0.5):
        self.max_radius = max_kernel_radius
        self.p = p

    def __call__(self, img_dict):
        keys = ['rgb', 'depth', 'ir']
        radius = random.uniform(0, self.max_radius)
        if random.random() < self.p:
            for k in keys:
                img_dict[k] = img_dict[k].filter(ImageFilter.GaussianBlur(radius))
            return img_dict
        else:
            return img_dict
    
    def __repr__(self):
        return self.__class__.__name__ + '(max_kernel_radius={0}, p={1})'.format(self.max_radius, self.p)

class CustomCutout(object):
    def __init__(self, n_holes, min_size, max_size):
        self.n_holes = n_holes
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img_dict):
        keys = ['rgb', 'depth', 'ir']
        h = img_dict[keys[0]].size[0]
        w = img_dict[keys[0]].size[1]

        mask = np.ones((h,w,3), np.uint8)
        for n in range(self.n_holes):
            length = np.random.randint(self.min_size, self.max_size)
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2, :] = 0.

        for key in keys:
            img_arr = np.array(img_dict[key])
            img = img_arr * mask
            img_dict[key] = Image.fromarray(img)

        return img_dict

    def __repr__(self):
        return self.__class__.__name__ + '(n_holes={0}, min_size={1}, max_size={2})'.format(self.n_holes, self.min_size, self.max_size)


