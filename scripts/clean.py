# some functions to clean the dataset
import os
import os.path as osp

def twodigit(addr):
    # make sure that all the viewIndex are two digits
    for img_name in os.listdir(addr):
        exten_idx = img_name.find('.')
        if img_name[exten_idx-2] == '_':
            img_name_new = img_name[:exten_idx-1] + '0' + img_name[exten_idx-1:]
            img_dir = osp.join(addr, img_name)
            img_dir_new = osp.join(addr, img_name_new)
            os.rename(img_dir, img_dir_new)

if __name__ == '__main__':
    twodigit('/home/max/projects/VLN-UCSD/data/v1/scans/1LXtFkjw3qL/sampled_semantic_images')