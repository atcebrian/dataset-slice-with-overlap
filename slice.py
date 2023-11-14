import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import glob
import math


# get all image names
imnames0 = glob.glob('dataset/train/*.jpg')
imnames0 = [imname.replace('\\', '/') for imname in imnames0]
imnames1 = glob.glob('dataset/valid/*.jpg')
imnames1 = [imname.replace('\\', '/') for imname in imnames1]
imnames2 = glob.glob('dataset/test/*.jpg')
imnames2 = [imname.replace('\\', '/') for imname in imnames2]

imnames = [imnames0,imnames1,imnames2]

# specify path for a new tiled dataset
newpath0 = 'tiled/train/'
newpath1 = 'tiled/valid/'
newpath2 = 'tiled/test/'

newpaths = [newpath0,newpath1,newpath2]
falsepath = 'tiled/false/'

# specify slice size (slices are square)
slice_size = 640

# specify overlap size
overlap = 128


# tile all images in a loop
for idx, ims in enumerate(imnames):
    for imname in ims:
        im = Image.open(imname)
        imr = np.array(im, dtype=np.uint8)
        height = imr.shape[0]
        width = imr.shape[1]
        labname = imname.replace('.jpg', '.txt')
        print(labname)
        # Define column names for the data

        labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
        print(labels)
        
        # we need to rescale coordinates from 0-1 to real image height and width
        labels[['x1', 'w']] = labels[['x1', 'w']] * width
        labels[['y1', 'h']] = labels[['y1', 'h']] * height
        
        boxes = []
        
        # convert bounding boxes to shapely polygons. We need to invert Y and find polygon vertices from center points
        for row in labels.iterrows():
            x1 = row[1]['x1'] - row[1]['w']/2
            y1 = (height - row[1]['y1']) - row[1]['h']/2
            x2 = row[1]['x1'] + row[1]['w']/2
            y2 = (height - row[1]['y1']) + row[1]['h']/2

            boxes.append((int(row[1]['class']), Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))
        
        counter = 0
        print('Image:', imname)
        # create tiles and find intersection with bounding boxes for each tile
        print(height)
        for i in range(math.ceil(height/(slice_size-overlap))):
            for j in range((math.ceil(width/(slice_size-overlap)))):
                x1 = min((j*(slice_size-overlap)), (width-slice_size))
                y1 = max(height - (i*(slice_size-overlap)),slice_size)
                x2 = x1+(slice_size-1)
                y2 = y1-(slice_size+1)

                pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                imsaved = False
                slice_labels = []

                for box in boxes:
                    if pol.intersects(box[1]):
                        inter = pol.intersection(box[1])        
                        
                        if not imsaved:
                            h = min((i*(slice_size-overlap)),(height-slice_size))
                            sliced = imr[h:h+slice_size, x1:x1+slice_size]
                            sliced_im = Image.fromarray(sliced)
                            filename = imname.split('/')[-1]
                            slice_path = newpaths[idx] + filename.replace('.jpg', f'_{i}_{j}.jpg')
                            
                            slice_labels_path = newpaths[idx] + filename.replace('.jpg', f'_{i}_{j}.txt')
                            sliced_im.save(slice_path)
                            imsaved = True                    
                        
                        # get the smallest polygon (with sides parallel to the coordinate axes) that contains the intersection
                        new_box = inter.envelope 
                        
                        # get central point for the new bounding box 
                        centre = new_box.centroid
                        
                        # get coordinates of polygon vertices
                        x, y = new_box.exterior.coords.xy
                        
                        # get bounding box width and height normalized to slice size
                        new_width = (max(x) - min(x)) / slice_size
                        new_height = (max(y) - min(y)) / slice_size
                        
                        # we have to normalize central x and invert y for yolo format
                        new_x = (centre.coords.xy[0][0] - x1) / slice_size
                        new_y = (y1 - centre.coords.xy[1][0]) / slice_size
                        
                        counter += 1

                        slice_labels.append([box[0], new_x, new_y, new_width, new_height])
                
                # save txt with labels for the current tile
                if len(slice_labels) > 0:
                    slice_df = pd.DataFrame(slice_labels, columns=['class', 'x1', 'y1', 'w', 'h'])
                    print(slice_df)
                    slice_df.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')
                
                # if there are no bounding boxes intersect current tile, save this tile to a separate folder 
                if not imsaved:
                    h = min((i*(slice_size-overlap)),(height-slice_size))
                    sliced = imr[h:h+slice_size, x1:x1+slice_size]
                    sliced_im = Image.fromarray(sliced)
                    filename = imname.split('/')[2]
                    slice_path = falsepath + filename.replace('.jpg', f'_{i}_{j}.jpg')                

                    sliced_im.save(slice_path)
                    print('Slice without boxes saved')
                    imsaved = True