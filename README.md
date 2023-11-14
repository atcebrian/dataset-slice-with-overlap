# dataset-slice-with-overlap
Based on [tile_yolo.py](https://gist.github.com/slanj/33cfde3d05cc0a6c81e71c5527f6d401)
but with added functionalities:
1. Images are tiled til the end so no parts of the image are lost.
2. Overlap between images is added to avoid some objects being represented only partially.

USE:
Download slice.py
Add your dataset with the following structure:

slice
├── dataset
│     ├── train
│     │   ├── img1.jpg
│     │   ├── img1.txt
│     │   ├── img2.jpg
│     │   ├── img2.txt
│     │   ├── img3.jpg
│     │   ├── img3.txt
│     │   └── ...
│     ├── test
│     │   ├── img1.jpg
│     │   ├── img1.txt
│     │   ├── img2.jpg
│     │   ├── img2.txt
│     │   ├── img3.jpg
│     │   ├── img3.txt
│     │   └── ...
│     └── valid
│         ├── img1.jpg
│         ├── img1.txt
│         ├── img2.jpg
│         ├── img2.txt
│         ├── img3.jpg
│         ├── img3.txt
│         └── ...
├── slice.py
└── README.md

Slices the images of your anotated dataset into smaller tiles to make your training easier.
