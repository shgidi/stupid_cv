
## TLDR
1 line script for training an image classification model (including data acquisition)

## Intro

All technologists are automation enthusiasts and - lets face it - script kiddies. And since according to Eric Schmidt computer vision is a [solved](https://twitter.com/math_rachel/status/1188863155612471306) problem, I’ve decided to assemble a small script that will be kind of auto solver for computer vision.

### How does it work

Open images v4 allows to easilly and programmitacly download many images according classes (unlike imagenet, there may be numerous objects in every image)

Frameworks such as keras and fast ai allow “easy” training. Why not combine?

Introducing: **stupid_cv**.

Use it ass follows:
1. `python stupid_cv.py --data_path <some_dir> --classes Apple,Banana,Orange`
2. Wait for it..
3. Find the model in `<some_dir>/models
4. Profit!


## Install

To install, clone this repo, and install the requirements. You'd better have GPU on your machine.

## Usage

First, get the open images data frames:

Dataset type is one of train, validation, test
https://storage.googleapis.com/openimages/2018_04/{dataset_type}/{dataset_type}-annotations-bbox.csv
### List all classes
`python list_classes.py`

### Run all pipeline
`python stupid_cv.py`

Arguments:

* `data_root` - the folder where the images should will be downloaded to, and the models will be saved. Open Images data frames should be placed in this folder
* `data_type` - select the data type form open images, as stated above
* `classes` - select classes to download and train a model on, from open images 600 classes
* `cut_image` - should the training script "cut" the relevant objects from the images

Run:

## Todo: 
* Add requirements
* Add list_classes.py
* Check functionallity of cut_images
* Add detection (and segmentation?) functionallity
* Add efficient tracking
* Add serving/deploy