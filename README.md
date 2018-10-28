## Solution for Airbus Ship Detection Competition
Link: https://www.kaggle.com/c/airbus-ship-detection
Data: https://www.kaggle.com/c/airbus-ship-detection/data

### Results:
- For now IoU score on Public LB = 0.674

### Solution Steps:
1. Get rid of corrupted images with size less than 50 kb.
2. Use ResNet34 pretrained on ImageNet dataset.
3. Use pretrained ResNet34 as encoder, add some new layers as decoder and build U-Net model with it.
4. Get rid of images without ships, as dataset is imbalanced (>75% of images has no ships).
5. Train UNet on images resized to 384x384 px with simple data augmentation (rotations, shearings, flips and zooms). Loss function = 10 * Focal Loss + Dice Loss. Use 5 epochs of initial training to train only decoder.
6. Add 30k images withoud ships, to have still balanced dataset.
7. Train UNet once again on images resized to 384x384 px with simple data augmentation (rotations, shearings, flips and zooms). Loss function = 10 * Focal Loss + Dice Loss.
8. (Optional) Perform erosion and dilation on each obtained result mask.


### To try:
- [ ] As masks provided by organisators are rectangular. It's worth to try using some simple preprocessing to transform masks obtained from UNet to more "rectangular" shape
- [x] Try more decoder filters
- [x] Try InceptionV3
- [x] Try use BatchNorm in decoder


### Requirements:
- Linux
- NVIDIA GPU, Cuda, CudaNN


### How to use:
- To train own model just create virtualenv, install requirements.txt and use:

    `python fit.py -mn resnet34_unet_v1 -lr 0.0001`

- To watch some evaluation results, open and run `ship_detection_simple_unet.ipynb`

### Sample results

![alt text](https://github.com/maciej3031/ship_detection/blob/master/examples/example1.png)
