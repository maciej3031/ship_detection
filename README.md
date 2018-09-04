## Solution for Airbus Ship Detection Competition
Link: https://www.kaggle.com/c/airbus-ship-detection

### Results:
- For now IoU ~0.72


### Solution Steps:
1. Get rid of corrupted images with size less than 50 kb.
2. Use VGG-19 pretrained on ImageNet dataset and train it to classify images if there is a ship in the image or not. Images are resized to 256x256 px.
3. Use pretrained VGG-19 as encoder, add some new layers as decoder and build U-Net model with it.
4. Get rid of images without ships, as dataset is imbalanced (>75% of images has no ships).
5. Freeze parameters of VGG-19 layers and train only freshly added decoder layers to segment ships. Images are still resized to 256x256 px. Loss function = 10 * Focal Loss + Dice Loss.
6. Unfreeze parameters of VGG-19 layers and train whole UNet network to segment ships. Images are still resized to 256x256 px. Loss function the same as in point 5.

### To try:
- [ ] As masks provided by organisators are rectangular. It's worth to try using some simple preprocessing to transform masks obtained from UNet to more "rectangular" shape

### Requirements:
- Linux
- NVIDIA GPU, Cuda, CudaNN

### How to use:
- To train own model just create virtualenv, install requirements.txt and use:

    `python fit.py -mn vgg19_unet_v1 -lr 0.001`

- To watch some evaluation results, open and run `ship_detection_simple_unet.ipynb`