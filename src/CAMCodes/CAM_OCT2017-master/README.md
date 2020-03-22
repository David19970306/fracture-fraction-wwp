# Using Class Activation Mapping (CAM) for classification and visualization OCT2017 dataset
## This repository is forcusing on using CAM to classify OCT2017
- CAM Paper: http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf
- Network Architecture: 
  + MobileNetV2 + Global Average Pooling (GAP) + Softmax Layer
  + The new architecture has fewer parameters (beacause of adding GAP)
- Comparison classification ability with original MobileNetV2:

![](https://github.com/HoSyTuyen/CAM_OCT2017/blob/master/classification_compare.png)

- Visualization:

![](https://github.com/HoSyTuyen/CAM_OCT2017/blob/master/visualization.png)

- For more details, please go into **CAM.ipynb**
- I also provide my output model **epoch=05_accuracy=0.9252_val_accuracy=0.9819.h5**
