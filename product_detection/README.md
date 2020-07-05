# Product Detection

## Competition description

At Shopee, we always strive to ensure the correct listing and categorization of products. For example due to the recent pandemic situation, face masks become extremely popular for both buyers and sellers, everyday we need to categorize and update a huge number of masks items. A robust product detection system will significantly improve the listing and categorization efficiency. But in the industrial field the data is always much more complicated and there exists mis-labelled images, complex background images and low resolution images, etc. The noisy and imbalanced data and multiple categories make this problem still challenging in the modern computer vision field.

In this competition, a multiple image classification model needs to be built. There are ~100k images within 42 different categories, including essential medical tools like masks, protective suits and thermometers, home & living products like air-conditioner and fashion products like T-shirts, rings, etc. For the data security purpose the category names will be desensitized. The evaluation metrics is top-1 accuracy.


## How to download my pretrained models

```
bash download.sh
```

## How to reproduce my results

```
bash test.sh <data_dir>
```

## How to train my code
```
python3 train.py <data_dir>
```
You can choose the model by change the line 53 or checkout the [torchvision website](https://pytorch.org/docs/stable/torchvision/models.html) to select the model you preferred.

NOTE: <data_dir> should have train, test, test.csv, train.csv

## Ensemble

We know that if our models are more diverse, we can improve our scores more effectively in ensemble (weighted voting). You can execute `ensemble.py` in the `results` directory to get our highest score on the kaggle leaderboard.

## Leaderboard (Kaggle)

Team name: Team SOTA

Public Leaderboard: 0.82845 

Private Leaderboard: 0.82839 (31/823)

Competition website: https://www.kaggle.com/c/shopee-product-detection-student/


## Contact information

For help or issues using our code, please contact Sung-Ping Chang (`joshspchang@gmail.com`).

