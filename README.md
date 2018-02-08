# LinkNet_tensorflow

This repository is tensorFlow implementation of  LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation (https://arxiv.org/abs/1707.03718) and the official Torch implementation (https://github.com/e-lab/LinkNet) and the pytorch implementation by PavlosMelissinos (https://github.com/e-lab/pytorch-linknet), trained on the Cityscapes dataset (https://www.cityscapes-dataset.com/).

- The video of results (logs/results/cityscapes_stuttgart_02_pred.avi):
- [![demo video with results](https://github.com/liuhengli/LinkNet_tensorflow/blob/9347a58539662fdfa59d64c40b047f17ae21b50a/logs/results/stuttgart_02_000000_005100_leftImg8bit_pred.png)](https://github.com/liuhengli/LinkNet_tensorflow/blob/9347a58539662fdfa59d64c40b047f17ae21b50a/logs/results/cityscapes_stuttgart_02_pred.avi)

## Dependencies:

+ Python 3.5 or greater
+ Tensorflow or greater
+ [OpenCV3](https://opencv.org/)
****

## Files/folders and their usage:
Linknet_model.py:  
- bulid the Linknet model.
*****

load_cityscapes_data.py:  
- preprocess dataset and generate train/val batch data.
that all Cityscapes training (validation) image directories have been placed in data_dir/cityscapes/leftImg8bit/train (data_dir/cityscapes/leftImg8bit/val) and that all corresponding ground truth directories have been placed in data_dir/cityscapes/gtFine/train (data_dir/cityscapes/gtFine/val).
*****

train_linknet.py:  
- train linknet class
*****

demo.py:  
- run a model checkpoint on all frames in a Cityscapes demo sequence directory and creates a video of the result.

****

### License

This software is released under a creative commons license which allows for personal and research use only.
For a commercial license please contact the authors.
You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/
