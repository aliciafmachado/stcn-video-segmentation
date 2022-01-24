# Unsupervised video segmentation with STCNs and Mask R-CNN

Object Recognition projection in video segmentation.

## Results

Obtained results are contained in the folder results.

## Organization of the files

### code/STCN

This is the code from the STCN repository (I forked myself). I slightly modified some files on it, but all the rest is the official implementation of the paper "Rethinking Space-Time Networks with Improved Memory Coverage for Efficient Video Object Segmentation", that you can find in https://arxiv.org/pdf/2106.05210.pdf.

### code/something-something

Here you may find specific code / scripts for dealing with the Something-Something dataset.

Unfortunately, if you want to use it entirely, you will need to download it yourself in https://developer.qualcomm.com/software/ai-datasets/something-something.

## How to use this repository?

There is a colab notebook inside the notebooks folder that goes through the main scripts and reproduces the results obtained.

In order to calculate relevant metrics, you might refer to https://github.com/davisvideochallenge/davis2017-evaluation.