On tools you can find the data correction utility

dataExtractionScript.py creates X_train.npy, Y_train.npy, X_test.npy, and Y_test.npy, the datasets used for training on the deceiver network.

The training process is as follows:
1) trainDeceiverUnet.ipynb
2) trainShapeUnet.ipynb
3) trainSRUnet.ipynb

This work is based on:
``Ravishankar, H., Venkataramani, R., Thiruvenkadam, S., Sudhakar, P., & Vaidya, V. (2017, September). Learning and incorporating shape models for semantic segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 203-211). Springer, Cham.``

pixelwise segmentation autoencoder + shape convolution auto encoder
 
Advantage of explicitly modeling shape: 
1) Generalization to appearance devia-tions from the training data
2) Data augmentation

Key contributions:
1) Learning a non-linear shape model and projection of arbitrary masks to theshape manifold space. We also discuss two novel data augmentation strategies to implement a shape convolution auto encoder.
2) Incorporating the shape model explicitly in a FCN formulation through anovel loss function that penalizes deviation of the predicted segmentation maskfrom a learnt shape model.


Shape regularization inside FCN and not as a post-processing 
Generic formulation that can be appended to other geometry or topology priors
two data augmentation strategies

Loss = mse(d(seg,seg_enc_dec), d(seg_enc,gnd_enc), d(seg,gnd)) lamda 0.5

cascade of two FCNs:
1) Unet (segmentation)
2) Unet - concatenation connections (shape regularization)

Segmentation
10 layers 5enc-5dec
relu activation
batch normalization regularization

shape regularization network pre-training:
    inaccurate shapes as input and ground truth masks as the output.
    two data segmentation strategies for creating these incomplete shapes:
    a) Random corruption of shapes: random points + erode
    b) Intermediate U-Net predictions: sample the U-Net predictions before  convergence

