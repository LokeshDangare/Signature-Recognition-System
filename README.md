# Signature-Recognition-System

#Problem statement
The task of signature recognition involves building a system that can automatically recognize an individual's signature from a given set of signature images. The system should be able to distinguish between genuine signatures and forged ones, and should work robustly even in the presence of noise and variations in the signature style.

#Solution Proposed
The goal of signature recognition is to develop an accurate and reliable system that can automatically recognize an individual's signature and distinguish it from forged signatures. This can have practical applications in areas such as document verification, fraud detection, and biometric identification.

#Dataset Used
CEDAR Signature is a database of off-line signatures for signature verification. Each of 55 individuals contributed 24 signatures thereby creating 1,320 genuine signatures. Some were asked to forge three other writers’ signatures, eight times per subject, thus creating 1,320 forgeries. Each signature was scanned at 300 dpi gray-scale and binarized using a gray-scale histogram. Salt pepper noise removal and slant normalization were two steps involved in image preprocessing. The database has 24 genuines and 24 forgeries available for each writer.

#Model Used
ResNet-34 is a popular deep convolutional neural network architecture that was introduced in the paper "Deep Residual Learning for Image Recognition" by Kaiming He et al. in 2016.

ResNet-34 consists of 34 layers, including 33 convolutional layers and 1 fully connected layer. The architecture of ResNet-34 is based on the residual learning framework, which allows the network to be deeper while maintaining good performance.
