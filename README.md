# project_distributedQNN
Course PHSX600-801 project - Topic: "Distributed Quantum Machine Learning"

## Introduction
This project is to implement a distributed quantum machine learning algorithm using PennyLane and PyTorch. The algorithm is based on the paper [Accelerating variational quantum algorithms with multiple quantum processors](https://arxiv.org/abs/2106.12819) by Yuxuan Du, Yang Qian, and Dacheng Tao. We modify the algorithm to use PennyLane and change the quantum circuit architecture based on phase encoding and data re-uploading techniques.

## Instructions to run the code
The classical vs VQC algorithm is implemented in the file `VQA_mnist.py`. 
The distributed VQC algorithm is implemented in the file `DVQA_mnist`.
