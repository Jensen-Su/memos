
Dataset: car1000

FC-CNN[M] Best Accuracy: 92.27%, Speed: 55 fps (GTX TITAN 12G)
Training method:

Input: resize to $224\times 224$

Lr 0.001 + momentum 0.9 + 90% training set: 0~20 Epochs
Loss 0.01, Accuracy 91.47% (Epoch 20)
Accuracy on **Validation Set (10% training set)** 95.9%

Lr 0.0005 + momentum 0.9 + 90% training set: 21~30 Epochs
Loss 0.003, Accuracy 92.27% (Epoch 23) 
Accuracy on **Validation Set (10% training set)** 96.3%

Lr 0.0005 + momentum 0.9 + 100% training set: 31~34 Epochs
Lr 0.0001 + momentum 0.9 + 100% training set: 35~50 Epochs
Loss 0.0004, Accuracy 91.87% (Epoch 38) 

B-CNN[M,M] Best Accuracy: 96.xx%, Speed: 25 fps (GTX 1080 8G)
Training method:
Input: resize to $448\times 448$
lr0.001 + 100% training set for 0~10 Epochs
lr0.0005 + 100% training set for 11~15 Epochs
lr0.0001 + 100% training set for 16~20 Epochs

WB-CNN[M,M] Best Accuracy: 9x.xx%, Speed: 22 fps (GTX TITAN 12G)
Training method: 

lr0.005 + 100% training set for 0~5 Epochs
lr0.001 + 100% training set for 6~10 Epochs
lr0.0005 + 100% training set for 11~15 Epochs
l40.0001 + 100% training set for 15~20 Epochs
