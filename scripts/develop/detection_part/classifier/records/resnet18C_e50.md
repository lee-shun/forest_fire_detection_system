# Model structure: Resnet18  
Total number of images: 818  
Total training images: 655  
Total valid_images: 163  
Computation device: cuda  

11,536,195 total parameters.  
11,536,195 training parameters.

## 2021-12-03
When transform the brightness with 0.5, the traning is unacceptable, the accuracy for 'smoke' is around $10%$.    
Therefore the brightness is changed with 0.2. Then the learning is fine to separately reach $70%$, where:  
|Epochs| Acc fire | Acc normal | Acc smoke| Total train loss| Total val loss|  
|------| ---------|------------|----------|-----------------|---------------|   
|17| 96.42857142857143| 88.63636363636364| 71.42857142857143| 80.458| 84.663|  
|18| 91.07142857142857| 79.54545454545455| 80.95238095238095| 79.084| 84.049|  
|19| 71.42857142857143| 88.63636363636364| 73.01587301587301| 79.389| 76.687|  
|49| 96.42857142857143| 84.09090909090911| 36.50793650793651| 83.664| 69.939|  
|50| 80.35714285714286| 93.18181818181819| 22.22222222222222| 86.565| 61.350|  

epochs = 20 is almost enough,. At e19, the loss of validation is climbing and unstable. And the detection of smoke is getting worse.  

It is considered to use lr = 1e-5 to slow the learning.
