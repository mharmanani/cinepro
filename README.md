# Cinepro: Robust Training of Foundation Models for Cancer Detection in Prostate Ultrasound Cineloops
---

### 🚧 This README is under construction. For now, only the section on hyperparameter tuning is available. 

### Hyperparameter tuning
##### Learning rate, optimizer, batch size
We experimented with several hyperparameter configurations. Namely, we tried **learning rates** of 1e-4 and 1e-5 for both the encoder and decoder respectively. We ultimately selected 1e-5 as our best learning rate. 

For the optimizer, we tried both Adam and AdamW, with little difference. We kept AdamW. Due to computational limitations, we were restricted to using a batch size of at most 2. 

##### Loss functions
One of the central aspects of the work is the design of a tailored loss function for US-based PCa detection. We tried a wide variety of loss functions before settling on $\mathcal{L}_\text{iMSE}$. 

One of the earlier loss functions we tried is $\mathcal{L}_\text{OBP}$, which is the same as masked cross-entropy, but with an added term aiming to minimize the activations of the model outside the needle region. We found that this model yielded worse validation performance overall.

We tried two other variations of an involvement-aware loss function, namely $\mathcal{L}_ \text{iCE}$ and $\mathcal{L}_ \text{iMAE}$. We found that $\mathcal{L}_ \text{iCE}$ did not differ in validation performance when compared to $\mathcal{L}_ \text{MaskCE}$, and $\mathcal{L}_ \text{iMAE}$ was correlated with very poor performance, in some cases failing to exceed 60\% AUROC on the validation set.

##### Augmentations
We experimented with various types of data augmentations, including gaussian noise, speckle noise, salt and pepper noise, rotations, translations, deformations, and pixel and line cuts.

We found that deformations hurt performance overall, likely because they corrupted the needle region more than needed, thus weakening the association between those pixels and the label. 

We also experimented with the choice of frames to use for our main method, as well as our baseline. We tried using the first frame, the last frame, the average of all 200 frames, and a random frame each time. For our baseline, we obtained the highest validation performance using the last frame in the series. However, it is worth mentioning the differences between using the first frame and the last frame were slight, and it is possible that using any other frame could work. 

For Cinepro, we tried the following combinations of frames:
* First frame as $X_w$, average of 199 subsequent frames as $X_s$
* Last frame as $X_w$, average of 199 previous frames as $X_s$
* First frame as $X_w$, average of 100 subsequent frames as $X_s$
* Last frame as $X_w$, average of 100 previous frames as $X_s$
* First frame as $X_w$, average of 50 subsequent frames as $X_s$
* Last frame as $X_w$, average of 50 previous frames as $X_s$
* First frame as $X_w$, rand. subsequent frame each time for $X_s$
* Last frame as $X_w$, rand. previous frame each time for $X_s$
* Random frame each time for both $X_w$ and $X_s$
