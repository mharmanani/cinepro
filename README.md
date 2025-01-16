# Cinepro: Robust Training of Foundation Models for Cancer Detection in Prostate Ultrasound Cineloops
---

[!cinepro](./img/cinepro_v3.png)

## Abstract
Prostate cancer (PCa) detection using deep learning (DL) models has shown potential for enhancing real-time guidance during biopsies. However, prostate ultrasound images lack pixel-level cancer annotations, introducing label noise. Current approaches often focus on limited regions of interest (ROIs), disregarding anatomical context necessary for accurate diagnosis. Foundation models can overcome this limitation by analyzing entire images to capture global spatial relationships; however, they still encounter challenges stemming from the weak labels associated with coarse pathology annotations in ultrasound data. We introduce Cinepro, a novel framework that strengthens foundation models' ability to localize PCa in ultrasound cineloops. 
Cinepro adapts robust training by integrating the proportion of cancer tissue reported by pathology in a biopsy core into its loss function to address label noise, providing a more nuanced supervision. Additionally, it leverages temporal data across multiple frames to apply robust augmentations, enhancing the model’s ability to learn stable cancer-related features. 
Cinepro demonstrates superior performance on a multi-center prostate ultrasound dataset, achieving an AUROC of 77.1\% and a balanced accuracy of 83.8\%, surpassing current benchmarks. These findings underscore Cinepro's promise in advancing foundation models for weakly labeled ultrasound data.

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

### Computational Efficiency
| Method |  Architecture   | Params. | Inference Time  (s)   | GPU Memory Req. (GB) |
|--------|-----|-----|----------------|--|
| iLR     | InceptionTime| 7,168 | TBD  | TBD       |
| UNet    | UNet   | 4,125       | 0.41 | 0.900 |
| SAM     | ViT-B  | 93,729,252  | 1.11 | 11.02 |
| MedSAM  | ViT-B  | 93,729,252  | 1.11 | 11.02 |
| Cinepro | ViT-B  | 93,729,252  | 1.11 | 11.02 |