### Perliminary

This repository stores some of the code used for synthetic tumors and for the training and evaluation of segmentation models with associated results.
In general, if not specifically labeled, during all model training, random patches of 96 × 96 × 96 were cropped from 3D image volumes during training. All models are trained for 4,000 epochs, and the base learning rate is 0.0002. The batch size is two per GPU. We adopt the linear warmup strategy and the cosine annealing learning rate schedule. For inference, we use the sliding window strategy by setting the overlapping area ratio to 0.75.

### Related Papers

**Generative Enhancement for 3D Medical Images**  *Zhu, Lingting and Codella, Noel and Chen, Dongdong and Jin, Zhenchao and Yuan, Lu and Yu, Lequan* arXiv preprint arXiv:2403.12852 | 19 Mar 2024 [paper](https://arxiv.org/abs/2403.12852)

**Towards Generalizable Tumor Synthesis** *Qi Chen, Xiaoxi Chen, Haorui Song, Zhiwei Xiong, Alan Yuille, Chen Wei, Zongwei Zhou* CVPR | 29 Feb 2024 [paper](https://arxiv.org/pdf/2402.19470.pdf)

**SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation** *Zhaohu Xing, Tian Ye, Yijun Yang, Guang Liu, Lei Zhu* arXiv preprint arXiv:2401.13560 | 25 Feb 2024 [paper](https://arxiv.org/abs/2401.13560)

**Label-Free Liver Tumor Segmentation** *Qixin Hu, Yixiong Chen, Junfei Xiao, Shuwen Sun, Jieneng Chen, Alan Yuille, Zongwei Zhou* CVPR | 27 March 2023 [paper](https://arxiv.org/abs/2303.14869) 

### AI model trained by synthetic tumors(Downloaded,Control experiment)

|                |         UNET         | Swin-UNETR-Base  (pretrain) | Swin-UNETR-Base (no.pretrain) | Swin-UNETR-Small (no.pretrain) | Swin-UNETR-Tiny (no.pretrain) |
| :------------: | :------------------: | :-------------------------: | :---------------------------: | :----------------------------: | :---------------------------: |
| **Liver Dice** |     **0.96093**      |         **0.96307**         |          **0.96441**          |          **0.96388**           |          **0.95996**          |
| **Liver Nsd**  |     **0.87736**      |         **0.88440**         |          **0.89196**          |          **0.88591**           |          **0.87198**          |
| **Tumor Dice** | **0.57766** - 0.5981 |    **0.55722** - 0.5594     |     **0.53344** - 0.5594      |      **0.54794** - 0.5637      |     **0.52479** - 0.5510      |
| **Tumor Nsd**  | **0.61974** - 0.6128 |    **0.60019** - 0.5820     |     **0.57556** - 0.5820      |      **0.57665** - 0.5824      |     **0.54078** - 0.5561      |

### AI model trained by real tumors for comparison(Downloaded,Control experiment)

|                |         UNET         |   Swin-UNETR-Base  (pretrain)    |  Swin-UNETR-Base (no.pretrain)  | Swin-UNETR-Small (no.pretrain) | Swin-UNETR-Tiny (no.pretrain) |
| :------------: | :------------------: | :------------------------------: | :-----------------------------: | :----------------------------: | :---------------------------: |
| **Liver Dice** |     **0.95290**      |           **0.96424**            |           **0.96737**           |          **0.96246**           |          **0.96115**          |
| **Liver Nsd**  |     **0.85500**      |           **0.88934**            |           **0.90179**           |          **0.88162**           |          **0.87801**          |
| **Tumor Dice** | **0.53054** - 0.5763 | **0.55744** - 0.5902 - (0.5592)† | **0.59590** - 0.5902- (0.5592)† |      **0.55345** - 0.5849      |     **0.54777** - 0.5592      |
| **Tumor Nsd**  | **0.55023** - 0.5810 |       **0.59046** - 0.6082       |      **0.61910** - 0.6082       |      **0.59696** - 0.5986      |     **0.56614** - 0.5655      |

### AI model trained by synthetic tumors(Self)

#### Traditional model

|                |         UNET         | UNET(Global) |   nnUNet    | nnUNet(Global) | Swin-UNETR-Base  (pretrain) | Swin-UNETR-Base (no.pretrain) | Swin-UNETR-Small (no.pretrain) | Swin-UNETR-Tiny (no.pretrain) |
| :------------: | :------------------: | :----------: | :---------: | :------------: | :-------------------------: | :---------------------------: | :----------------------------: | :---------------------------: |
| **Liver Dice** |     **0.96450**      | **0.95408**  | **0.96340** |  **0.95520**   |         **0.96476**         |          **0.96409**          |          **0.96401**           |          **0.96276**          |
| **Liver Nsd**  |     **0.89170**      | **0.87262**  | **0.88493** |  **0.87021**   |         **0.89265**         |          **0.88995**          |          **0.88962**           |          **0.88353**          |
| **Tumor Dice** | **0.57500** - 0.5981 | **0.60817**  | **0.54407** |  **0.59870**   |    **0.57025 - 0.5594**     |     **0.55158 - 0.5594**      |      **0.54079 - 0.5637**      |     **0.53372 - 0.5510**      |
| **Tumor Nsd**  | **0.62084** - 0.6128 | **0.63772**  | **0.57940** |  **0.62433**   |    **0.60829** - 0.5820     |      **0.60278** - 0.582      |      **0.58507** - 0.5824      |     **0.56287** - 0.5561      |

#### Mamba Model

|                | SegMambaV0  | SegMambaV0(Global) | SegMambaV1  | SegMambaV1(Global) |
| :------------: | :---------: | :----------------: | :---------: | :----------------: |
| **Liver Dice** | **0.95851** |    **0.94532**     | **0.96300** |    **0.95292**     |
| **Liver Nsd**  | **0.86826** |    **0.84355**     | **0.88415** |    **0.86724**     |
| **Tumor Dice** | **0.51011** |    **0.54694**     | **0.55682** |    **0.58262**     |
| **Tumor Nsd**  | **0.54203** |    **0.56692**     | **0.58413** |    **0.59475**     |

#### Difftumor(Semi-Supervised)

|                | UNet(Chen‘s Val) | UNet(Hu's Val) | UNet(Global) |
| :------------: | :--------------: | :------------: | :----------: |
| **Liver Dice** |   **0.94906**    |  **0.96005**   | **0.94981**  |
| **Liver Nsd**  |   **0.88588**    |  **0.91371**   | **0.88923**  |
| **Tumor Dice** |   **0.69001**    |  **0.68217**   | **0.75031**  |
| **Tumor Nsd**  |   **0.69970**    |  **0.74257**   | **0.77798**  |

|                | nnUNet(Chen‘s Val) | nnUNet(Hu's Val) | nnUNet(Global) |
| :------------: | :----------------: | :--------------: | :------------: |
| **Liver Dice** |                    |                  |                |
| **Liver Nsd**  |                    |                  |                |
| **Tumor Dice** |                    |                  |                |
| **Tumor Nsd**  |                    |                  |                |

|                | SwinUNETR(Chen‘s Val) | SwinUNETR(Hu's Val) | SwinUNETR(Global) |
| :------------: | :-------------------: | :-----------------: | :---------------: |
| **Liver Dice** |                       |                     |                   |
| **Liver Nsd**  |                       |                     |                   |
| **Tumor Dice** |                       |                     |                   |
| **Tumor Nsd**  |                       |                     |                   |

#### Manual method modification

|                | UNET(GMM) | UNET(GMM,Global) | UNET(+5% Sphere) | UNET(+10% Sphere) | UNET(+12% Sphere) | UNET(+15% Sphere) | UNET(+17% Sphere) | UNET(+25% Sphere) | Zhang et al. |
| :------------: | :-------: | :--------------: | :--------------: | :---------------: | :---------------: | :---------------: | :---------------: | :---------------: | :----------: |
| **Liver Dice** |           |                  |   **0.96086**    |    **0.96161**    |    **0.96114**    |    **0.96185**    |    **0.96171**    |    **0.95622**    | **0.95761**  |
| **Liver Nsd**  |           |                  |   **0.88020**    |    **0.88341**    |    **0.87753**    |    **0.87860**    |    **0.87605**    |    **0.85615**    | **0.86665**  |
| **Tumor Dice** |           |                  |   **0.55767**    |    **0.56752**    |    **0.53391**    |    **0.58081**    |    **0.54269**    |    **0.45940**    | **0.51179**  |
| **Tumor Nsd**  |           |                  |   **0.60392**    |    **0.61096**    |    **0.56607**    |    **0.62556**    |    **0.55893**    |    **0.45606**    | **0.54367**  |

### AI model trained by real tumors for comparison(Self)

|                |         UNET         |   Swin-UNETR-Base  (pretrain)   | Swin-UNETR-Base (no.pretrain)  | Swin-UNETR-Small (no.pretrain) | Swin-UNETR-Tiny (no.pretrain) |
| :------------: | :------------------: | :-----------------------------: | :----------------------------: | :----------------------------: | :---------------------------: |
| **Liver Dice** |     **0.95963**      |           **0.96585**           |          **0.96414**           |          **0.96415**           |          **0.96204**          |
| **Liver Nsd**  |     **0.87500**      |           **0.89222**           |          **0.88773**           |          **0.88408**           |          **0.87628**          |
| **Tumor Dice** | **0.56542** - 0.5763 | **0.58100**- 0.5902 - (0.5592)† | **0.54147**- 0.5902- (0.5592)† |      **0.56391**- 0.5849       |      **0.53814**- 0.5592      |
| **Tumor Nsd**  | **0.58887** - 0.5810 |       **0.61395**- 0.6082       |      **0.55647** - 0.6082      |      **0.59873**- 0.5986       |     **0.54783** - 0.5655      |

*-x: The figure represents the original data in the paper.*

*†: The 5-fold cross validation results are provided by Tang et al.* 

*Global: The data in this column was evaluated across the entire dataset.*
