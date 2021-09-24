# ICCV2021-Papers-with-Code-Demo

 :star_and_crescent:**论文下载：**

**密码：aicv**

> CVPR 2021整理：https://github.com/DWCTOD/CVPR2021-Papers-with-Code-Demo
>
> **论文下载：https://pan.baidu.com/share/init?surl=gjfUQlPf73MCk4vM8VbzoA**
>
> **密码：aicv**

:star2: [ICCV 2021](http://iccv2021.thecvf.com/home )持续更新最新论文/paper和相应的开源代码/code！

:car: ICCV 2021 收录[列表](https://docs.google.com/spreadsheets/u/1/d/e/2PACX-1vRfaTmsNweuaA0Gjyu58H_Cx56pGwFhcTYII0u1pg0U7MbhlgY0R6Y-BbK3xFhAiwGZ26u3TAtN5MnS/pubhtml)

:steam_locomotive:**ICCV 2021 报告和demo视频汇总**  https://space.bilibili.com/288489574

:car: 官网链接：http://iccv2021.thecvf.com/home

> :timer_clock: 时间
> :watch: 论文/paper接收公布时间：2021年7月23日

> :hand: ​注：欢迎各位大佬提交issue，分享ICCV 2021论文/paper和开源项目！共同完善这个项目
>
> :airplane: 为了方便下载，已将论文/paper存储在文件夹中 :heavy_check_mark: 表示论文/paper[已下载 / Paper Download](https://github.com/DWCTOD/ICCV2021-Papers-with-Code-Demo/tree/main/paper)

## **:fireworks: 欢迎进群** | Welcome

ICCV 2021 论文/paper交流群已成立！已经收录的同学，可以添加微信：**nvshenj125**，请备注：**ICCV+姓名+学校/公司名称**！一定要根据格式申请，可以拉你进群。

 image<a name="Contents"></a>

## :hammer: **目录 |Table of Contents（点击直接跳转）**

- [Backbone](#Backbone)
- [Dataset](#dataset)
- [Loss](#loss)
- [NAS](#NAS)
- [Image Classification](#Image-Classification)
- [Vision Transformer](#VisualTransformer)
- [目标检测/Object Detection](#ObjectDetection)
- [显著性检测/Salient Object Detection](#Salient-Object-Detection)
- [3D目标检测 / 3D Object Detection](#3D-Object-Detection)
- [目标跟踪 / Object Tracking](#ObjectTracking)
- [Image Semantic Segmentation](#ImageSemanticSegmentation)
- [Semantic Scene Segmentation](#Semantic-Scene-Segmentation)
- [3D Semantic Segmentation](#3D-Semantic-Segmentation)
- [3D Instance Segmentation](#3D-Instance-Segmentation)
- [实例分割/Instance Segmentation](#InstanceSegmentation)
- [视频分割 / video semantic segmentation](#video-semantic-segmentation)
- [医学图像分割/ Medical Image Segmentation](#MedicalImageSegmentation)
- [医学图像分析/Medical Image Analysis](#Medical-Image-Analysis)
- [GAN](#GAN)
- [Style Transfer](#Style-Transfer)
- [细粒度分类/Fine-Grained Visual Categorization](#Fine-Grained-Visual-Categorization)
- [Multi-Label Recognition](#Multi-Label-Recognition)
- [Long-Tailed Recognition](#Long-Tailed-Recognition)
- [Geometric deep learning](#GeometricDeepLearning)
- [Zero/Few Shot](#ZeroFewShot)
- [Unsupervised](#Unsupervised)
- [Self-supervised](#Self-supervised)
- [Semi Supervised](#Semi-Supervised)
- [Weakly Supervised](#Weakly-Supervised)
- [Active Learning](#Active-Learning)
- [Action Detection](#Action-Detection)
- [ 动作识别/Action Recognition](#HumanActions)
- [时序行为检测 / Temporal Action Localization](#TemporalActionLocalization)
- [手语识别/Sign Language Recognition](#SignLanguageRecognition)
- [Hand Pose Estimation](#Hand-Pose-Estimation)
- [Pose Estimation](#PoseEstimation)
- [6D Object Pose Estimation](#6D-Object-Pose-Estimation)
- [Human Reconstruction](#Human-Reconstruction)
- [3D Scene Understanding](#3D-Scene-Understanding)
- [人脸识别/Face Recognition](#Face-Recognition)
- [人脸对齐/Face Alignment](#Face-Alignment)
- [人脸编辑/Facial Editing](#Facial-Editing)
- [Face Reconstruction](#FaceReconstruction)
- [Facial Expression Recognition](#Facial-Expression-Recognition)
- [行人重识别/Re-Identification](#Re-Identification)
- [Vehicle Re-identification](#VehicleRe-identification)
- [Pedestrian Detection](#Pedestrian-Detection)
- [人群计数 / Crowd Counting](#Crowd-Counting)
- [Motion Forecasting](#MotionForecasting)
- [Pedestrian Trajectory Prediction](#Pedestrian-Trajectory-Prediction)
- [Face-Anti-spoofing](#Face-Anti-spoofing)
- [deepfake](#deepfake)
- [对抗攻击/ Adversarial Attacks](#AdversarialAttacks)
- [跨模态检索/Cross-Modal Retrieval](#Cross-Modal-Retrieval)
- [神经辐射场/NeRF](#NeRF)
- [阴影去除/Shadow Removal](#Shadow-Removal)
- [图像检索/Image Retrieval](#ImageRetrieval)
- [超分辨/Super-Resolution](#Super-Resolution)
- [图像重建/Image Reconstruction](#ImageReconstruction)
- [去模糊/Image Deblurring](#Deblurring)
- [去噪/Image Denoising](#ImageDenoising)
- [去雪/Image Desnowing](#ImageDesnowing)
- [图像增强/Image Enhancement](#ImageEnhancement)
- [图像匹配/Image Matching](#Image-Matching)
- [图像质量/Image Quality](#Image-Quality )
- [图像压缩/Image Compression](#Image-Compression)
- [图像复原/Image Restoration](#Image-Restoration)
- [图像修复/Image Inpainting](#Image-Inpainting)
- [视频修复/Video Inpainting](#VideoInpainting)
- [视频插帧/Video Frame Interpolation](#VideoFrameInterpolation)
- [Video Reasoning](#Video-Reasoning)
- [Video Recognition](#Video-Recognition)
- [Visual Question Answering](#Visual-Question-Answering)
- [Matching](#Matching)
- [人机交互/Hand-object Interaction](#Hand-object-Interaction)
- [视线估计 / Gaze Estimation](#GazeEstimation)
- [深度估计 / Depth Estimation](#DepthEstimation)
- [Contrastive-Learning](#Contrastive-Learning)
- [Graph Convolution Networks](#Graph-Convolution-Networks)
- [模型压缩/Compress](#Compress)
- [Quantization](#Quantization)
- [Knowledge Distillation](#Knowledge-Distillation)
- [点云/Point Cloud](#pointcloud)
- [3D reconstruction](#3D-reconstruction)
- [字体生成/Font Generation](#FontGeneration)
- [文本检测 / Text Detection](#TextDetection)
- [文本识别 / Text Recognition](#TextRecognition)
- [Scene Text Recognizer](#SceneTextRecognizer)
- [Autonomous-Driving](#Autonomous-Driving)
- [Visdrone_detection](#Visdrone_detection)
- [异常检测 / Anomaly Detection](#AnomalyDetection)
- [其他/Others](#Others)

<a name="Backbone"></a>

## Backbone

:heavy_check_mark:**Conformer: Local Features Coupling Global Representations for Visual Recognition**

- 论文/paper：https://arxiv.org/abs/2105.03889
- 代码/code：https://github.com/pengzhiliang/Conformer

**Contextual Convolutional Neural Networks**

- 论文/paper：https://arxiv.org/abs/2108.07387
- 代码/code：https://github.com/iduta/coconv

**Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions**

- 解读：https://zhuanlan.zhihu.com/p/353222035

- 论文/paper：https://arxiv.org/abs/2102.12122
- 代码/code：https://github.com/whai362/PVT

**Reg-IBP: Efficient and Scalable Neural Network Robustness Training via Interval Bound Propagation**

- 论文/paper：None
- 代码/code：https://github.com/harrywuhust2022/Reg_IBP_ICCV2021

**Why Approximate Matrix Square Root Outperforms Accurate SVD in Global Covariance Pooling?**

- 论文/paper：https://arxiv.org/abs/2105.02498
- 代码/code：https://github.com/KingJamesSong/DifferentiableSVD

[返回目录/back](#Contents)

<a name="dataset"></a>

## Dataset

:heavy_check_mark:**FineAction: A Fined Video Dataset for Temporal Action Localization**

- 论文/paper：https://arxiv.org/abs/2105.11107 | [主页/Homepage](https://deeperaction.github.io/fineaction/)
- 代码/code： None

**KoDF: A Large-scale Korean DeepFake Detection Dataset**

- 论文/paper：https://arxiv.org/abs/2103.10094
- 代码/code：https://moneybrain-research.github.io/kodf

**LLVIP: A Visible-infrared Paired Dataset for Low-light Vision**

- 论文/paper：https://arxiv.org/abs/2108.10831 | [主页/Homepage](https://bupt-ai-cz.github.io/LLVIP/)
- 代码/code： None

**Matching in the Dark: A Dataset for Matching Image Pairs of Low-light Scenes**

- 论文/paper：https://arxiv.org/abs/2109.03585
- 代码/code： None

**Meta Self-Learning for Multi-Source Domain Adaptation: A Benchmark**

- 论文/paper：https://arxiv.org/abs/2108.10840 | [主页/Homepage](https://bupt-ai-cz.github.io/Meta-SelfLearning/)
- 代码/code：https://github.com/bupt-ai-cz/Meta-SelfLearning

:heavy_check_mark:**MultiSports: A Multi-Person Video Dataset of Spatio-Temporally Localized Sports Actions**

- 论文/paper：https://arxiv.org/abs/2105.07404 | [主页/Homepage](https://deeperaction.github.io/multisports/)
- 代码/code：https://github.com/MCG-NJU/MultiSports/

**Semantically Coherent Out-of-Distribution Detection**

- 论文/paper：https://arxiv.org/abs/2108.11941 | [主页/Homepage](https://jingkang50.github.io/projects/scood)
- 代码/code：https://github.com/jingkang50/ICCV21_SCOOD

**StereOBJ-1M: Large-scale Stereo Image Dataset for 6D Object Pose Estimation**

- 论文/paper：https://arxiv.org/abs/2109.10115
- 代码/code：None

**STRIVE: Scene Text Replacement In Videos**

- 论文/paper：https://arxiv.org/abs/2109.02762 | [主页/Homepage](https://striveiccv2021.github.io/STRIVE-ICCV2021/)
- 代码/code：None

**The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization**

- 论文/paper：https://arxiv.org/abs/2006.16241
- 代码/code：https://github.com/hendrycks/imagenet-r

**Webly Supervised Fine-Grained Recognition: Benchmark Datasets and An Approach**

- 论文/paper：https://arxiv.org/abs/2108.02399
- 代码/code：https://github.com/NUST-Machine-Intelligence-Laboratory/weblyFG-dataset

**Who's Waldo? Linking People Across Text and Images** (Oral)

- 论文/paper：https://arxiv.org/abs/2108.07253
- 代码/code：None

[返回目录/back](#Contents)

<a name="loss"></a>

## Loss

**Asymmetric Loss For Multi-Label Classification**

- 论文/paper：https://arxiv.org/abs/2009.14119
- 代码/code：https://github.com/Alibaba-MIIL/ASL

**Bias Loss for Mobile Neural Networks**

- 论文/paper：https://arxiv.org/abs/2107.11170
- 代码/code：None

**Focal Frequency Loss for Image Reconstruction and Synthesis**

- 论文/paper：https://arxiv.org/abs/2012.12821
- 代码/code：https://github.com/EndlessSora/focal-frequency-loss

**Orthogonal Projection Loss**

- 论文/paper：https://arxiv.org/abs/2103.14021
- 代码/code：https://github.com/kahnchana/opl

**Rank & Sort Loss for Object Detection and Instance Segmentation** (Oral)

- 论文/paper：https://arxiv.org/abs/2107.11669
- 代码/code：https://github.com/kemaloksuz/RankSortLoss

[返回目录/back](#Contents)

<a name="NAS"></a>

## NAS

**BN-NAS: Neural Architecture Search with Batch Normalization**

- 论文/paper：https://arxiv.org/abs/2108.07375
- 代码/code：None

**BossNAS: Exploring Hybrid CNN-transformers with Block-wisely Self-supervised Neural Architecture Search**

- 论文/paper：https://arxiv.org/pdf/2103.12424.pdf
- 代码/code：https://github.com/changlin31/BossNAS

**CONet: Channel Optimization for Convolutional Neural Networks**

- 论文/paper：https://arxiv.org/abs/2108.06822
- 代码/code：None

**FOX-NAS: Fast, On-device and Explainable Neural Architecture Search**

- 论文/paper：https://arxiv.org/abs/2108.08189
- 代码/code：https://github.com/great8nctu/FOX-NAS

**Pi-NAS: Improving Neural Architecture Search by Reducing Supernet Training Consistency Shift**

- 论文/paper：https://arxiv.org/abs/2108.09671v1
- 代码/code：https://github.com/Ernie1/Pi-NAS

**RANK-NOSH: Efficient Predictor-Based Architecture Search via Non-Uniform Successive Halving**

- 论文/paper：https://arxiv.org/abs/2108.08019
- 代码/code：https://github.com/ruocwang

**Single-DARTS: Towards Stable Architecture Search**

- 论文/paper：https://arxiv.org/abs/2108.08128
- 代码/code：https://github.com/PencilAndBike/Single-DARTS.git

[返回目录/back](#Contents)

<a name="Image-Classification"></a>

## Image Classification

**Low-Shot Validation: Active Importance Sampling for Estimating Classifier Performance on Rare Categories**

- 论文/paper：https://arxiv.org/abs/2109.05720
- 代码/code：None

**Tune It or Don't Use It: Benchmarking Data-Efficient Image Classification**

- 论文/paper：https://arxiv.org/abs/2108.13122
- 代码/code：None

[返回目录/back](#Contents)

<a name="VisualTransformer"></a>

## Vision Transformer

**An End-to-End Transformer Model for 3D Object Detection**

- 论文/paper：https://arxiv.org/abs/2109.08141
- 代码/code：None

**AutoFormer: Searching Transformers for Visual Recognition**

- 论文/paper：https://arxiv.org/abs/2107.00651
- 代码/code：https://github.com/microsoft/AutoML

**BossNAS: Exploring Hybrid CNN-transformers with Block-wisely Self-supervised Neural Architecture Search**

- 论文/paper：https://arxiv.org/pdf/2103.12424.pdf
- 代码/code：https://github.com/changlin31/BossNAS

**Conditional DETR for Fast Training Convergence**

- 论文/paper：https://arxiv.org/abs/2108.06152
- 代码/code：https://git.io/ConditionalDETR

**Dyadformer: A Multi-modal Transformer for Long-Range Modeling of Dyadic Interactions**

- 论文/paper：https://arxiv.org/abs/2109.09487
- 代码/code：None

**Eformer: Edge Enhancement based Transformer for Medical Image Denoising**

- 论文/paper：https://arxiv.org/abs/2109.08044
- 代码/code：None

**Fast Convergence of DETR with Spatially Modulated Co-Attention**

- 解读：https://zhuanlan.zhihu.com/p/397083124
- 论文/paper：https://arxiv.org/abs/2108.02404
- 代码/code：https://github.com/gaopengcuhk/SMCA-DETR

**FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting**

- 论文/paper：https://arxiv.org/abs/2108.01912
- 代码/code：None

**Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers**  (Oral)

- 论文/paper：https://arxiv.org/pdf/2103.15679.pdf
- 代码/code：https://github.com/hila-chefer/Transformer-MM-Explainability

**GroupFormer: Group Activity Recognition with Clustered Spatial-Temporal Transformer**

- 论文/paper：https://arxiv.org/abs/2108.12630
- 代码/code：https://github.com/xueyee/GroupFormer

**HiFT: Hierarchical Feature Transformer for Aerial Tracking**

- 论文/paper：https://arxiv.org/abs/2108.00202
- 代码/code：https://github.com/vision4robotics/HiFT

**High-Fidelity Pluralistic Image Completion with Transformers**

- 论文/paper：https://arxiv.org/pdf/2103.14031.pdf | [主页/Homepage](http://raywzy.com/ICT/)
- 代码/code： https://github.com/raywzy/ICT

**Improving 3D Object Detection with Channel-wise Transformer**

- 论文/paper：https://arxiv.org/abs/2108.10723
- 代码/code：None

**Is it Time to Replace CNNs with Transformers for Medical Images?**

- 论文/paper：https://arxiv.org/abs/2108.09038
- 代码/code：None

**Learning Spatio-Temporal Transformer for Visual Tracking**

- 论文/paper：https://arxiv.org/abs/2103.17154
- 代码/code：https://github.com/researchmm/Stark

**MUSIQ: Multi-scale Image Quality Transformer**

- 论文/paper：https://arxiv.org/abs/2108.05997
- 代码/code：None

**Paint Transformer: Feed Forward Neural Painting with Stroke Prediction** (Oral)

- 解读：https://zhuanlan.zhihu.com/p/400017971

- 论文/paper：https://arxiv.org/abs/2108.03798
- 代码/code：https://github.com/Huage001/PaintTransformer

**PlaneTR: Structure-Guided Transformers for 3D Plane Recovery**

- 论文/paper：https://arxiv.org/abs/2107.13108
- 代码/code： https://github.com/IceTTTb/PlaneTR3D

**PnP-DETR: Towards Efficient Visual Analysis with Transformers**

- 论文/paper：https://arxiv.org/abs/2109.07036
- 代码/code：https://github.com/twangnh/pnp-detr

**Pose Transformers (POTR): Human Motion Prediction with Non-Autoregressive Transformers**

- 论文/paper：https://arxiv.org/abs/2109.07531
- 代码/code：None

**PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers**  (Oral)

- 论文/paper：https://arxiv.org/abs/2108.08839
- 代码/code：https://github.com/yuxumin/PoinTr

**Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions**

- 解读：https://zhuanlan.zhihu.com/p/353222035

- 论文/paper：https://arxiv.org/abs/2102.12122
- 代码/code：https://github.com/whai362/PVT

**Rethinking and Improving Relative Position Encoding for Vision Transformer**

- 论文/paper：https://houwenpeng.com/publications/iRPE.pdf
- 代码/code：https://github.com/wkcn/iRPE-model-zoo

**Rethinking Spatial Dimensions of Vision Transformers**

- 论文/paper：https://arxiv.org/abs/2103.16302
- 代码/code：https://github.com/naver-ai/pit

**Simpler is Better: Few-shot Semantic Segmentation with Classifier Weight Transformer**

- 论文/paper：https://arxiv.org/abs/2108.03032
- 代码/code：https://github.com/zhiheLu/CWTfor-FSS

**SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer**

- 论文/paper：https://arxiv.org/abs/2108.04444
- 代码/code：https://github.com/AllenXiangX/SnowflakeNet

**Spatial-Temporal Transformer for Dynamic Scene Graph Generation**

- 解读：[用于视频场景图生成的Spatial-Temporal Transformer](https://zhuanlan.zhihu.com/p/393637591)
- 论文/paper：https://arxiv.org/abs/2107.12309
- 代码/code：None

**SOTR: Segmenting Objects with Transformers**

- 论文/paper：https://arxiv.org/abs/2108.06747
- 代码/code：https://github.com/easton-cau/SOTR

**Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**

- 论文/paper：https://arxiv.org/abs/2103.14030
- 代码/code：https://github.com/microsoft/Swin-Transformer

**Revisiting Stereo Depth Estimation From a Sequence-to-Sequence Perspective with Transformers**

- 论文/paper：https://arxiv.org/abs/2011.02910
- 代码/code：https://github.com/mli0603/stereo-transformer

**The Animation Transformer: Visual Correspondence via Segment Matching**

- 论文/paper：https://arxiv.org/abs/2109.02614
- 代码/code：None

**The Right to Talk: An Audio-Visual Transformer Approach**

- 论文/paper：https://arxiv.org/abs/2108.03256
- 代码/code：None

**TPH-YOLOv5: Improved YOLOv5 Based on Transformer Prediction Head for Object Detection on Drone-captured Scenarios**

- 论文/paper：https://arxiv.org/abs/2108.11539
- 代码/code：None

**TransFER: Learning Relation-aware Facial Expression Representations with Transformers**

- 论文/paper：https://arxiv.org/abs/2108.11116
- 代码/code：None

**TransPose: Keypoint Localization via Transformer**

- 论文/paper：https://arxiv.org/abs/2012.14214
- 代码/code：https://github.com/yangsenius/TransPose

:heavy_check_mark:**Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet**

- 论文/paper：https://arxiv.org/abs/2101.11986

- 代码/code： https://github.com/yitu-opensource/T2T-ViT

:heavy_check_mark:**Visual Transformer with Statistical Test for COVID-19 Classification**

- 论文/paper：https://arxiv.org/abs/2107.05334
- 代码/code： None

**Vision Transformer with Progressive Sampling**

- 论文/paper：https://arxiv.org/abs/2108.01684
- 代码/code：https://github.com/yuexy/PS-ViT

**Visual Saliency Transformer**

- 解读：https://blog.csdn.net/qq_39936426/article/details/117199411

- 论文/paper：https://arxiv.org/abs/2104.12099
- 代码/code： https://github.com/nnizhang/VST

**Vision-Language Transformer and Query Generation for Referring Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.05565
- 代码/code：https://github.com/henghuiding/Vision-Language-Transformer

**Voxel Transformer for 3D Object Detection**

- 论文/paper：https://arxiv.org/abs/2109.02497
- 代码/code： None

[返回目录/back](#Contents)

<a name="ObjectDetection"></a>

## 目标检测/Object Detection

**Active Learning for Deep Object Detection via Probabilistic Modeling**

- 论文/paper：https://arxiv.org/abs/2103.16130
- 代码/code：None

**Boosting Weakly Supervised Object Detection via Learning Bounding Box Adjusters**

- 论文/paper：https://arxiv.org/abs/2108.01499
- 代码/code：https://github.com/DongSky/lbba_boosted_wsod

**Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery**

- 论文/paper：https://arxiv.org/abs/2108.07002
- 代码/code：https://github.com/Z-Zheng/ChangeStar

**Conditional Variational Capsule Network for Open Set Recognition**

- 论文/paper： https://arxiv.org/abs/2104.09159

- 代码/code：https://github.com/guglielmocamporese/cvaecaposr

**DetCo: Unsupervised Contrastive Learning for Object Detection**

- 论文/paper：https://arxiv.org/abs/2102.04803
- 代码/code： https://github.com/xieenze/DetCo

**DeFRCN: Decoupled Faster R-CNN for Few-Shot Object Detection**

- 论文/paper：https://arxiv.org/abs/2108.09017
- 代码/code：None

**Deployment of Deep Neural Networks for Object Detection on Edge AI Devices with Runtime Optimization**

- 论文/paper：https://arxiv.org/abs/2108.08166
- 代码/code：None

**Detecting Invisible People**

- 论文/paper：https://arxiv.org/abs/2012.08419 | [主页/Homepage](http://www.cs.cmu.edu/~tkhurana/invisible.htm)
- 代码/code：None

**FMODetect: Robust Detection and Trajectory Estimation of Fast Moving Objects**

- 论文/paper：None
- 代码/code：https://github.com/rozumden/FMODetect

**GraphFPN: Graph Feature Pyramid Network for Object Detection**

- 论文/paper：https://arxiv.org/abs/2108.00580
- 代码/code：None

**Human Detection and Segmentation via Multi-view Consensus**

- 论文/paper：None
- 代码/code：https://github.com/isinsukatircioglu/mvc

**MDETR : Modulated Detection for End-to-End Multi-Modal Understanding**

- 论文/paper：https://arxiv.org/abs/2104.12763 | [主页/Homepage](https://ashkamath.github.io/mdetr_page/)
- 代码/code： https://github.com/ashkamath/mdetr

**Mutual Supervision for Dense Object Detection**

- 论文/paper：https://arxiv.org/abs/2109.05986
- 代码/code：None

**Oriented R-CNN for Object Detection**

- 论文/paper：https://arxiv.org/abs/2108.05699
- 代码/code：https://github.com/jbwang1997/OBBDetection

**Rank & Sort Loss for Object Detection and Instance Segmentation** (Oral)

- 论文/paper：https://arxiv.org/abs/2107.11669
- 代码/code：https://github.com/kemaloksuz/RankSortLoss

**Reconcile Prediction Consistency for Balanced Object Detection**

- 论文/paper：https://arxiv.org/abs/2108.10809
- 代码/code：None

**TOOD: Task-aligned One-stage Object Detection** (Oral)

- 论文/paper：https://arxiv.org/abs/2108.07755
- 代码/code：https://github.com/fcjian/TOOD

**Vector-Decomposed Disentanglement for Domain-Invariant Object Detection**

- 论文/paper：https://arxiv.org/abs/2108.06685
- 代码/code：None

[返回目录/back](#Contents)

<a name="Salient-Object-Detection"></a>

## Salient Object Detections

**Disentangled High Quality Salient Object Detection**

- 论文/paper：https://arxiv.org/abs/2108.03551
- 代码/code：None

**RGB-D Saliency Detection via Cascaded Mutual Information Minimization**

- 论文/paper：https://arxiv.org/abs/2109.07246
- 代码/code：

**Specificity-preserving RGB-D Saliency Detection**

- 论文/paper：https://arxiv.org/abs/2108.08162
- 代码/code：https://github.com/taozh2017/SPNet

[返回目录/back](#Contents)

<a name="3D-Object-Detection"></a>

# 3D目标检测 / 3D Object Detection

**An End-to-End Transformer Model for 3D Object Detection**

- 论文/paper：https://arxiv.org/abs/2109.08141
- 代码/code：None

**Fog Simulation on Real LiDAR Point Clouds for 3D Object Detection in Adverse Weather**

- 论文/paper：https://arxiv.org/abs/2108.05249
- 代码/code：https://github.com/MartinHahner/LiDAR_fog_sim

**LIGA-Stereo: Learning LiDAR Geometry Aware Representations for Stereo-based 3D Detector**

- 论文/paper：https://arxiv.org/abs/2108.08258
- 代码/code：None

**Improving 3D Object Detection with Channel-wise Transformer**

- 论文/paper：https://arxiv.org/abs/2108.10723
- 代码/code：None

**Is Pseudo-Lidar needed for Monocular 3D Object detection?**

- 论文/paper：https://arxiv.org/abs/2108.06417
- 代码/code：None

**ODAM: Object Detection, Association, and Mapping using Posed RGB Video** （Oral）

- 论文/paper：https://arxiv.org/abs/2108.10165v1
- 代码/code：None

**Pyramid R-CNN: Towards Better Performance and Adaptability for 3D Object Detection**

- 论文/paper：https://arxiv.org/abs/2109.02499
- 代码/code：None

**RandomRooms: Unsupervised Pre-training from Synthetic Shapes and Randomized Layouts for 3D Object Detection**

- 论文/paper：https://arxiv.org/abs/2108.07794
- 代码/code：None

**Voxel Transformer for 3D Object Detection**

- 论文/paper：https://arxiv.org/abs/2109.02497
- 代码/code： None

**Unsupervised Domain Adaptive 3D Detection with Multi-Level Consistency**

- 论文/paper：https://arxiv.org/pdf/2107.11355.pdf
- 代码/code：None

[返回目录/back](#Contents)

<a name="ObjectTracking"></a>

## 目标跟踪 / Object Tracking

**DepthTrack : Unveiling the Power of RGBD Tracking**

- 论文/paper：https://arxiv.org/abs/2108.13962
- 代码/code：None

**Exploring Simple 3D Multi-Object Tracking for Autonomous Driving**

- 论文/paper：https://arxiv.org/abs/2108.10312v1
- 代码/code：None

**Is First Person Vision Challenging for Object Tracking?**

- 论文/paper：https://arxiv.org/abs/2108.13665
- 代码/code：None

**Learning to Track Objects from Unlabeled Videos**

- 论文/paper：https://arxiv.org/abs/2108.12711
- 代码/code：https://github.com/VISION-SJTU/USOT

**Learn to Match: Automatic Matching Network Design for Visual Tracking**

- 论文/paper：https://arxiv.org/abs/2108.00803
- 代码/code：https://github.com/JudasDie/SOTS

**Making Higher Order MOT Scalable: An Efficient Approximate Solver for Lifted Disjoint Paths**

- 论文/paper：https://arxiv.org/abs/2108.10606
- 代码/code：https://github.com/TimoK93/ApLift

**Saliency-Associated Object Tracking**

- 论文/paper：https://arxiv.org/abs/2108.03637
- 代码/code：None

**Video Annotation for Visual Tracking via Selection and Refinement**

- 论文/paper：https://arxiv.org/abs/2108.03821
- 代码/code：https://github.com/Daikenan/VASR

[返回目录/back](#Contents)

<a name="ImageSemanticSegmentation"></a>

## Image Semantic Segmentation

**Complementary Patch for Weakly Supervised Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.03852
- 代码/code：None

**Calibrated Adversarial Refinement for Stochastic Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2006.13144
- 代码/code：https://github.com/EliasKassapis/CARSSS

**Deep Metric Learning for Open World Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.04562
- 代码/code：None

**Dual Path Learning for Domain Adaptation of Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.06337
- 代码/code：https://github.com/royee182/DPL

**EdgeFlow: Achieving Practical Interactive Segmentation with Edge-Guided Flow**

- 论文/paper：https://arxiv.org/abs/2109.09406
- 代码/code：https://github.com/PaddlePaddle/PaddleSeg

**Exploiting Spatial-Temporal Semantic Consistency for Video Scene Parsing**

- 论文/paper：https://arxiv.org/abs/2109.02281
- 代码/code：None

**Exploiting a Joint Embedding Space for Generalized Zero-Shot Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.06536
- 代码/code：None

**Exploring Cross-Image Pixel Contrast for Semantic Segmentation** （Oral）

- 论文/paper：https://arxiv.org/abs/2101.11939
- 代码/code：https://github.com/tfzhou/ContrastiveSeg

**Enhanced Boundary Learning for Glass-like Object Segmentation**

- 论文/paper：https://arxiv.org/abs/2103.15734
- 代码/code：https://github.com/hehao13/EBLNet

**From Contexts to Locality: Ultra-high Resolution Ie Segmentation via Locality-aware Contextual Correlation**

- 论文/paper：https://arxiv.org/abs/2109.02580
- 代码/code：https://github.com/liqiokkk/FCtL

**ISNet: Integrate Image-Level and Semantic-Level Context for Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.12382v1
- 代码/code：None

**Generalize then Adapt: Source-Free Domain Adaptive Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.11249
- 代码/code：https://sites.google.com/view/sfdaseg

**Labels4Free: Unsupervised Segmentation using StyleGAN**

- 论文/paper：https://arxiv.org/abs/2103.14968 | [主页/Homepage](https://rameenabdal.github.io/Labels4Free)
- 代码/code：None

**LabOR: Labeling Only if Required for Domain Adaptive Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.05570
- 代码/code：None

**Learning Meta-class Memory for Few-Shot Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.02958
- 代码/code：None

**Leveraging Auxiliary Tasks with Affinity Learning for Weakly Supervised Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2107.11787
- 代码/code：https://github.com/xulianuwa/AuxSegNet

**Mining Contextual Information Beyond Image for Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.11819
- 代码/code：None

**Mining Latent Classes for Few-shot Segmentation**(Oral)

- 论文/paper：https://arxiv.org/abs/2103.15402
- 代码/code：https://github.com/LiheYoung/MiningFSS

**Multi-Target Adversarial Frameworks for Domain Adaptation in Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.06962
- 代码/code：None

**Multi-Anchor Active Domain Adaptation for Semantic Segmentation** (Oral)

- 论文/paper：https://arxiv.org/abs/2108.08012
- 代码/code：None

**Personalized Image Semantic Segmentation**

- 论文/paper：None
- 代码/code： https://github.com/zhangyuygss/PIS

**Pixel Contrastive-Consistent Semi-Supervised Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.09025
- 代码/code：None

**Pseudo-mask Matters inWeakly-supervised Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.12995
- 代码/code：https://github.com/Eli-YiLi/PMM

**RECALL: Replay-based Continual Learning in Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.03673
- 代码/code：None

**Re-distributing Biased Pseudo Labels for Semi-supervised Semantic Segmentation: A Baseline Investigation**(Oral)

- 论文/paper：https://arxiv.org/abs/2107.11279
- 代码/code：https://github.com/CVMI-Lab/DARS

**Semantic Segmentation on VSPW Dataset through Aggregation of Transformer Models**

- 论文/paper：https://arxiv.org/abs/2109.01316
- 代码/code：None

**Self-Regulation for Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.09702v1
- 代码/code：None

**Semantic Concentration for Domain Adaptation**

- 论文/paper：https://arxiv.org/abs/2108.05720
- 代码/code：None

**ShapeConv: Shape-aware Convolutional Layer for Indoor RGB-D Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.10528
- 代码/code：None

**Simpler is Better: Few-shot Semantic Segmentation with Classifier Weight Transformer**

- 论文/paper：https://arxiv.org/abs/2108.03032
- 代码/code：https://github.com/zhiheLu/CWTfor-FSS

**SOTR: Segmenting Objects with Transformers**

- 论文/paper：https://arxiv.org/abs/2108.06747
- 代码/code：https://github.com/easton-cau/SOTR

**Standardized Max Logits: A Simple yet Effective Approach for Identifying Unexpected Road Obstacles in Urban-Scene Segmentation**

- 论文/paper：https://arxiv.org/abs/2107.11264v1
- 代码/code：None

**The Marine Debris Dataset for Forward-Looking Sonar Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.06800
- 代码/code：https://github.com/mvaldenegro/marine-debris-fls-datasets/

**Unsupervised Semantic Segmentation by Contrasting Object Mask Proposals**

- 论文/paper：https://arxiv.org/pdf/2102.06191.pdf
- 代码/code：https://github.com/wvangansbeke/Unsupervised-Semantic-Segmentation

**Weakly Supervised Temporal Anomaly Segmentation with Dynamic Time Warping**

- 论文/paper：https://arxiv.org/abs/2108.06816
- 代码/code：None

[返回目录/back](#Contents)

<a name="Semantic-Scene-Segmentation"></a>

## Semantic Scene Segmentation

**BiMaL: Bijective Maximum Likelihood Approach to Domain Adaptation in Semantic Scene Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.03267
- 代码/code：None

[返回目录/back](#Contents)

<a name="3D-Semantic-Segmentation"></a>

## 3D Semantic Segmentation

**VMNet: Voxel-Mesh Network for Geodesic-aware 3D Semantic Segmentation**

- 论文/paper：None
- 代码/code：https://github.com/hzykent/VMNet

[返回目录/back](#Contents)

<a name="3D-Instance-Segmentation"></a>

## 3D Instance Segmentation

**Hierarchical Aggregation for 3D Instance Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.02350
- 代码/code：https://github.com/hustvl/HAIS

**Instance Segmentation in 3D Scenes using Semantic Superpoint Tree Networks**

- 论文/paper：https://arxiv.org/abs/2108.07478
- 代码/code：https://github.com/Gorilla-Lab-SCUT/SSTNet

[返回目录/back](#Contents)

<a name="InstanceSegmentation"></a>

## 实例分割/Instance Segmentation

**CDNet: Centripetal Direction Network for Nuclear Instance Segmentation**

- 论文/paper：None

- 代码/code： https://github.com/2021-ICCV/CDNet

:heavy_check_mark:**Crossover Learning for Fast Online Video Instance Segmentation**

- 论文/paper：https://arxiv.org/abs/2104.05970

- 代码/code： https://github.com/hustvl/CrossVIS

:heavy_check_mark:**Instances as Queries**

- 论文/paper：https://arxiv.org/abs/2105.01928
- 代码/code： https://github.com/hustvl/QueryInst

**Rank & Sort Loss for Object Detection and Instance Segmentation** (Oral)

- 论文/paper：https://arxiv.org/abs/2107.11669
- 代码/code：https://github.com/kemaloksuz/RankSortLoss

[返回目录/back](#Contents)

<a name="video-semantic-segmentation"></a>

## 视频分割 / video semantic segmentation

**Domain Adaptive Video Segmentation via Temporal Consistency Regularization**

- 论文/paper：https://arxiv.org/abs/2107.11004
- 代码/code：https://github.com/Dayan-Guan/DA-VSN

**Full-Duplex Strategy for Video Object Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.03151 | [主页/homepage](http://dpfan.net/FSNet/)
- 代码/code：https://github.com/GewelsJI/FSNet

**Hierarchical Memory Matching Network for Video Object Segmentation**

- demo：https://www.bilibili.com/video/BV1Eg41157q3

- 论文/paper：https://arxiv.org/abs/2109.11404 | [主页/homepage](https://joonyoung-cv.github.io/)
- 代码/code：Hierarchical Memory Matching Network for Video Object Segmentation

**Joint Inductive and Transductive Learning for Video Object Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.03679
- 代码/code：https://github.com/maoyunyao/JOINT

[返回目录/back](#Contents)

<a name="MedicalImageSegmentation"></a>

# Medical Image Segmentation

**Recurrent Mask Refinement for Few-Shot Medical Image Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.00622
- 代码/code：None

[返回目录/back](#Contents)

<a name="Medical-Image-Analysis"></a>

## Medical Image Analysis

**Eformer: Edge Enhancement based Transformer for Medical Image Denoising**

- 论文/paper：https://arxiv.org/abs/2109.08044
- 代码/code：None

**Improving Tuberculosis (TB) Prediction using Synthetically Generated Computed Tomography (CT) Images**

- 论文/paper：https://arxiv.org/abs/2109.11480
- 代码/code：None

**Preservational Learning Improves Self-supervised Medical Image Models by Reconstructing Diverse Contexts**

- 论文/paper：https://arxiv.org/abs/2109.04379
- 代码/code：https://github.com/Luchixiang/PCRL

**Studying the Effects of Self-Attention for Medical Image Analysis**

- 论文/paper：https://arxiv.org/abs/2109.01486
- 代码/code：None

[返回目录/back](#Contents)

<a name="GAN"></a>

## GAN

**3DStyleNet: Creating 3D Shapes with Geometric and Texture Style Variations** (Oral)

- 论文/paper：https://arxiv.org/abs/2108.12958
- 代码/code：https://nv-tlabs.github.io/3DStyleNet/

**AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer**

- 论文/paper：https://arxiv.org/abs/2108.03647
- 代码/code：https://github.com/Huage001/AdaAttN

**Click to Move: Controlling Video Generation with Sparse Motion**

- 论文/paper：https://arxiv.org/abs/2108.08815
- 代码/code：https://github.com/PierfrancescoArdino/C2M

**Disentangled Lifespan Face Synthesis**

- 论文/paper：https://arxiv.org/abs/2108.02874 | [主页/Homepage](https://senhe.github.io/projects/iccv_2021_lifespan_face/)
- 代码/code：https://github.com/SenHe/DLFS

**Dual Projection Generative Adversarial Networks for Conditional Image Generation**

- 论文/paper：https://arxiv.org/abs/2108.09016
- 代码/code：None

**EigenGAN: Layer-Wise Eigen-Learning for GANs** 

- 论文/paper：https://arxiv.org/pdf/2104.12476.pdf
- 代码/code：https://github.com/LynnHo/EigenGAN-Tensorflow

**GAN Inversion for Out-of-Range Images with Geometric Transformations**

- 论文/paper：https://arxiv.org/abs/2108.08998
- 代码/code：https://kkang831.github.io/publication/ICCV_2021_BDInvert/

**Generative Models for Multi-Illumination Color Constancy**

- 论文/paper：https://arxiv.org/abs/2109.00863
- 代码/code：None

**Gradient Normalization for Generative Adversarial Networks**

- 论文/paper：https://arxiv.org/abs/2109.02235
- 代码/code：None

**Graph-to-3D: End-to-End Generation and Manipulation of 3D Scenes Using Scene Graphs**

- 论文/paper：https://arxiv.org/abs/2108.08841
- 代码/code：None

**Image Synthesis via Semantic Composition**

- 论文/paper：https://arxiv.org/abs/2109.07053 | [主页/Homepage](#https://shepnerd.github.io/scg/)
- 代码/code：https://github.com/dvlab-research/SCGAN

**InSeGAN: A Generative Approach to Segmenting Identical Instances in Depth Images**

- 论文/paper：https://arxiv.org/abs/2108.13865
- 代码/code：None

**Learning to Diversify for Single Domain Generalization**

- 论文/paper：https://arxiv.org/abs/2108.11726
- 代码/code：None

**Manifold Matching via Deep Metric Learning for Generative Modeling**

- 论文/paper：https://arxiv.org/abs/2106.10777
- 代码/code：https://github.com/dzld00/pytorch-manifold-matching

**Meta Gradient Adversarial Attack**

- 论文/paper：https://arxiv.org/abs/2108.04204
- 代码/code：None

**Online Multi-Granularity Distillation for GAN Compression**

- 论文/paper：https://arxiv.org/abs/2108.06908
- 代码/code：https://github.com/bytedance/OMGD

**Orthogonal Jacobian Regularization for Unsupervised Disentanglement in Image Generation**

- 论文/paper：https://arxiv.org/abs/2108.07668
- 代码/code：https://github.com/csyxwei/OroJaR

**PixelSynth: Generating a 3D-Consistent Experience from a Single Image**

- 论文/paper：https://arxiv.org/abs/2108.05892 | [主页/Homepage](https://crockwell.github.io/pixelsynth/)
- 代码/code：https://github.com/crockwell/pixelsynth/

**Robustness and Generalization via Generative Adversarial Training**

- 论文/paper：https://arxiv.org/abs/2109.02765
- 代码/code：None

**SemIE: Semantically-Aware Image Extrapolation**

- 论文/paper：https://arxiv.org/abs/2108.13702
- 代码/code：https://semie-iccv.github.io/

**SketchLattice: Latticed Representation for Sketch Manipulation**

- 论文/paper：https://arxiv.org/abs/2108.11636
- 代码/code：None

**Sketch Your Own GAN**

- 论文/paper：https://arxiv.org/abs/2108.02774
- 代码/code：https://github.com/PeterWang512/GANSketching

**Target Adaptive Context Aggregation for Video Scene Graph Generation**

- 论文/paper：https://arxiv.org/abs/2108.08121
- 代码/code：https://github.com/MCG-NJU/TRACE

**Toward Spatially Unbiased Generative Models**

- 论文/paper：https://arxiv.org/abs/2108.01285
- 代码/code：None

**Towards Vivid and Diverse Image Colorization with Generative Color Prior**

- 论文/paper：https://arxiv.org/abs/2108.08826
- 代码/code：None

**Unconditional Scene Graph Generation**

- 论文/paper：https://arxiv.org/abs/2108.05884
- 代码/code：None

**Unsupervised Geodesic-preserved Generative Adversarial Networks for Unconstrained 3D Pose Transfer**

- 论文/paper：https://arxiv.org/abs/2108.07520
- 代码/code：https://github.com/mikecheninoulu/Unsupervised_IEPGAN

[返回目录/back](#Contents)

<a name="Style-Transfer"></a>

## Style Transfer

**Domain-Aware Universal Style Transfer**

- 论文/paper：https://arxiv.org/abs/2108.04441
- 代码/code：None

[返回目录/back](#Contents)

<a name="Fine-Grained-Visual-Categorization"></a>

## 细粒度分类/Fine-Grained Visual Categorization

**Benchmark Platform for Ultra-Fine-Grained Visual Categorization BeyondHuman Performance**

- 论文/paper：None
- 代码/code：https://github.com/XiaohanYu-GU/Ultra-FGVC

**Webly Supervised Fine-Grained Recognition: Benchmark Datasets and An Approach**

- 论文/paper：https://arxiv.org/abs/2108.02399
- 代码/code：https://github.com/NUST-Machine-Intelligence-Laboratory/weblyFG-dataset

[返回目录/back](#Contents)

<a name="Multi-Label-Recognition"></a>

## Multi-Label Recognition

**Residual Attention: A Simple but Effective Method for Multi-Label Recognition**

- 论文/paper：https://arxiv.org/abs/2108.02456
- 代码/code：None

[返回目录/back](#Contents)

<a name="Long-Tailed-Recognition"></a>

## Long-Tailed Recognition

**ACE: Ally Complementary Experts for Solving Long-Tailed Recognition in One-Shot** Oral

- 论文/paper：https://arxiv.org/abs/2108.02385
- 代码/code：https://github.com/jrcai/ACE

[返回目录/back](#Contents)

<a name="GeometricDeepLearning"></a>

## Geometric deep learning

**Manifold Matching via Deep Metric Learning for Generative Modeling**

- 论文/paper：https://arxiv.org/abs/2106.10777
- 代码/code：https://github.com/dzld00/pytorch-manifold-matching

**Orthogonal Jacobian Regularization for Unsupervised Disentanglement in Image Generation**

- 论文/paper：None
- 代码/code：https://github.com/csyxwei/OroJaR

[返回目录/back](#Contents)

<a name="ZeroFewShot"></a>

## Zero/Few Shot

**Binocular Mutual Learning for Improving Few-shot Classification**

- 论文/paper：https://arxiv.org/abs/2108.12104v1
- 代码/code：None

**Boosting the Generalization Capability in Cross-Domain Few-shot Learning via Noise-enhanced Supervised Autoencoder**

- 论文/paper：https://arxiv.org/abs/2108.05028
- 代码/code：None

**Discriminative Region-based Multi-Label Zero-Shot Learning**

- 论文/paper：https://arxiv.org/abs/2108.05028
- 代码/code：None

**Domain Generalization via Gradient Surgery**

- 论文/paper：https://arxiv.org/abs/2108.01621
- 代码/code：None

**Exploiting a Joint Embedding Space for Generalized Zero-Shot Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.06536
- 代码/code：None

**Few-Shot Batch Incremental Road Object Detection via Detector Fusion**

- 论文/paper：https://arxiv.org/abs/2108.08048
- 代码/code：None

**Field-Guide-Inspired Zero-Shot Learning**

- 论文/paper：https://arxiv.org/abs/2108.10967
- 代码/code：None

**Few-shot Visual Relationship Co-localization**

- 论文/paper：https://arxiv.org/abs/2108.11618
- 代码/code：None

**Generalized Source-free Domain Adaptation**

- 论文/paper：https://arxiv.org/abs/2108.01614
- 代码/code：https://github.com/Albert0147/G-SFDA

**Generalized and Incremental Few-Shot Learning by Explicit Learning and Calibration without Forgetting**

- 论文/paper：https://arxiv.org/abs/2108.08165
- 代码/code：None

**Meta Navigator: Search for a Good Adaptation Policy for Few-shot Learning**

- 论文/paper：https://arxiv.org/abs/2109.05749
- 代码/code：None

**On the Importance of Distractors for Few-Shot Classification**

- 论文/paper：https://arxiv.org/abs/2109.09883
- 代码/code：None

**Relational Embedding for Few-Shot Classification**

- 论文/paper：https://arxiv.org/abs/2108.09666v1
- 代码/code：None

**SIGN: Spatial-information Incorporated Generative Network for Generalized Zero-shot Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.12517
- 代码/code：None

**Transductive Few-Shot Classification on the Oblique Manifold**

- 论文/paper：https://arxiv.org/abs/2108.04009
- 代码/code：None

**Visual Domain Adaptation for Monocular Depth Estimation on Resource-Constrained Hardware**

- 论文/paper：https://arxiv.org/abs/2108.02671
- 代码/code：None

[返回目录/back](#Contents)

<a name="Unsupervised"></a>

## Unsupervised

**Adversarial Robustness for Unsupervised Domain Adaptation**

- 论文/paper：https://arxiv.org/abs/2109.00946
- 代码/code：None

**Collaborative Unsupervised Visual Representation Learning from Decentralized Data**

- 论文/paper：https://arxiv.org/abs/2108.06492
- 代码/code：None

**Instance Similarity Learning for Unsupervised Feature Representation**

- 论文/paper：https://arxiv.org/abs/2108.02721
- 代码/code：https://github.com/ZiweiWangTHU/ISL

**Skeleton Cloud Colorization for Unsupervised 3D Action Representation Learning**

- 论文/paper：https://arxiv.org/abs/2108.01959
- 代码/code：None

**Unsupervised Dense Deformation Embedding Network for Template-Free Shape Correspondence**

- 论文/paper：https://arxiv.org/abs/2108.11609
- 代码/code：None

**Tune it the Right Way: Unsupervised Validation of Domain Adaptation via Soft Neighborhood Density**

- 论文/paper：https://arxiv.org/abs/2108.10860
- 代码/code：https://github.com/VisionLearningGroup/SND

[返回目录/back](#Contents)

<a name="Self-supervised"></a>

## Self-supervised

**Digging into Uncertainty in Self-supervised Multi-view Stereo**

- 论文/paper：https://arxiv.org/abs/2108.12966
- 代码/code：None

**Enhancing Self-supervised Video Representation Learning via Multi-level Feature Optimization**

- 论文/paper：https://arxiv.org/abs/2108.02183
- 代码/code：None

**Focus on the Positives: Self-Supervised Learning for Biodiversity Monitoring**

- 论文/paper：https://arxiv.org/abs/2108.06435
- 代码/code：None

**Improving Self-supervised Learning with Hardness-aware Dynamic Curriculum Learning: An Application to Digital Pathology**

- 论文/paper：https://arxiv.org/abs/2108.07183
- 代码/code：https://github.com/srinidhiPY/ICCVCDPATH2021-ID-8

**Meta Self-Learning for Multi-Source Domain Adaptation: A Benchmark**

- 论文/paper：https://arxiv.org/abs/2108.10840 | [主页/Homepage](https://bupt-ai-cz.github.io/Meta-SelfLearning/)
- 代码/code：https://github.com/bupt-ai-cz/Meta-SelfLearning

**Reducing Label Effort: Self-Supervised meets Active Learning**

- 论文/paper：https://arxiv.org/abs/2108.11458
- 代码/code：None

**Self-supervised Neural Networks for Spectral Snapshot Compressive Imaging**

- 论文/paper：https://arxiv.org/abs/2108.12654
- 代码/code：https://github.com/mengziyi64/CASSI-Self-Supervised

**Self-Supervised Visual Representations Learning by Contrastive Mask Prediction**

- 论文/paper：https://arxiv.org/abs/2108.07954
- 代码/code：None

**Self-Supervised Video Representation Learning with Meta-Contrastive Network**

- 论文/paper：https://arxiv.org/abs/2108.08426
- 代码/code：None

**SSH: A Self-Supervised Framework for Image Harmonization**

- 论文/paper：https://arxiv.org/abs/2108.06805
- 代码/code：https://github.com/VITA-Group/SSHarmonization

[返回目录/back](#Contents)

<a name="Semi-Supervised"></a>

## Semi Supervised

**Trash to Treasure: Harvesting OOD Data with Cross-Modal Matching for Open-Set Semi-Supervised Learning**

- 论文/paper：https://arxiv.org/abs/2108.05617
- 代码/code：None

[返回目录/back](#Contents)

<a name="Weakly-Supervised"></a>

## Weakly Supervised

**A Weakly Supervised Amodal Segmenter with Boundary Uncertainty Estimation**

- 论文/paper：https://arxiv.org/abs/2108.09897v1
- 代码/code：None

**Foreground-Action Consistency Network for Weakly Supervised Temporal Action Localization**

- 论文/paper：https://arxiv.org/abs/2108.06524
- 代码/code：https://github.com/LeonHLJ/FAC-Net

[返回目录/back](#Contents)

<a name="Active-Learning"></a>

## Active Learning

**Influence Selection for Active Learning**

- 论文/paper：https://arxiv.org/abs/2108.09331v1
- 代码/code：None

[返回目录/back](#Contents)

<a name="Action-Detection"></a>

## Action Detection

**Class Semantics-based Attention for Action Detection**

- 论文/paper：https://arxiv.org/abs/2109.02613
- 代码/code：None

[返回目录/back](#Contents)

<a name="HumanActions"></a>

## Action Recognition

**Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition**

- 论文/paper：https://arxiv.org/abs/2107.12213
- 代码/code：https://github.com/Uason-Chen/CTR-GCN

**Elaborative Rehearsal for Zero-shot Action Recognition**

- 论文/paper：https://arxiv.org/abs/2108.02833

- 代码/code： https://github.com/DeLightCMU/ElaborativeRehearsal

:heavy_check_mark:**FineAction: A Fined Video Dataset for Temporal Action Localization**

- 论文/paper：https://arxiv.org/abs/2105.11107 | [主页/Homepage](https://deeperaction.github.io/fineaction/)

- 代码/code： None

:heavy_check_mark:**MultiSports: A Multi-Person Video Dataset of Spatio-Temporally Localized Sports Actions**

- 论文/paper：https://arxiv.org/abs/2105.07404 | [主页/Homepage](https://deeperaction.github.io/multisports/)
- 代码/code：https://github.com/MCG-NJU/MultiSports/

**Spatio-Temporal Dynamic Inference Network for Group Activity Recognition**

- 论文/paper：https://arxiv.org/abs/2108.11743
- 代码/code：None

**Video Pose Distillation for Few-Shot, Fine-Grained Sports Action Recognition**

- 论文/paper：https://arxiv.org/abs/2109.01305
- 代码/code：None

[返回目录/back](#Contents)

<a name=" TemporalActionLocalization"></a>

## 时序行为检测 / Temporal Action Localization

**Enriching Local and Global Contexts for Temporal Action Localization**

- 论文/paper：https://arxiv.org/abs/2104.02330
- 代码/code：None

**Boundary-sensitive Pre-training for Temporal Localization in Videos**

- 论文/paper：https://arxiv.org/abs/2011.10830
- 代码/code：None

[返回目录/back](#Contents)

<a name="SignLanguageRecognition"></a>

## 手语识别/Sign Language Recognition

**Visual Alignment Constraint for Continuous Sign Language Recognition**

- 论文/paper：https://arxiv.org/abs/2104.02330
- 代码/code： https://github.com/Blueprintf/VAC_CSLR

[返回目录/back](#Contents)

<a name="Hand-Pose-Estimation"></a>

## Hand Pose Estimation

**HandFoldingNet: A 3D Hand Pose Estimation Network Using Multiscale-Feature Guided Folding of a 2D Hand Skeleton**

- 论文/paper：https://arxiv.org/abs/2108.05545
- 代码/code：https://github.com/cwc1260/HandFold

[返回目录/back](#Contents)

<a name="PoseEstimation"></a>

# Pose Estimation

## 2D Pose Estimation

**Hand-Object Contact Consistency Reasoning for Human Grasps Generation**

- 论文/paper：https://arxiv.org/pdf/2104.03304.pdf | [主页/Homepage](https://hwjiang1510.github.io/GraspTTA/)
- 代码/code： None

**Human Pose Regression with Residual Log-likelihood Estimation** Oral

- 论文/paper：https://arxiv.org/abs/2107.11291| [主页/Homepage](https://jeffli.site/res-loglikelihood-regression/)
- 代码/code：https://github.com/Jeff-sjtu/res-loglikelihood-regression

**Online Knowledge Distillation for Efficient Pose Estimation**

- 论文/paper：https://arxiv.org/abs/2108.02092
- 代码/code： None

**TransPose: Keypoint Localization via Transformer**

- 论文/paper：https://arxiv.org/abs/2012.14214
- 代码/code：https://github.com/yangsenius/TransPose

## 3D Pose Estimation

**EventHPE: Event-based 3D Human Pose and Shape Estimation**

- 论文/paper：https://arxiv.org/abs/2108.06819
- 代码/code：None

**DECA: Deep viewpoint-Equivariant human pose estimation using Capsule Autoencoders**（Oral）

- 论文/paper：https://arxiv.org/abs/2108.08557
- 代码/code：https://github.com/mmlab-cv/DECA

**FrankMocap: A Monocular 3D Whole-Body Pose Estimation System via Regression and Integration**

- 论文/paper：https://arxiv.org/abs/2108.06428
- 代码/code：None

**Learning Skeletal Graph Neural Networks for Hard 3D Pose Estimation**

- 论文/paper：https://arxiv.org/abs/2108.07181
- 代码/code：https://github.com/ailingzengzzz/Skeletal-GNN

**Probabilistic-Monocular-3D-Human-Pose-Estimation-with-Normalizing-Flows**

- 论文/paper：https://arxiv.org/abs/2107.13788
- 代码/code： https://github.com/twehrbein/Probabilistic-Monocular-3D-Human-Pose-Estimation-with-Normalizing-Flows

**PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop**

- 论文/paper：https://arxiv.org/abs/2103.16507 | [主页/Homepage](https://hongwenzhang.github.io/pymaf/)
- 代码/code： https://github.com/HongwenZhang/PyMAF

**Unsupervised 3D Pose Estimation for Hierarchical Dance Video Recognition**

- 论文/paper：https://arxiv.org/abs/2109.09166
- 代码/code：None

[返回目录/back](#Contents)

<a name="6D-Object-Pose-Estimation"></a>

## 6D Object Pose Estimation

**RePOSE: Real-Time Iterative Rendering and Refinement for 6D Object Pose Estimation**

- 论文/paper：https://arxiv.org/abs/2104.00633
- 代码/code：https://github.com/sh8/RePOSE

**SO-Pose: Exploiting Self-Occlusion for Direct 6D Pose Estimation**

- 论文/paper：https://arxiv.org/abs/2108.08367
- 代码/code：None

**StereOBJ-1M: Large-scale Stereo Image Dataset for 6D Object Pose Estimation**

- 论文/paper：https://arxiv.org/abs/2109.10115
- 代码/code：None

[返回目录/back](#Contents)

<a name="Human-Reconstruction"></a>

## Human Reconstruction

**ARCH++: Animation-Ready Clothed Human Reconstruction Revisited**

- 论文/paper：https://arxiv.org/abs/2108.07845
- 代码/code：None

**imGHUM: Implicit Generative Models of 3D Human Shape and Articulated Pose**

- 论文/paper：https://arxiv.org/abs/2108.10842
- 代码/code：None

**Learning Motion Priors for 4D Human Body Capture in 3D Scenes** (Oral)

- 论文/paper：https://arxiv.org/abs/2108.10399 |[主页/Homepage](https://sanweiliti.github.io/LEMO/LEMO.html)
- 代码/code：https://github.com/sanweiliti/LEMO

**Physics-based Human Motion Estimation and Synthesis from Videos**

- 论文/paper：https://arxiv.org/abs/2109.09913
- 代码/code：None

**Probabilistic Modeling for Human Mesh Recovery**

- 论文/paper：https://arxiv.org/abs/2108.11944
- 代码/code：https://www.seas.upenn.edu/~nkolot/projects/prohmr/

[返回目录/back](#Contents)

<a name="3D-Scene-Understanding"></a>

## 3D Scene Understanding

**DeepPanoContext: Panoramic 3D Scene Understanding with Holistic Scene Context Graph and Relation-based Optimization** (Oral)

- 论文/paper：https://arxiv.org/abs/2108.10743 |[主页/Homepage](https://chengzhag.github.io/publication/dpc/)
- 代码/code：None

**Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal Estimation**  [oral]

- 论文/paper：https://arxiv.org/abs/2109.09881
- 代码/code：https://github.com/baegwangbin/surface_normal_uncertainty

[返回目录/back](#Contents)

<a name="Face-Recognition"></a>

# Face Recognition

**Masked Face Recognition Challenge: The InsightFace Track Report**

- 论文/paper：https://arxiv.org/abs/2108.08191
- 代码/code：https://github.com/deepinsight/insightface/tree/master/challenges/iccv21-mfr

**Masked Face Recognition Challenge: The WebFace260M Track Report**

- 论文/paper：https://arxiv.org/abs/2108.07189
- 代码/code：None

**PASS: Protected Attribute Suppression System for Mitigating Bias in Face Recognition**

- 论文/paper：https://arxiv.org/abs/2108.03764
- 代码/code：None

**Rethinking Common Assumptions to Mitigate Racial Bias in Face Recognition Datasets**

- 论文/paper：https://arxiv.org/abs/2109.03229
- 代码/code：https://github.com/j-alex-hanson/rethinking-race-face-datasets

**SynFace: Face Recognition with Synthetic Data**

- 论文/paper：https://arxiv.org/abs/2108.07960
- 代码/code：None

**Unravelling the Effect of Image Distortions for Biased Prediction of Pre-trained Face Recognition Models**

- 论文/paper：https://arxiv.org/abs/2108.06581
- 代码/code：None

[返回目录/back](#Contents)

<a name="Face-Alignment"></a>

## Face Alignment

**ADNet: Leveraging Error-Bias Towards Normal Direction in Face Alignment**

- 论文/paper：https://arxiv.org/abs/2109.05721
- 代码/code：None

[返回目录/back](#Contents)

<a name="Facial-Editing"></a>

## Facial Editing 

**Talk-to-Edit: Fine-Grained Facial Editing via Dialog**

- 论文/paper：https://arxiv.org/abs/2109.04425 | [主页/Homepage](https://www.mmlab-ntu.com/project/talkedit/)
- 代码/code：https://github.com/yumingj/Talk-to-Edit

[返回目录/back](#Contents)

<a name="FaceReconstruction"></a>

# Face Reconstruction

**Towards High Fidelity Monocular Face Reconstruction with Rich Reflectance using Self-supervised Learning and Ray Tracing**

- 论文/paper：https://arxiv.org/abs/2103.15432

- 代码/code：None

[返回目录/back](#Contents)

<a name="Facial-Expression-Recognition"></a>

## Facial Expression Recognition

**TransFER: Learning Relation-aware Facial Expression Representations with Transformers**

- 论文/paper：https://arxiv.org/abs/2108.11116
- 代码/code：None

**Understanding and Mitigating Annotation Bias in Facial Expression Recognition**

- 论文/paper：https://arxiv.org/abs/2108.08504
- 代码/code：None

[返回目录/back](#Contents)

<a name="Re-Identification"></a>

## 行人重识别/Re-Identification

**ASMR: Learning Attribute-Based Person Search with Adaptive Semantic Margin Regularizer**

- 论文/paper：https://arxiv.org/abs/2108.04533
- 代码/code：None

**Counterfactual Attention Learning for Fine-Grained Visual Categorization and Re-identification**

- 论文/paper：https://arxiv.org/abs/2108.08728
- 代码/code：https://github.com/raoyongming/CAL

**IDM: An Intermediate Domain Module for Domain Adaptive Person Re-ID** Oral

- 论文/paper：https://arxiv.org/abs/2108.02413
- 代码/code：https://github.com/SikaStar/IDM

**Learning by Aligning: Visible-Infrared Person Re-identification using Cross-Modal Correspondences**

- 论文/paper：https://arxiv.org/abs/2108.07422
- 代码/code：None

**Learning Instance-level Spatial-Temporal Patterns for Person Re-identification**

- 论文/paper：https://arxiv.org/abs/2108.00171

- 代码/code：https://github.com/RenMin1991/cleaned-DukeMTMC-reID/

**Learning Compatible Embeddings**

- 论文/paper：None
- 代码/code：https://github.com/IrvingMeng/LCE

**Multi-Expert Adversarial Attack Detection in Person Re-identification Using Context Inconsistency**

- 论文/paper：https://arxiv.org/abs/2108.09891v1
- 代码/code：None

**Towards Discriminative Representation Learning for Unsupervised Person Re-identification**

- 论文/paper：https://arxiv.org/abs/2108.03439
- 代码/code：None

**TransReID: Transformer-based Object Re-Identification**

- 论文/paper：https://arxiv.org/abs/2102.04378
- 代码/code：https://github.com/heshuting555/TransReID

**Video-based Person Re-identification with Spatial and Temporal Memory Networks**

- 论文/paper：https://arxiv.org/abs/2108.09039
- 代码/code：None

**Weakly Supervised Person Search with Region Siamese Networks**

- 论文/paper：https://arxiv.org/abs/2109.06109
- 代码/code：None

[返回目录/back](#Contents)

<a name="Vehicle Re-identification"></a>

## Vehicle Re-identification

**Heterogeneous Relational Complement for Vehicle Re-identification**

- 论文/paper：https://arxiv.org/abs/2109.07894
- 代码/code：None

[返回目录/back](#Contents)

<a name="Pedestrian-Detection"></a>

## Pedestrian Detection

**MOTSynth: How Can Synthetic Data Help Pedestrian Detection and Tracking?**

- 论文/paper：https://arxiv.org/abs/2108.09518v1
- 代码/code：None

**Spatial and Semantic Consistency Regularizations for Pedestrian Attribute Recognition**

- 论文/paper：https://arxiv.org/abs/2109.05686
- 代码/code：None

[返回目录/back](#Contents)

<a name="Crowd-Counting"></a>

## 人群计数 /Crowd Counting

**Rethinking Counting and Localization in Crowds:A Purely Point-Based Framework** (Oral)

- 论文/paper：https://arxiv.org/abs/2107.12746
- 代码/code：https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet

**Uniformity in Heterogeneity:Diving Deep into Count Interval Partition for Crowd Counting**

- 论文/paper：https://arxiv.org/abs/2107.12619
- 代码/code：https://github.com/TencentYoutuResearch/CrowdCounting-UEPNet

**Variational Attention: Propagating Domain-Specific Knowledge for Multi-Domain Learning in Crowd Counting**

- 论文/paper：https://arxiv.org/abs/2108.08023
- 代码/code：None

[返回目录/back](#Contents)

<a name="MotionForecasting"></a>

## Motion Forecasting

**Generating Smooth Pose Sequences for Diverse Human Motion Prediction**

- 论文/paper：https://arxiv.org/abs/2108.08422
- 代码/code：https://github.com/wei-mao-2019/gsps

**MSR-GCN: Multi-Scale Residual Graph Convolution Networks for Human Motion Prediction**

- 论文/paper：https://arxiv.org/abs/2108.07152
- 代码/code：https://github.com/Droliven/MSRGCN

**RAIN: Reinforced Hybrid Attention Inference Network for Motion Forecasting**

- 论文/paper：https://arxiv.org/abs/2108.01316 | [主页/Homepage](https://jiachenli94.github.io/publications/RAIN/)
- 代码/code：None

**Skeleton-Graph: Long-Term 3D Motion Prediction From 2D Observations Using Deep Spatio-Temporal Graph CNNs**

- 论文/paper：https://arxiv.org/abs/2109.10257
- 代码/code：https://github.com/abduallahmohamed/Skeleton-Graph

[返回目录/back](#Contents)

<a name="Pedestrian-Trajectory-Prediction"></a>

## Pedestrian Trajectory Prediction

**DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets**

- 论文/paper：https://arxiv.org/abs/2108.09640v1
- 代码/code：None

**MG-GAN: A Multi-Generator Model Preventing Out-of-Distribution Samples in Pedestrian Trajectory Prediction**

- 论文/paper：https://arxiv.org/abs/2108.09274
- 代码/code：https://github.com/selflein/MG-GAN

[返回目录/back](#Contents)

<a name="Face-Anti-spoofing"></a>

## Face-Anti-spoofing

**CL-Face-Anti-spoofing**

- 论文/paper：None
- 代码/code：https://github.com/xxheyu/CL-Face-Anti-spoofing

**3D High-Fidelity Mask Face Presentation Attack Detection Challenge**

- 论文/paper：https://arxiv.org/abs/2108.06968
- 代码/code：None

**Exploring Temporal Coherence for More General Video Face Forgery Detection**

- 论文/paper：https://arxiv.org/abs/2108.06693
- 代码/code：None

[返回目录/back](#Contents)

<a name="deepfake"></a>

## deepfake

- 论文/paper：https://arxiv.org/abs/2107.14480 | [Dataset](https://sites.google.com/view/ltnghia/research/openforensics)
- 代码/code：None

[返回目录/back](#Contents)

<a name="AdversarialAttacks"></a>

## 对抗攻击/ Adversarial Attacks

**A Hierarchical Assessment of Adversarial Severity**

- 论文/paper：https://arxiv.org/abs/2108.11785
- 代码/code：None

**AdvDrop: Adversarial Attack to DNNs by Dropping Information**

- 论文/paper：https://arxiv.org/abs/2108.09034
- 代码/code：None

**AGKD-BML: Defense Against Adversarial Attack by Attention Guided Knowledge Distillation and Bi-directional Metric Learning**

- 论文/paper：https://arxiv.org/abs/2108.06017
- 代码/code：https://github.com/hongw579/AGKD-BML

**Optical Adversarial Attack**

- 论文/paper：https://arxiv.org/abs/2108.06247
- 代码/code：None

**Sample Efficient Detection and Classification of Adversarial Attacks via Self-Supervised Embeddings**

- 论文/paper：https://arxiv.org/abs/2108.13797
- 代码/code：None

**T*k*ML-AP: Adversarial Attacks to Top-*k* Multi-Label Learning**

- 论文/paper：https://arxiv.org/abs/2108.00146
- 代码/code：None

[返回目录/back](#Contents)

<a name="Cross-Modal-Retrieval"></a>

## 跨模态检索/Cross-Modal Retrieval

**Wasserstein Coupled Graph Learning for Cross-Modal Retrieval**

- 论文/paper：None
- 代码/code：None

[返回目录/back](#Contents)

<a name="DepthEstimation"></a>

## 深度估计 / Depth Estimation

**AA-RMVSNet: Adaptive Aggregation Recurrent Multi-view Stereo Network**

- 论文/paper：https://arxiv.org/abs/2108.03824
- 代码/code：https://github.com/QT-Zhu/AA-RMVSNet

**Augmenting Depth Estimation with Geospatial Context**

- 论文/paper：https://arxiv.org/abs/2109.09879
- 代码/code：None

**Fine-grained Semantics-aware Representation Enhancement for Self-supervised Monocular Depth Estimation**   (oral)

- 论文/paper：https://arxiv.org/abs/2108.08829
- 代码/code：https://github.com/hyBlue/FSRE-Depth

**Motion Basis Learning for Unsupervised Deep Homography Estimationwith Subspace Projection**

- 论文/paper：None
- 代码/code：https://github.com/NianjinYe/Motion-Basis-Homography

**Regularizing Nighttime Weirdness: Efficient Self-supervised Monocular Depth Estimation in the Dark**

- 论文/paper：https://arxiv.org/abs/2108.03830
- 代码/code：None

**Revisiting Stereo Depth Estimation From a Sequence-to-Sequence Perspective with Transformers**

- 论文/paper：https://arxiv.org/abs/2011.02910
- 代码/code：https://github.com/mli0603/stereo-transformer

**Self-supervised Monocular Depth Estimation for All Day Images using Domain Separation**

- 论文/paper：https://arxiv.org/abs/2108.07628
- 代码/code：None

**SLIDE: Single Image 3D Photography with Soft Layering and Depth-aware Inpainting** (Oral)

- 论文/paper：https://arxiv.org/abs/2109.01068 | [主页/Homepage](https://varunjampani.github.io/slide/)
- 代码/code：None

**StructDepth: Leveraging the structural regularities for self-supervised indoor depth estimation**

- 论文/paper：https://arxiv.org/abs/2108.08574
- 代码/code：https://github.com/SJTU-ViSYS/StructDepth

[返回目录/back](#Contents)

<a name="VideoFrameInterpolation"></a>

## 视频插帧/Video Frame Interpolation

**Asymmetric Bilateral Motion Estimation for Video Frame Interpolation**

- 论文/paper：https://arxiv.org/abs/2108.06815

- 代码/code：https://github.com/JunHeum/ABME

:heavy_check_mark:**XVFI: eXtreme Video Frame Interpolation**(Oral)

- 论文/paper：https://arxiv.org/abs/2103.16206

- 代码/code： https://github.com/JihyongOh/XVFI

[返回目录/back](#Contents)

<a name="Video-Reasoning"></a>

## Video Reasoning

**The Multi-Modal Video Reasoning and Analyzing Competition**

- 论文/paper：https://arxiv.org/abs/2108.08344
- 代码/code：None

[返回目录/back](#Contents)

<a name="NeRF"></a>

## NeRF

**CodeNeRF: Disentangled Neural Radiance Fields for Object Categories**

- 论文/paper：https://arxiv.org/abs/2109.01750 | [主页/Homepage](https://sites.google.com/view/wbjang/home/codenerf)
- 代码/code：https://github.com/wayne1123/code-nerf

**GNeRF: GAN-based Neural Radiance Field without Posed Camera**

- 论文/paper：https://arxiv.org/abs/2103.15606 | [主页/Homepage](https://nvlabs.github.io/GANcraft/)
- 代码/code：https://github.com/MQ66/gnerf

**In-Place Scene Labelling and Understanding with Implicit Scene Representation** (Oral)

- 论文/paper：https://arxiv.org/abs/2103.15875 | [主页/Homepage](https://shuaifengzhi.com/Semantic-NeRF/)
- 代码/code：None

**KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs**

- 论文/paper：https://arxiv.org/abs/2103.13744| [主页/Homepage](https://pengsongyou.github.io/)
- 代码/code：https://github.com/creiser/kilonerf

**Learning Object-Compositional Neural Radiance Field for Editable Scene Rendering**

- 论文/paper：https://arxiv.org/abs/2109.01847 | [主页/Homepage](https://zju3dv.github.io/object_nerf/)
- 代码/code：https://github.com/zju3dv/object_nerf

**NerfingMVS: Guided Optimization of Neural Radiance Fields for Indoor Multi-view Stereo** (Oral)

- 论文/paper：https://arxiv.org/abs/2109.01129 | [主页/Homepage](https://weiyithu.github.io/NerfingMVS/)
- 代码/code：https://github.com/weiyithu/NerfingMVS

**Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis**

- 论文/paper：https://arxiv.org/abs/2104.00677 | [主页/Homepage](https://ajayj.com/dietnerf)
- 代码/code：None

**Self-Calibrating Neural Radiance Fields**

- 论文/paper：https://arxiv.org/abs/2108.13826
- 代码/code：https://github.com/POSTECH-CVLab/SCNeRF

**UNISURF: Unifying Neural Implicit Surfaces and Radiance Fields for Multi-View Reconstruction** (Oral)

- 论文/paper：https://arxiv.org/abs/2104.10078 | [主页/Homepage](https://pengsongyou.github.io/)

- 代码/code：None

[返回目录/back](#Contents)

<a name="Shadow-Removal"></a>

## Shadow Removal

**CANet: A Context-Aware Network for Shadow Removal**

- 论文/paper：https://arxiv.org/abs/2108.09894v1
- 代码/code：None

[返回目录/back](#Contents)

<a name="ImageRetrieval"></a>

## Image Retrieval

**DOLG: Single-Stage Image Retrieval with Deep Orthogonal Fusion of Local and Global Features**

- 论文/paper：https://arxiv.org/abs/2108.02927
- 代码/code：None

**Image Retrieval on Real-life Images with Pre-trained Vision-and-Language Models**

- 论文/paper：https://arxiv.org/abs/2108.04024
- 代码/code：https://github.com/Cuberick-Orion/CIRR

**Self-supervised Product Quantization for Deep Unsupervised Image Retrieval**

- 论文/paper：https://arxiv.org/abs/2109.02244
- 代码/code：None

[返回目录/back](#Contents)

<a name="Super-Resolution"></a>

## 超分辨/Super-Resolution

**Designing a Practical Degradation Model for Deep Blind Image Super-Resolution**

- 论文/paper：https://arxiv.org/pdf/2103.14006.pdf
- 代码/code：https://github.com/cszn/BSRGAN

**Dual-Camera Super-Resolution with Aligned Attention Modules**

- 论文/paper：https://arxiv.org/abs/2109.01349

- 代码/code：None

**Generalized Real-World Super-Resolution through Adversarial Robustness**

- 论文/paper：https://arxiv.org/abs/2108.11505

- 代码/code：None

**Learning for Scale-Arbitrary Super-Resolution from Scale-Specific Networks**

- 论文/paper：https://arxiv.org/abs/2004.03791

- 代码/code：https://github.com/LongguangWang/ArbSR

**Overfitting the Data: Compact Neural Video Delivery via Content-aware Feature Modulation**

- 论文/paper：None

- 代码/code： https://github.com/Anonymous-iccv2021-paper3163/CaFM-Pytorch

[返回目录/back](#Contents)

<a name="ImageReconstruction"></a>

## Image Reconstruction

**Equivariant Imaging: Learning Beyond the Range Space** (Oral)

- 论文/paper：https://arxiv.org/abs/2103.14756
- 代码/code：https://github.com/edongdongchen/EI

**Spatially-Adaptive Image Restoration using Distortion-Guided Networks**

- 论文/paper：https://arxiv.org/abs/2108.08617
- 代码/code：https://github.com/human-analysis/spatially-adaptive-image-restoration

[返回目录/back](#Contents)

<a name="Deblurring"></a>

## Image Deblurring

**Single Image Defocus Deblurring Using Kernel-Sharing Parallel Atrous Convolutions**

- 论文/paper：https://arxiv.org/abs/2108.09108
- 代码/code：None

[返回目录/back](#Contents)

<a name="ImageDenoising"></a>

## Image Denoising

**Deep Reparametrization of Multi-Frame Super-Resolution and Denoising** (Oral)

- 论文/paper：https://arxiv.org/abs/2108.08286
- 代码/code：None

**Eformer: Edge Enhancement based Transformer for Medical Image Denoising**

- 论文/paper：https://arxiv.org/abs/2109.08044
- 代码/code：None

**ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models **Oral

- 论文/paper：https://arxiv.org/abs/2108.02938
- 代码/code：None

**Rethinking Deep Image Prior for Denoising**

- 论文/paper：https://arxiv.org/abs/2108.12841
- 代码/code：None

[返回目录/back](#Contents)

<a name="ImageDesnowing"></a>

## Image Desnowing

**ALL Snow Removed: Single Image Desnowing Algorithm Using Hierarchical Dual-tree Complex Wavelet Representation and Contradict Channel Loss**

- 论文/paper：None
- 代码/code：https://github.com/weitingchen83/ICCV2021-Single-Image-Desnowing-HDCWNet

[返回目录/back](#Contents)

<a name="ImageEnhancement"></a>

## Image Enhancement

**Gap-closing Matters: Perceptual Quality Assessment and Optimization of Low-Light Image Enhancement**

- 论文/paper：None
- 代码/code：https://github.com/Baoliang93/Gap-closing-Matters

**Real-time Image Enhancer via Learnable Spatial-aware 3D Lookup Tables**

- 论文/paper：https://arxiv.org/abs/2108.08697
- 代码/code：None

[返回目录/back](#Contents)

<a name="Image-Matching"></a>

## Image Matching

**Effect of Parameter Optimization on Classical and Learning-based Image Matching Methods**

- 论文/paper：https://arxiv.org/abs/2108.08179
- 代码/code：None

**Viewpoint Invariant Dense Matching for Visual Geolocalization**

- 论文/paper：https://arxiv.org/abs/2109.09827
- 代码/code：https://github.com/gmberton/geo_warp

[返回目录/back](#Contents)

<a name="Image-Quality"></a>

## Image Quality 

**MUSIQ: Multi-scale Image Quality Transformer**

- 论文/paper：https://arxiv.org/abs/2108.05997
- 代码/code：None

[返回目录/back](#Contents)

<a name="Image-Compression"></a>

## Image Compression

**Dense Deep Unfolding Network with 3D-CNN Prior for Snapshot Compressive Imaging**

- 论文/paper：https://arxiv.org/abs/2109.06548
- 代码/code：https://github.com/jianzhangcs/SCI3D

**Variable-Rate Deep Image Compression through Spatially-Adaptive Feature Transform**

- 论文/paper：https://arxiv.org/abs/2108.09551v1
- 代码/code：https://github.com/micmic123/QmapCompression

[返回目录/back](#Contents)

<a name="Image-Restoration"></a>

## Image Restoration

**Dynamic Attentive Graph Learning for Image Restoration**

- 论文/paper：https://arxiv.org/abs/2109.06620
- 代码/code：https://github.com/jianzhangcs/DAGL

[返回目录/back](#Contents)

<a name="Image-Inpainting"></a>

## Image Inpainting

**Image Inpainting via Conditional Texture and Structure Dual Generation**

- 论文/paper：https://arxiv.org/abs/2108.09760v1
- 代码/code：https://github.com/Xiefan-Guo/CTSDG

[返回目录/back](#Contents)

<a name="VideoInpainting"></a>

## Video Inpainting

**FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting**

- 论文/paper：https://arxiv.org/abs/2108.01912
- 代码/code：None

**Internal Video Inpainting by Implicit Long-range Propagation**

- 论文/paper：https://arxiv.org/abs/2108.01912
- 代码/code：None

**Occlusion-Aware Video Object Inpainting**

- 论文/paper：https://arxiv.org/abs/2108.06765
- 代码/code：None

[返回目录/back](#Contents)

<a name="Video-Recognition"></a>

# Video Recognition

**Searching for Two-Stream Models in Multivariate Space for Video Recognition**

- 论文/paper：https://arxiv.org/abs/2108.12957
- 代码/code：None

[返回目录/back](#Contents)

<a name="Visual-Question-Answering"></a>

## Visual Question Answering

**Weakly Supervised Relative Spatial Reasoning for Visual Question Answering**

- 论文/paper：https://arxiv.org/abs/2109.01934
- 代码/code：https://github.com/pratyay-banerjee/weak_sup_vqa

[返回目录/back](#Contents)

<a name="Matching"></a>

## Matching

**Multi-scale Matching Networks for Semantic Correspondence**

- 论文/paper：https://arxiv.org/abs/2108.00211
- 代码/code：None

[返回目录/back](#Contents)

<a name="Hand-object-Interaction"></a>

## 人机交互/Hand-object Interaction

:heavy_check_mark:**CPF: Learning a Contact Potential Field to Model the Hand-object Interaction**

- 论文/paper：https://arxiv.org/abs/2012.00924
- 代码/code：https://github.com/lixiny/CPF

**Exploiting Scene Graphs for Human-Object Interaction Detection**

- 论文/paper：https://arxiv.org/abs/2108.08584
- 代码/code：https://github.com/ht014/SG2HOI

**Spatially Conditioned Graphs for Detecting Human–Object Interactions**

- 论文/paper：https://arxiv.org/pdf/2012.06060.pdf
- 代码/code：https://github.com/fredzzhang/spatially-conditioned-graphs

 [返回目录/back](#Contents)

<a name="GazeEstimation"></a>

## 视线估计/Gaze Estimation

**Generalizing Gaze Estimation with Outlier-guided Collaborative Adaptation**

- 论文/paper：https://arxiv.org/abs/2107.13780 | [主页/Homepage](https://liuyunfei.net/publication/iccv2021_pnp-ga/)

- 代码/code：https://github.com/DreamtaleCore/PnP-GA

 [返回目录/back](#Contents)

<a name="Contrastive-Learning"></a>

## Contrastive-Learning

**Improving Contrastive Learning by Visualizing Feature Transformation**

- 论文/paper：https://arxiv.org/abs/2108.02982

- 代码/code：https://github.com/DTennant/CL-Visualizing-Feature-Transformation

**Social NCE: Contrastive Learning of Socially-aware Motion Representations**

- 论文/paper：https://arxiv.org/abs/2012.11717

- 代码/code：https://github.com/vita-epfl/social-nce-crowdnav

**Parametric Contrastive Learning**

- 论文/paper：https://arxiv.org/abs/2107.12028

- 代码/code：https://github.com/jiequancui/Parametric-Contrastive-Learning

 [返回目录/back](#Contents)

<a name="Graph-Convolution-Networks"></a>

## Graph Convolution Networks

 **MSR-GCN: Multi-Scale Residual Graph Convolution Networks for Human Motion Prediction**

- 论文/paper：None
- 代码/code：https://github.com/Droliven/MSRGCN

 [返回目录/back](#Contents)

<a name="Compress"></a>

## 模型压缩/Compress

**GDP: Stabilized Neural Network Pruning via Gates with Differentiable Polarization**

- 论文/paper：https://arxiv.org/abs/2109.02220

- 代码/code：None

**Sub-bit Neural Networks: Learning to Compress and Accelerate Binary Neural Networks**

- 论文/paper：None

- 代码/code：https://github.com/yikaiw/SNN

 [返回目录/back](#Contents)

<a name="Quantization"></a>

## Quantization

**Cluster-Promoting Quantization with Bit-Drop for Minimizing Network Quantization Loss**

- 论文/paper：https://arxiv.org/abs/2109.02100
- 代码/code：None

**Distance-aware Quantization**

- 论文/paper：https://arxiv.org/abs/2108.06983
- 代码/code：None

**Dynamic Network Quantization for Efficient Video Inference**

- 论文/paper：https://arxiv.org/abs/2108.10394
- 代码/code：None

**Generalizable Mixed-Precision Quantization via Attribution Rank Preservation**

- 论文/paper：https://arxiv.org/abs/2108.02720
- 代码/code：https://github.com/ZiweiWangTHU/GMPQ

 [返回目录/back](#Contents)

## Knowledge Distillation

**Distilling Holistic Knowledge with Graph Neural Networks**

- 论文/paper：https://arxiv.org/abs/2108.05507
- 代码/code：https://github.com/wyc-ruiker/HKD

**Lipschitz Continuity Guided Knowledge Distillation**

- 论文/paper：https://arxiv.org/abs/2108.12905

- 代码/code：None

**G-DetKD: Towards General Distillation Framework for Object Detectors via Contrastive and Semantic-guided Feature Imitation**

- 论文/paper：https://arxiv.org/abs/2108.07482
- 代码/code：None

**Self Supervision to Distillation for Long-Tailed Visual Recognition**

- 论文/paper：https://arxiv.org/abs/2109.04075
- 代码/code：https://github.com/MCG-NJU/SSD-LT

 [返回目录/back](#Contents)

<a name="pointcloud"></a>

## 点云/Point Cloud

**A Robust Loss for Point Cloud Registration**

- 论文/paper：https://arxiv.org/abs/2108.11682

- 代码/code：None

**A Technical Survey and Evaluation of Traditional Point Cloud Clustering Methods for LiDAR Panoptic Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.09522v1

- 代码/code：None

**(Just) A Spoonful of Refinements Helps the Registration Error Go Down** Oral

- 论文/paper：https://arxiv.org/abs/2108.03257

- 代码/code：None

**ABD-Net: Attention Based Decomposition Network for 3D Point Cloud Decomposition**

- 论文/paper：https://arxiv.org/abs/2108.04221
- 代码/code：None

**AdaFit: Rethinking Learning-based Normal Estimation on Point Clouds**

- 论文/paper：https://arxiv.org/abs/2108.05836
- 代码/code：None

**Box-Aware Feature Enhancement for Single Object Tracking on Point Clouds**

- 论文/paper：https://arxiv.org/abs/2108.04728
- 代码/code：None

**CPFN: Cascaded Primitive Fitting Networks for High-Resolution Point Clouds**

- 论文/paper：https://arxiv.org/abs/2109.00113
- 代码/code：None

**DRINet: A Dual-Representation Iterative Learning Network for Point Cloud Segmentation**

- 论文/paper：https://arxiv.org/abs/2108.04023
- 代码/code：None

**Learning Inner-Group Relations on Point Clouds**

- 论文/paper：https://arxiv.org/abs/2108.12468
- 代码/code：None

**InstanceRefer: Cooperative Holistic Understanding for Visual Grounding on Point Clouds through Instance Multi-level Contextual Referring**

- 论文/paper：https://arxiv.org/pdf/2103.01128.pdf
- 代码/code：https://github.com/CurryYuan/InstanceRefer

**ME-PCN: Point Completion Conditioned on Mask Emptiness**

- 论文/paper：https://arxiv.org/abs/2108.08187
- 代码/code：None

**MVP Benchmark: Multi-View Partial Point Clouds for Completion and Registration**

- 论文/paper：None |[主页/Homepage](https://mvp-dataset.github.io/)
- 代码/code：https://github.com/paul007pl/MVP_Benchmark

**Out-of-Core Surface Reconstruction via Global *TGV* Minimization**

- 论文/paper：https://arxiv.org/abs/2107.14790
- 代码/code：None

**PICCOLO: Point Cloud-Centric Omnidirectional Localization**

- 论文/paper：https://arxiv.org/abs/2108.06545
- 代码/code：None

**PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers**  (Oral)

- 论文/paper：https://arxiv.org/abs/2108.08839
- 代码/code：https://github.com/yuxumin/PoinTr

**ReDAL: Region-based and Diversity-aware Active Learning for Point Cloud Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2107.11769
- 代码/code：None

**Sampling Network Guided Cross-Entropy Method for Unsupervised Point Cloud Registration**

- 论文/paper：https://arxiv.org/abs/2109.06619
- 代码/code：None

**SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer**

- 论文/paper：https://arxiv.org/abs/2108.04444
- 代码/code：https://github.com/AllenXiangX/SnowflakeNet

**Spatio-temporal Self-Supervised Representation Learning for 3D Point Clouds**

- 论文/paper：https://arxiv.org/abs/2109.00179
- 代码/code：None

**Towards Efficient Point Cloud Graph Neural Networks Through Architectural Simplification**

- 论文/paper：https://arxiv.org/abs/2108.06317
- 代码/code：None

**Unsupervised Learning of Fine Structure Generation for 3D Point Clouds by 2D Projection Matching**

- 论文/paper：https://arxiv.org/abs/2108.03746
- 代码/code：https://github.com/chenchao15/2D

**Unsupervised Point Cloud Pre-Training via View-Point Occlusion, Completion**

- 论文/paper：https://arxiv.org/abs/2010.01089 |[主页/Homepage](https://hansen7.github.io/OcCo/)
- 代码/code：https://github.com/hansen7/OcCo

**Vis2Mesh: Efficient Mesh Reconstruction from Unstructured Point Clouds of Large Scenes with Learned Virtual View Visibility**

- 论文/paper：https://arxiv.org/abs/2108.08378
- 代码/code：https://github.com/GDAOSU/vis2mesh

**Voxel-based Network for Shape Completion by Leveraging Edge Generation**

- 论文/paper：https://arxiv.org/abs/2108.09936v1
- 代码/code：https://github.com/xiaogangw/VE-PCN

**Walk in the Cloud: Learning Curves for Point Clouds Shape Analysis**

- 论文/paper：https://arxiv.org/abs/2105.01288v1| [主页/Homepage](https://curvenet.github.io/)

- 代码/code：https://github.com/tiangexiang/CurveNet

 [返回目录/back](#Contents)

<a name="3D-reconstruction"></a>

## 3D reconstruction

**3D Shapes Local Geometry Codes Learning with SDF**

- 论文/paper：https://arxiv.org/abs/2108.08593
- 代码/code：None

**3DIAS: 3D Shape Reconstruction with Implicit Algebraic Surfaces**

- 论文/paper：https://arxiv.org/abs/2108.08653
- 代码/code：https://myavartanoo.github.io/3dias/

**DensePose 3D: Lifting Canonical Surface Maps of Articulated Objects to the Third Dimension**

- 论文/paper：https://arxiv.org/abs/2109.00033
- 代码/code：None

**Learning Anchored Unsigned Distance Functions with Gradient Direction Alignment for Single-view Garment Reconstruction**

- 论文/paper：https://arxiv.org/abs/2108.08478
- 代码/code：None

**Pixel-Perfect Structure-from-Motion with Featuremetric Refinement**(Oral)

- 论文/paper：https://arxiv.org/abs/2108.08291
- 代码/code：https://github.com/cvg/pixel-perfect-sfm

**VolumeFusion: Deep Depth Fusion for 3D Scene Reconstruction**

- 论文/paper：https://arxiv.org/abs/2108.08623
- 代码/code：None

 [返回目录/back](#Contents)

<a name="FontGeneration"></a>

## 字体生成/Font Generation

:heavy_check_mark:**Multiple Heads are Better than One: Few-shot Font Generation with Multiple Localized Experts**

- 论文/paper：https://arxiv.org/abs/2104.00887

- 代码/code：https://github.com/clovaai/mxfont

 [返回目录/back](#Contents)

<a name="TextDetection"></a>

## 文本检测 / Text Detection

**Adaptive Boundary Proposal Network for Arbitrary Shape Text Detection**

- 论文/paper：https://arxiv.org/abs/2107.12664
- 代码/code：https://github.com/GXYM/TextBPN

 [返回目录/back](#Contents)

<a name="TextRecognition"></a>

## 文本识别 / Text Recognition

**From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network**

- 论文/paper：https://arxiv.org/abs/2108.09661v1
- 代码/code：https://github.com/wangyuxin87/VisionLAN

**Joint Visual Semantic Reasoning: Multi-Stage Decoder for Text Recognition**

- 论文/paper：https://arxiv.org/abs/2107.12090
- 代码/code：None

 [返回目录/back](#Contents)

<a name="SceneTextRecognizer"></a>

## Scene Text Recognizer

**Data Augmentation for Scene Text Recognition**

- 论文/paper：https://arxiv.org/abs/2108.06949

- 代码/code：https://github.com/roatienza/straug

**From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network**

- 论文/paper：None

- 代码/code：https://github.com/wangyuxin87/VisionLAN

 [返回目录/back](#Contents)

<a name="Autonomous-Driving"></a>

## Autonomous-Driving

**End-to-End Urban Driving by Imitating a Reinforcement Learning Coach**

- 论文/paper：https://arxiv.org/abs/2108.08265
- 代码/code：None

**FOVEA: Foveated Image Magnification for Autonomous Navigation**

- 论文/paper：https://arxiv.org/abs/2108.12102v1
- 代码/code：https://www.cs.cmu.edu/~mengtial/proj/fovea/

**Learning to drive from a world on rails**

- 论文/paper：https://arxiv.org/abs/2105.00636
- 代码/code：https://arxiv.org/abs/2105.00636

**MultiSiam: Self-supervised Multi-instance Siamese Representation Learning for Autonomous Driving**

- 论文/paper：https://arxiv.org/abs/2108.12178v1
- 代码/code：https://github.com/KaiChen1998/MultiSiam

**NEAT: Neural Attention Fields for End-to-End Autonomous Driving**

- 论文/paper：https://arxiv.org/abs/2109.04456
- 代码/code：None

**Road-Challenge-Event-Detection-for-Situation-Awareness-in-Autonomous-Driving**

- 论文/paper：None
- 代码/code：https://github.com/Trevorchenmsu/Road-Challenge-Event-Detection-for-Situation-Awareness-in-Autonomous-Driving

**Safety-aware Motion Prediction with Unseen Vehicles for Autonomous Driving**

- 论文/paper：https://arxiv.org/abs/2109.01510

- 代码/code：https://github.com/xrenaa/Safety-Aware-Motion-Prediction

 [返回目录/back](#Contents)

<a name="Visdrone_detection"></a>

## Visdrone_detection

**ICCV2021_Visdrone_detection**

- 论文/paper：None

- 代码/code：https://github.com/Gumpest/ICCV2021_Visdrone_detection

 [返回目录/back](#Contents)

<a name="AnomalyDetection"></a>

##  Anomaly Detection

**DRÆM -- A discriminatively trained reconstruction embedding for surface anomaly detection**

- 论文/paper：https://arxiv.org/abs/2108.07610

- 代码/code：None

**Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning**

- 论文/paper：https://arxiv.org/pdf/2101.10030.pdf

- 代码/code：https://github.com/tianyu0207/RTFM

<a name="Others"></a>

## 其他/Others

**Cross-Camera Convolutional Color Constancy**

- 论文/paper：https://arxiv.org/abs/2011.11164

- 代码/code：https://github.com/mahmoudnafifi/C5

**Learnable Boundary Guided Adversarial Training**

- 论文/paper：https://arxiv.org/abs/2011.11164

- 代码/code：https://github.com/FPNAS/LBGAT

**Prior-Enhanced network with Meta-Prototypes (PEMP)**

- 论文/paper：None
- 代码/code：https://github.com/PaperSubmitAAAA/ICCV2021-2337

**MDETR -- Modulated Detection for End-to-End Multi-Modal Understanding**

- 论文/paper：https://arxiv.org/abs/2104.12763 | [主页/Homepage](https://ashkamath.github.io/mdetr_page/)
- 代码/code：https://github.com/ashkamath/mdetr

**Generalized-Shuffled-Linear-Regression** （Oral）

- 论文/paper：https://drive.google.com/file/d/1Qu21VK5qhCW8WVjiRnnBjehrYVmQrDNh/view
- 代码/code：https://github.com/SILI1994/Generalized-Shuffled-Linear-Regression

**VLGrammar: Grounded Grammar Induction of Vision and Language**

- 论文/paper：https://arxiv.org/abs/2103.12975
- 代码/code：https://github.com/evelinehong/VLGrammar

**A New Journey from SDRTV to HDRTV**

- 论文/paper：None
- 代码/code：https://github.com/chxy95/HDRTVNet

**IICNet: A Generic Framework for Reversible Image Conversion**

- 论文/paper：None
- 代码/code：https://github.com/felixcheng97/IICNet

**Structure-Preserving Deraining with Residue Channel Prior Guidance**

- 论文/paper：None
- 代码/code：https://github.com/Joyies/SPDNet

**Learning with Noisy Labels via Sparse Regularization**

- 论文/paper：https://arxiv.org/abs/2108.00192
- 代码/code：https://github.com/hitcszx/lnl_sr

**Neural Strokes: Stylized Line Drawing of 3D Shapes**

- 论文/paper：None
- 代码/code：https://github.com/DifanLiu/NeuralStrokes

**COOKIE: Contrastive Cross-Modal Knowledge Sharing Pre-training for Vision-Language Representation**

- 论文/paper：None
- 代码/code：https://github.com/kywen1119/COOKIE

**RINDNet: Edge Detection for Discontinuity in Reflectance, Illumination, Normal and Depth**

- 论文/paper：https://arxiv.org/abs/2108.00616
- 代码/code：None

**ELLIPSDF: Joint Object Pose and Shape Optimization with a Bi-level Ellipsoid and Signed Distance Function Description**

- 论文/paper：https://arxiv.org/abs/2108.00355
- 代码/code：None

**Unlimited Neighborhood Interaction for Heterogeneous Trajectory Prediction**

- 论文/paper：https://arxiv.org/abs/2108.00238
- 代码/code：None

**CanvasVAE: Learning to Generate Vector Graphic Documents**

- 论文/paper：https://arxiv.org/abs/2108.01249
- 代码/code：None

**Refining activation downsampling with SoftPool**

- 论文/paper：https://arxiv.org/abs/2101.00440
- 代码/code：https://github.com/alexandrosstergiou/SoftPool

**Aligning Latent and Image Spaces to Connect the Unconnectable**

- 论文/paper：https://arxiv.org/abs/2104.06954 | [主页/Homepage](https://universome.github.io/alis)
- 代码/code：https://github.com/universome/alis

**Unifying Nonlocal Blocks for Neural Networks**

- 论文/paper：https://arxiv.org/abs/2108.02451
- 代码/code：None

**SLAMP: Stochastic Latent Appearance and Motion Prediction**

- 论文/paper：https://arxiv.org/abs/2108.02760
- 代码/code：None

**TransForensics: Image Forgery Localization with Dense Self-Attention**

- 论文/paper：https://arxiv.org/abs/2108.03871
- 代码/code：None

**Learning Facial Representations from the Cycle-consistency of Face**

- 论文/paper：https://arxiv.org/abs/2108.03427
- 代码/code：https://github.com/JiaRenChang/FaceCycle

**NASOA: Towards Faster Task-oriented Online Fine-tuning with a Zoo of Models**

- 论文/paper：https://arxiv.org/abs/2108.03434
- 代码/code：None

**Impact of Aliasing on Generalization in Deep Convolutional Networks**

- 论文/paper：https://arxiv.org/abs/2108.03489
- 代码/code：None

**Learning Canonical 3D Object Representation for Fine-Grained Recognition**

- 论文/paper：https://arxiv.org/abs/2108.04628
- 代码/code：None

**UniNet: A Unified Scene Understanding Network and Exploring Multi-Task Relationships through the Lens of Adversarial Attacks**

- 论文/paper：https://arxiv.org/abs/2108.04584
- 代码/code：None

**SUNet: Symmetric Undistortion Network for Rolling Shutter Correction**

- 论文/paper：https://arxiv.org/abs/2108.04775
- 代码/code：None

**Learning to Cut by Watching Movies**

- 论文/paper：https://arxiv.org/abs/2108.04294
- 代码/code：https://github.com/PardoAlejo/LearningToCut

**Continual Neural Mapping: Learning An Implicit Scene Representation from Sequential Observations**

- 论文/paper：https://arxiv.org/abs/2108.05851
- 代码/code：None

**Towers of Babel: Combining Images, Language, and 3D Geometry for Learning Multimodal Vision**

- 论文/paper：https://arxiv.org/abs/2108.05863 |[主页/Homepage](https://www.cs.cornell.edu/projects/babel/)
- 代码/code：https://github.com/tgxs002/wikiscenes

**Towards Interpretable Deep Metric Learning with Structural Matching**

- 论文/paper：https://arxiv.org/abs/2108.05889
- 代码/code：https://github.com/wl-zhao/DIML

**m-RevNet: Deep Reversible Neural Networks with Momentum**

- 论文/paper：https://arxiv.org/abs/2108.05862
- 代码/code：None

**DiagViB-6: A Diagnostic Benchmark Suite for Vision Models in the Presence of Shortcut and Generalization Opportunities**

- 论文/paper：https://arxiv.org/abs/2108.05779
- 代码/code：None

**perf4sight: A toolflow to model CNN training performance on Edge GPUs**

- 论文/paper：https://arxiv.org/abs/2108.05580
- 代码/code：None

**MT-ORL: Multi-Task Occlusion Relationship Learning**

- 论文/paper：https://arxiv.org/abs/2108.05722
- 代码/code：https://github.com/fengpanhe/MT-ORL

**ProAI: An Efficient Embedded AI Hardware for Automotive Applications - a Benchmark Study**

- 论文/paper：https://arxiv.org/abs/2108.05170
- 代码/code：None

**SPACE: A Simulator for Physical Interactions and Causal Learning in 3D Environments**

- 论文/paper：https://arxiv.org/abs/2108.06180
- 代码/code：https://github.com/jiafei1224/SPACE

**CODEs: Chamfer Out-of-Distribution Examples against Overconfidence Issue**

- 论文/paper：https://arxiv.org/abs/2108.06024
- 代码/code：None

**Towards Real-World Prohibited Item Detection: A Large-Scale X-ray Benchmark**

- 论文/paper：https://arxiv.org/abs/2108.07020
- 代码/code：None

**Pixel Difference Networks for Efficient Edge Detection**

- 论文/paper：https://arxiv.org/abs/2108.07009
- 代码/code：https://github.com/zhuoinoulu/pidinet

**Online Continual Learning For Visual Food Classification**

- 论文/paper：https://arxiv.org/abs/2108.06781
- 代码/code：None

**DICOM Imaging Router: An Open Deep Learning Framework for Classification of Body Parts from DICOM X-ray Scans**

- 论文/paper：https://arxiv.org/abs/2108.06490 |[主页/Homepage](https://vindr.ai/)
- 代码/code：None

**PIT: Position-Invariant Transform for Cross-FoV Domain Adaptation**

- 论文/paper：https://arxiv.org/abs/2108.07142
- 代码/code：https://github.com/sheepooo/PIT-Position-Invariant-Transform

**Learning to Automatically Diagnose Multiple Diseases in Pediatric Chest Radiographs Using Deep Convolutional Neural Networks**

- 论文/paper：https://arxiv.org/abs/2108.06486
- 代码/code：None

**FaPN: Feature-aligned Pyramid Network for Dense Image Prediction**

- 论文/paper：https://arxiv.org/abs/2108.07058
- 代码/code：https://github.com/EMI-Group/FaPN

**Finding Representative Interpretations on Convolutional Neural Networks**

- 论文/paper：https://arxiv.org/abs/2108.06384
- 代码/code：None

**Investigating transformers in the decomposition of polygonal shapes as point collections**

- 论文/paper：https://arxiv.org/abs/2108.07533
- 代码/code：None

**Self-Supervised Pretraining and Controlled Augmentation Improve Rare Wildlife Recognition in UAV Images**

- 论文/paper：https://arxiv.org/abs/2108.07582
- 代码/code：None

**Group-aware Contrastive Regression for Action Quality Assessment**

- 论文/paper：https://arxiv.org/abs/2108.07797
- 代码/code：None

**End-to-End Dense Video Captioning with Parallel Decoding**

- 论文/paper：https://arxiv.org/abs/2108.07781
- 代码/code：https://github.com/ttengwang/PDVC

**PR-RRN: Pairwise-Regularized Residual-Recursive Networks for Non-rigid Structure-from-Motion**

- 论文/paper：https://arxiv.org/abs/2108.07506
- 代码/code：None

**Scene Designer: a Unified Model for Scene Search and Synthesis from Sketch**

- 论文/paper：https://arxiv.org/abs/2108.07353
- 代码/code：None

**Structured Outdoor Architecture Reconstruction by Exploration and Classification**

- 论文/paper：https://arxiv.org/abs/2108.07990
- 代码/code：None

**Learning RAW-to-sRGB Mappings with Inaccurately Aligned Supervision**

- 论文/paper：https://arxiv.org/abs/2108.08119
- 代码/code：https://github.com/cszhilu1998/RAW-to-sRGB

**Overfitting the Data: Compact Neural Video Delivery via Content-aware Feature Modulation**

- 论文/paper：https://arxiv.org/abs/2108.08202
- 代码/code：https://github.com/Neural-video-delivery/CaFM-Pytorch-ICCV2021

**Deep Hybrid Self-Prior for Full 3D Mesh Generation**

- 论文/paper：https://arxiv.org/abs/2108.08017
- 代码/code：None

**FACIAL: Synthesizing Dynamic Talking Face with Implicit Attribute Learning**

- 论文/paper：https://arxiv.org/abs/2108.07938
- 代码/code：None

**Thermal Image Processing via Physics-Inspired Deep Networks**

- 论文/paper：https://arxiv.org/abs/2108.07973
- 代码/code：None

**A New Journey from SDRTV to HDRTV**

- 论文/paper：https://arxiv.org/abs/2108.07978
- 代码/code：https://github.com/chxy95/HDRTVNet

**Global Pooling, More than Meets the Eye: Position Information is Encoded Channel-Wise in CNNs**

- 论文/paper：https://arxiv.org/abs/2108.07884
- 代码/code：None

**Speech Drives Templates: Co-Speech Gesture Synthesis with Learned Templates**

- 论文/paper：https://arxiv.org/abs/2108.08020
- 代码/code：None

**LOKI: Long Term and Key Intentions for Trajectory Prediction**

- 论文/paper：https://arxiv.org/abs/2108.08236
- 代码/code：None

**Stochastic Scene-Aware Motion Prediction**

- 论文/paper：https://arxiv.org/abs/2108.08284
- 代码/code：https://samp.is.tue.mpg.de/

**Exploiting Multi-Object Relationships for Detecting Adversarial Attacks in Complex Scenes**

- 论文/paper：https://arxiv.org/abs/2108.08421
- 代码/code：None

**Social Fabric: Tubelet Compositions for Video Relation Detection**

- 论文/paper：https://arxiv.org/abs/2108.08363
- 代码/code：https://github.com/shanshuo/Social-Fabric

**Causal Attention for Unbiased Visual Recognition**

- 论文/paper：https://arxiv.org/abs/2108.08782
- 代码/code：https://github.com/Wangt-CN/CaaM

**Universal Cross-Domain Retrieval: Generalizing Across Classes and Domains**

- 论文/paper：https://arxiv.org/abs/2108.08356
- 代码/code：None

**Amplitude-Phase Recombination: Rethinking Robustness of Convolutional Neural Networks in Frequency Domain**

- 论文/paper：https://arxiv.org/abs/2108.08487
- 代码/code：None

**Learning to Match Features with Seeded Graph Matching Network**

- 论文/paper：https://arxiv.org/abs/2108.08771
- 代码/code：https://github.com/vdvchen/SGMNet

**A Unified Objective for Novel Class Discovery**

- 论文/paper：https://arxiv.org/abs/2108.08536
- 代码/code：https://github.com/DonkeyShot21/UNO

**How to cheat with metrics in single-image HDR reconstruction**

- 论文/paper：https://arxiv.org/abs/2108.08713
- 代码/code：None

**Towards Understanding the Generative Capability of Adversarially Robust Classifiers** （Oral）

- 论文/paper：https://arxiv.org/abs/2108.09093
- 代码/code：None

**Airbert: In-domain Pretraining for Vision-and-Language Navigation**

- 论文/paper：https://arxiv.org/abs/2108.09105
- 代码/code：None

**Out-of-boundary View Synthesis Towards Full-Frame Video Stabilization**

- 论文/paper：https://arxiv.org/abs/2108.09041
- 代码/code：https://github.com/Annbless/OVS_Stabilization

**PatchMatch-RL: Deep MVS with Pixelwise Depth, Normal, and Visibility**

- 论文/paper：https://arxiv.org/abs/2108.08943
- 代码/code：None

**Continual Learning for Image-Based Camera Localization**

- 论文/paper：https://arxiv.org/abs/2108.09112
- 代码/code：None

**Online Continual Learning with Natural Distribution Shifts: An Empirical Study with Visual Data**

- 论文/paper：https://arxiv.org/abs/2108.09020
- 代码/code：https://github.com/IntelLabs/continuallearning

**Detecting and Segmenting Adversarial Graphics Patterns from Images**

- 论文/paper：https://arxiv.org/abs/2108.09383v1
- 代码/code：None

**TACo: Token-aware Cascade Contrastive Learning for Video-Text Alignment**

- 论文/paper：https://arxiv.org/abs/2108.09980v1
- 代码/code：None

**BlockCopy: High-Resolution Video Processing with Block-Sparse Feature Propagation and Online Policies**

- 论文/paper：https://arxiv.org/abs/2108.09376v1
- 代码/code：None

**Learning Signed Distance Field for Multi-view Surface Reconstruction**  (Oral)

- 论文/paper：https://arxiv.org/abs/2108.09964v1
- 代码/code：None

**Deep Relational Metric Learning**

- 论文/paper：https://arxiv.org/abs/2108.10026v1
- 代码/code：https://github.com/zbr17/DRML

**Ranking Models in Unlabeled New Environments**

- 论文/paper：https://arxiv.org/abs/2108.10310v1
- 代码/code：https://github.com/sxzrt/Proxy-Set

**Patch2CAD: Patchwise Embedding Learning for In-the-Wild Shape Retrieval from a Single Image**

- 论文/paper：https://arxiv.org/abs/2108.09368v1
- 代码/code：None

**LSD-StructureNet: Modeling Levels of Structural Detail in 3D Part Hierarchies**

- 论文/paper：https://arxiv.org/abs/2108.13459
- 代码/code：None

**BiaSwap: Removing dataset bias with bias-tailored swapping augmentation**

- 论文/paper：https://arxiv.org/abs/2108.10008v1
- 代码/code：None

**LoOp: Looking for Optimal Hard Negative Embeddings for Deep Metric Learning**

- 论文/paper：https://arxiv.org/abs/2108.09335v1
- 代码/code：None

**Learning of Visual Relations: The Devil is in the Tails**

- 论文/paper：https://arxiv.org/abs/2108.09668v1
- 代码/code：None

**Bridging Unsupervised and Supervised Depth from Focus via All-in-Focus Supervision**

- 论文/paper：https://arxiv.org/abs/2108.10843
- 代码/code：https://github.com/albert100121/AiFDepthNet

**Support-Set Based Cross-Supervision for Video Grounding**

- 论文/paper：https://arxiv.org/abs/2108.10576
- 代码/code：None

**Fast Robust Tensor Principal Component Analysis via Fiber CUR Decomposition**

- 论文/paper：https://arxiv.org/abs/2108.10448
- 代码/code：None

**Improving Generalization of Batch Whitening by Convolutional Unit Optimization**

- 论文/paper：https://arxiv.org/abs/2108.10629
- 代码/code：None

**CSG-Stump: A Learning Friendly CSG-Like Representation for Interpretable Shape Parsing**

- 论文/paper：https://arxiv.org/abs/2108.11305 |[主页/Homepage](https://kimren227.github.io/projects/CSGStump/)
- 代码/code：https://github.com/kimren227/CSGStumpNet

**NGC: A Unified Framework for Learning with Open-World Noisy Data**

- 论文/paper：https://arxiv.org/abs/2108.11035
- 代码/code：None

**LocTex: Learning Data-Efficient Visual Representations from Localized Textual Supervision**

- 论文/paper：https://arxiv.org/abs/2108.11950
- 代码/code：https://loctex.mit.edu/

**The Surprising Effectiveness of Visual Odometry Techniques for Embodied PointGoal Navigation**

- 论文/paper：https://arxiv.org/abs/2108.11550
- 代码/code：None

**Learning Cross-modal Contrastive Features for Video Domain Adaptation**

- 论文/paper：https://arxiv.org/abs/2108.11974v1
- 代码/code：None

**Lifelong Infinite Mixture Model Based on Knowledge-Driven Dirichlet Process**

- 论文/paper：https://arxiv.org/abs/2108.12278v1
- 代码/code：https://github.com/dtuzi123/Lifelong-infinite-mixture-model

**A Dual Adversarial Calibration Framework for Automatic Fetal Brain Biometry**

- 论文/paper：https://arxiv.org/abs/2108.12719
- 代码/code：None

**LUAI Challenge 2021 on Learning to Understand Aerial Images**

- 论文/paper：https://arxiv.org/abs/2108.13246
- 代码/code：None

**Embedding Novel Views in a Single JPEG Image**

- 论文/paper：https://arxiv.org/abs/2108.13003
- 代码/code：None

**Learning to Discover Reflection Symmetry via Polar Matching Convolution**

- 论文/paper：https://arxiv.org/abs/2108.12952
- 代码/code：None

**Deep 3D Mask Volume for View Synthesis of Dynamic Scenes**

- 论文/paper：https://arxiv.org/abs/2108.13408
- 代码/code：https://cseweb.ucsd.edu//~viscomp/projects/ICCV21Deep/

**Cross-category Video Highlight Detection via Set-based Learning**

- 论文/paper：https://arxiv.org/abs/2108.11770
- 代码/code： https://github.com/ChrisAllenMing/Cross_Category_Video_Highlight

**Overfitting the Data: Compact Neural Video Delivery via Content-aware Feature Modulation**

- 论文/paper：https://arxiv.org/abs/2108.08202
- 代码/code： https://github.com/Anonymous-iccv2021-paper3163/CaFM-Pytorch

**Sparse to Dense Motion Transfer for Face Image Animation**

- 论文/paper：https://arxiv.org/abs/2109.00471
- 代码/code：None

**SlowFast Rolling-Unrolling LSTMs for Action Anticipation in Egocentric Videos**

- 论文/paper：https://arxiv.org/abs/2109.00829
- 代码/code：None

**4D-Net for Learned Multi-Modal Alignment**

- 论文/paper：https://arxiv.org/abs/2109.01066
- 代码/code：None

**The Power of Points for Modeling Humans in Clothing**

- 论文/paper：https://arxiv.org/abs/2109.01137
- 代码/code：None

**The Functional Correspondence Problem**

- 论文/paper：https://arxiv.org/abs/2109.01097
- 代码/code：None

**On the Limits of Pseudo Ground Truth in Visual Camera Re-localisation**

- 论文/paper：https://arxiv.org/abs/2109.00524
- 代码/code：None

**Towards Learning Spatially Discriminative Feature Representations**

- 论文/paper：https://arxiv.org/abs/2109.01359
- 代码/code：None

**Learning Fast Sample Re-weighting Without Reward Data**

- 论文/paper：https://arxiv.org/abs/2109.03216
- 代码/code：https://github.com/google-research/google-research/tree/master/ieg

**CTRL-C: Camera calibration TRansformer with Line-Classification**

- 论文/paper：https://arxiv.org/abs/2109.02259
- 代码/code：None

**PR-Net: Preference Reasoning for Personalized Video Highlight Detection**

- 论文/paper：https://arxiv.org/abs/2109.01799
- 代码/code：None

**Dual Transfer Learning for Event-based End-task Prediction via Pluggable Event to Image Translation**

- 论文/paper：https://arxiv.org/abs/2109.01801
- 代码/code：None

**Learning to Generate Scene Graph from Natural Language Supervision**

- 论文/paper：https://arxiv.org/abs/2109.02227
- 代码/code：https://github.com/YiwuZhong/SGG_from_NLS

**Parsing Table Structures in the Wild**

- 论文/paper：https://arxiv.org/abs/2109.02199
- 代码/code：None

**Hierarchical Object-to-Zone Graph for Object Navigation**

- 论文/paper：https://arxiv.org/abs/2109.02066
- 代码/code：None

**Square Root Marginalization for Sliding-Window Bundle Adjustment**

- 论文/paper：https://arxiv.org/abs/2109.02182
- 代码/code：None

**YouRefIt: Embodied Reference Understanding with Language and Gesture**

- 论文/paper：https://arxiv.org/abs/2109.03413
- 代码/code：None

**Deep Hough Voting for Robust Global Registration**

- 论文/paper：https://arxiv.org/abs/2109.04310
- 代码/code：None

**IICNet: A Generic Framework for Reversible Image Conversion**

- 论文/paper：https://arxiv.org/abs/2109.04242
- 代码/code：https://github.com/felixcheng97/IICNet

**Estimating Leaf Water Content using Remotely Sensed Hyperspectral Data**

- 论文/paper：https://arxiv.org/abs/2109.02250
- 代码/code：None

**What Matters for Ad-hoc Video Search? A Large-scale Evaluation on TRECVID**

- 论文/paper：https://arxiv.org/abs/2109.01774
- 代码/code：None

**Shape-Biased Domain Generalization via Shock Graph Embeddings**

- 论文/paper：https://arxiv.org/abs/2109.05671
- 代码/code：None

**Explain Me the Painting: Multi-Topic Knowledgeable Art Description Generation**

- 论文/paper：https://arxiv.org/abs/2109.05743
- 代码/code：None

**Learning Indoor Inverse Rendering with 3D Spatially-Varying Lighting**（Oral）

- 论文/paper：https://arxiv.org/abs/2109.06061
- 代码/code：None

**Multiresolution Deep Implicit Functions for 3D Shape Representation**

- 论文/paper：https://arxiv.org/abs/2109.05591
- 代码/code：None

**Image Shape Manipulation from a Single Augmented Training Sample** （Oral）

- 论文/paper：https://arxiv.org/abs/2109.06151
- 代码/code：None

**ZFlow: Gated Appearance Flow-based Virtual Try-on with 3D Priors**

- 论文/paper：https://arxiv.org/abs/2109.07001
- 代码/code：None

**Contact-Aware Retargeting of Skinned Motion**

- 论文/paper：https://arxiv.org/abs/2109.07431
- 代码/code：None

**DisUnknown: Distilling Unknown Factors for Disentanglement Learning**

- 论文/paper：https://arxiv.org/abs/2109.08090
- 代码/code：https://github.com/stormraiser/disunknown

**FSER: Deep Convolutional Neural Networks for Speech Emotion Recognition**

- 论文/paper：https://arxiv.org/abs/2109.07916
- 代码/code：None

**A Pathology Deep Learning System Capable of Triage of Melanoma Specimens Utilizing Dermatopathologist Consensus as Ground Truth**

- 论文/paper：https://arxiv.org/abs/2109.07554
- 代码/code：None

**PIRenderer: Controllable Portrait Image Generation via Semantic Neural Rendering**

- 论文/paper：https://arxiv.org/abs/2109.08379
- 代码/code：https://github.com/RenYurui/PIRender

**The First Vision For Vitals (V4V) Challenge for Non-Contact Video-Based Physiological Estimation**

- 论文/paper：https://arxiv.org/abs/2109.10471 | [Dataset and Challenge](https://vision4vitals.github.io/)
- 代码/code：None

**FaceEraser: Removing Facial Parts for Augmented Reality**

- 论文/paper：https://arxiv.org/abs/2109.10760
- 代码/code：None

**S3VAADA: Submodular Subset Selection for Virtual Adversarial Active Domain Adaptation**

- 论文/paper：https://arxiv.org/abs/2109.08901
- 代码/code：None

**JEM++: Improved Techniques for Training JEM**

- 论文/paper：https://arxiv.org/abs/2109.09032
- 代码/code：https://github.com/sndnyang/JEMPP

**Rational Polynomial Camera Model Warping for Deep Learning Based Satellite Multi-View Stereo Matching**

- 论文/paper：https://arxiv.org/abs/2109.11121
- 代码/code：https://github.com/WHU-GPCV/SatMVS

 [返回目录/back](#Contents)



