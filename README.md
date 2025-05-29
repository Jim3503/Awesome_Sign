<div align="center">

# 🤟 Awesome Sign Language [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

**A Comprehensive Collection of Sign Language Research Papers, Datasets, and Resources**

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](https://github.com/mingji/SignLLM/pulls)
[![GitHub stars](https://img.shields.io/github/stars/mingji/SignLLM?style=social)](https://github.com/mingji/SignLLM/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/mingji/SignLLM?style=social)](https://github.com/mingji/SignLLM/network/members)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

[![Last Update](https://img.shields.io/badge/Last%20Update-2025--05-green.svg)](https://github.com/mingji/SignLLM)
[![Papers](https://img.shields.io/badge/Papers-150+-blue.svg)](https://github.com/mingji/SignLLM)
[![Datasets](https://img.shields.io/badge/Datasets-30+-orange.svg)](https://github.com/mingji/SignLLM)
[![Survey](https://img.shields.io/badge/Survey-Sign_Language-red.svg)](https://github.com/mingji/SignLLM)

---

*🌟 "From Recognition to Understanding: Bridging the Gap Between Sign Language and AI" 🌟*

</div>

<br />
<p align="center">
  <h1 align="center">Sign Language: A Comprehensive Survey</h1>
  <!-- <p align="center">
    <br />
    <a href="https://chen-yang-liu.github.io/"><strong>Chenyang Liu </strong></a>
    ·
    <a href="https://levir.buaa.edu.cn/members/index.html"><strong> Jiafan Zhang </strong></a>
    ·
    <a href="https://chenkeyan.top/"><strong> Keyan Chen </strong></a>
    ·
    <a href="https://levir.buaa.edu.cn/members/index.html"><strong> Man Wang </strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=DzwoyZsAAAAJ"><strong> Zhengxia Zou </strong></a>
    ·   
    <a href="https://scholar.google.com/citations?user=kNhFWQIAAAAJ"><strong> Zhenwei Shi*✉ </strong></a>
    
  
  </p> -->

  <p align="center">
    <!-- <a href='https://arxiv.org/abs/2412.02573'>
      <img src='https://img.shields.io/badge/arXiv-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
    </a> -->
<!--     <a href='https://ieeexplore.ieee.org/document/'>
      <img src='https://img.shields.io/badge/TPAMI-PDF-blue?style=flat&logo=IEEE&logoColor=green' alt='TPAMI PDF'>
    </a> -->
  </p>
<br />

This repo is used for recording and tracking recent Sign Language research, including Recognition, Translation, Production, and Unified Understanding. If you find any work missing or have any suggestions (papers, implementations, and other resources), feel free to [pull requests](https://github.com/ZechengLi19/Awesome-Sign-Language/pulls).

### :star: Share us a :star:
Share us a :star: if you're interested in this repo. We will continue to track relevant progress and update this repository.

### 🙌 Add Your Paper in our Repo and Survey!

- You are welcome to give us an issue or PR for your Sign Language work !!!!! We will record it for next version update of our survey

### 🥳 News

🔥🔥🔥 The repo is updating 🔥🔥🔥

[//]: # (- **2025.5.26**: The first version is available.)

### ✨ Highlight!!

✅ The first comprehensive survey for Sign Language research covering Recognition, Translation, Production, and Unified Understanding.

✅ Extensive datasets and code links are provided.

✅ Focus on the latest trends: **Gloss-free modeling**, **LLM integration**, and **Unified architectures**.

## 📖 Introduction

**Background and Motivation:**
- Sign Language research tasks include: **Recognition**, **Translation**, **Production**, and **Unified Understanding**.
- Most tasks are multimodal + multilingual problems, facing challenges like: **difficult data acquisition**, **heavy gloss dependency**, **weak semantic understanding**.
- With the advancement of **Large Language Models (like GPT)** and multimodal modeling capabilities, significant progress is being made in **gloss-free approaches**, **unified architectures**, and **multilingual transfer**.

Timeline of Sign Language works:

<!-- ![Alt Text](fig/overview_3.png) -->

## 📖 Table of Contents
- [📚 Sign Language Tasks and Methods](#methods-a-survey)
  - [Isolated Sign Recognition](#isolated-sign-recognition)
  - [Continuous Sign Recognition](#continuous-sign-recognition)
  - [Sign Language Translation](#sign-language-translation)
  - [Sign Language Production](#sign-language-production)
  - [Sign Language Retrieval](#sign-language-retrieval)
  - [Unified Understanding](#unified-understanding)
- [👨‍🏫 Large Language Models Meets Sign Language](#Large-Language-Models-Meets-Sign-Language)
  - [LLM-driven Gloss-free Translation](#LLM-driven-Gloss-free-Translation)
  - [Unified Sign Language Foundation Models](#Unified-Sign-Language-Foundation-Models)
  - [Visual-Language Alignment](#Visual-Language-Alignment)
  - ......
- [📊 Dataset](#Dataset)
  - [Isolated Word Recognition](#Dataset_1)  
  - [Continuous Recognition](#Dataset_2)
  - [Translation Tasks](#Dataset_3)
  - [Video Generation](#Dataset_4)
  - [Retrieval Tasks](#Dataset_5)
  - [Unified Understanding](#Dataset_6)
  - ......
- [💻 Others](#Others) 
- [🖊️ Citation](#Citation)
- [🐲 Contact](#Contact)

## 📚 Sign Language Tasks and Methods <a id="methods-a-survey"></a>

### Isolated Sign Recognition 
|   Time   |   Model Name    | Paper Title                                                                                                                                                                            |    Visual Encoder     |         Method          |                    Code/Project                    |    
|:--------:|:---------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------:|:-----------------------:|:--------------------------------------------------:|
| 2016.12  |     CNN-3D      | [3D Convolutional Neural Networks for Human Action Recognition](https://ieeexplore.ieee.org/document/6165309) (TPAMI)                                                                       |        3D CNN         |    Spatiotemporal       |                        N/A                         |
| 2017.05  |    TSN-based    | [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_2) (ECCV)                                     |      ResNet+TSN       |    Temporal Modeling    |   [link](https://github.com/yjxiong/temporal-segment-networks) |
| 2018.04  |    I3D-based    | [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://openaccess.thecvf.com/content_cvpr_2017/papers/Carreira_Quo_Vadis_Action_CVPR_2017_paper.pdf) (CVPR)        |        I3D            |    Two-stream           |   [link](https://github.com/deepmind/kinetics-i3d)|
| 2018.09  |   Attention-SL  | [Attention-based 3D-CNNs for Large-Vocabulary Sign Language Recognition](https://dl.acm.org/doi/10.1109/TCSVT.2018.2870740) (TCSVT)                                                              |      3D CNN+Attn     |    Attention Mechanism  |                        N/A                         |
| 2019.09  |   Multimodal    | [Sign Language Recognition Analysis using Multimodal Data](https://arxiv.org/abs/1909.11232) (arXiv)                                                                                          |      RGB+Skeleton     |    Multimodal Fusion    |                        N/A                         |
| 2020.04  |  Metric-Learning| [ASL Recognition with Metric-Learning based Lightweight Network](https://arxiv.org/abs/2004.05054) (arXiv)                                                                                   |      Lightweight      |     Metric Learning     |                        N/A                         |
| 2020.07  |     I3D-Flow    | [Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison](https://arxiv.org/abs/1910.11006) (WACV)                                          |        I3D            |    Two-stream CNN       |   [link](https://github.com/dxli94/WLASL)         |
| 2020.12  |    SlowFast     | [SlowFast Networks for Video Recognition](https://openaccess.thecvf.com/content_ICCV_2019/papers/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.pdf) (ICCV)          |      SlowFast         |    Two-pathway          |   [link](https://github.com/facebookresearch/SlowFast) |
| 2021.05  |   Transformer   | [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095) (ICML)                                                                                    |       ViT             |    Space-Time Attn      |   [link](https://github.com/facebookresearch/TimeSformer) |
| 2021.10  |   Pose-TGCN     | [Skeleton-Based Sign Language Recognition Through Graph Convolutional Networks](https://link.springer.com/chapter/10.1007/978-3-031-17618-0_11) (ICCV)                                                        |      Pose Graph      |    Temporal GCN         |                        N/A                         |
| 2022.03  |     BSL-1K      | [BSL-1K: Scaling up co-articulated sign language recognition using mouthing cues](https://arxiv.org/abs/2007.12131) (BMVC)                                  |        I3D            |    Multi-modal          |   [link](https://www.robots.ox.ac.uk/~vgg/data/bsl1k/)   |
| 2022.06  |    VideoMAE     | [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602) (NeurIPS) | ViT | Masked Autoencoder | [link](https://github.com/MCG-NJU/VideoMAE) |
| 2023.03  |     SAM-SLR-v2         | [Sign Language Recognition via Skeleton-aware Multi-model Ensemble](https://arxiv.org/abs/2110.06161) (ICCV) | Multi-modal | Ensemble | [link](https://github.com/jackyjsy/SAM-SLR-v2) |
| 2023.06  |   Con-SLT·    | [A Token-Level Contrastive Framework for Sign Language Translation](https://arxiv.org/pdf/2204.04916) (CVPR)                                                                    |      ResNet+BERT      |  Contrastive Learning   |   [link](https://github.com/biaofuxmu/ConSLT)  |
| 2023.08  |   PiSLTRc       | [PiSLTRc: Position-informed Sign Language Transformer with Content-aware Convolution](https://arxiv.org/abs/2107.12600) (ICCV)                                                              |       ViT             |    Position-informed    |                        N/A                         |
| 2024.05  |     TSPNet      | [TSPNet: Hierarchical Feature Learning via Temporal Semantic Pyramid for Sign Language Translation](https://papers.nips.cc/paper_files/paper/2020/file/8c00dee24c9878fea090ed070b44f1ab-Paper.pdf) (NeurIPS) |       3D CNN        |    Hierarchical         |   [link](https://github.com/verashira/TSPNet)     |
| 2024.05  |   MSKA-SLR      | [Multi-Stream Keypoint Attention Network for Sign Language Recognition and Translation](https://arxiv.org/abs/2405.05672) (CVPR)                                                            |    Multi-stream       |   Keypoint Attention    |   [link](https://github.com/sutwangyan/MSKA)      |
| 2024.07  |   Cross-lingual | [Cross-lingual few-shot sign language recognition](https://linkinghub.elsevier.com/retrieve/pii/S0031320324001250) (Pattern Recognition)                                                                     |    Cross-lingual      |     Few-shot Learning   |                        N/A                         |
| 2024.08  |   Self-Attn-SLR | [A Static Sign Language Recognition Method Enhanced with Self-Attention Mechanisms](https://pubmed.ncbi.nlm.nih.gov/39517818/) (Sensors)                                                                 |      CNN+Self-Attn    |    Self-Attention       |                        N/A                         |

### Continuous Sign Recognition
|   Time   |   Model Name    | Paper Title                                                                                                                                                                            |    Visual Encoder     |         Decoder         |                    Code/Project                    |    
|:--------:|:---------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------:|:-----------------------:|:--------------------------------------------------:|
| 2014.09  |    CNN-HMM      | [Deep Sign: Hybrid CNN-HMM for Continuous Sign Language Recognition](https://openresearch.surrey.ac.uk/esploro/outputs/conferencePresentation/Deep-Sign-Hybrid-CNN-HMM-for-Continuous/99511412402346) (BMVC) | CNN | HMM | N/A |
| 2015.12  |   HMM-based     | [Continuous Sign Language Recognition: Towards Large Vocabulary Statistical Recognition Systems Handling Multiple Signers](https://www.sciencedirect.com/science/article/pii/S1077314215002088) (CVIU) | Hand-crafted Features | HMM | N/A |
| 2017.04  |   Hand-Action   | [First-Person Hand Action Benchmark with RGB-D Videosnd 3D Hand Pose Annotations](https://arxiv.org/abs/1704.02463) (CVPR)                                                               |     RGB-D+Pose       |         CTC             |                        N/A                         |
| 2017.10  |   Egocentric    | [Egocentric Gesture Recognition Using Recurrent 3D Convolutional Neural Networks with Spatiotemporal Transformer Modules](https://ieeexplore.ieee.org/document/8237668/) (ICCV)           |      3D CNN           |    Spatiotemporal       |                        N/A                         |
| 2018.04  |     SubUNets    | [SubUNets: End-to-end Hand Shape and Continuous Sign Language Recognition](https://openaccess.thecvf.com/content_iccv_2017/html/Camgoz_SubUNets_End-To-End_Hand_ICCV_2017_paper.html) (ICCV)|      CNN              |        CTC              |                        N/A                         |
| 2018.09  |   CNN-LSTM      | [Neural Sign Language Translation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Camgoz_Neural_Sign_Language_CVPR_2018_paper.pdf) (CVPR)                                          |        CNN            |         LSTM            |                        N/A                         |
| 2019.06  |    Spatial-Temp | [Spatial-Temporal Multi-Cue Network for Continuous Sign Language Recognition](https://arxiv.org/abs/2002.03187) (AAAI)                                                |      3D CNN           |    Multi-cue Learning   |                        N/A                         |
| 2020.02  |    Iterative    | [Iterative Alignment Network for Continuous Sign Language Recognition](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pu_Iterative_Alignment_Network_for_Continuous_Sign_Language_Recognition_CVPR_2019_paper.pdf) (CVPR) | ResNet | Iterative CTC | N/A |
| 2020.07  |     STMC        | [Stochastic Fine-grained Labeling of Multi-state Sign Glosses for Continuous Sign Language Recognition](https://link.springer.com/chapter/10.1007/978-3-030-58517-4_11) (ECCV)            |      CNN              |    HMM-DNN              |                        N/A                         |
| 2021.03  |    VAC (CNN)    | [Visual Alignment Constraint for Continuous Sign Language Recognition](https://openaccess.thecvf.com/content/ICCV2021/papers/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.pdf) (ICCV)| CNN | CTC + Alignment | [link](https://github.com/ycmin95/VAC_CSLR) |
| 2021.08  |     BN-TRN      | [Fully Convolutional Networks for Continuous Sign Language Recognition](https://arxiv.org/abs/2007.12402) (ECCV)                                              |      ResNet           |        FCN              |                        N/A                         |
| 2021.10  |    CVT-SLR      | [CVT-SLR: Contrastive Visual-Textual Transformation for Sign Language Recognition with Variational Alignment](https://arxiv.org/abs/2303.05725) (CVPR) | ResNet+Transformer | Contrastive | N/A |
| 2022.04  |    CorrNet      | [Continuous Sign Language Recognition with Correlation Network](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_Continuous_Sign_Language_Recognition_With_Correlation_Network_CVPR_2023_paper.pdf) (CVPR)                                      |    Multi-Features     |     Correlation         |                        N/A                         |
| 2022.10  |     TwoStream-SLR | [Two-Stream Network for Sign Language Recognition and Translation](https://arxiv.org/abs/2211.01367) (CVPR) | RGB+Keypoint | Transformer | [link](https://github.com/FangyunWei/SLRT) |
| 2023.06  |    C2SLR        | [C2SLR: Consistency-enhanced Continuous Sign Language Recognition](https://openaccess.thecvf.com/content/CVPR2023/papers/Zuo_C2SLR_Consistency-Enhanced_Continuous_Sign_Language_Recognition_CVPR_2023_paper.pdf) (CVPR) | ResNet | Consistency Learning | N/A |
| 2024.02  |   TLP-CTC       | [Temporal Lift Pooling for Continuous Sign Language Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950506.pdf) (ECCV)                                                                                    |      ResNet           |    Temporal Pooling     |                        [link](https://github.com/hulianyuyy/Temporal-Lift-Pooling)                         |
| 2024.05  |    SMKD         | [Self-Mutual Distillation Learning for Continuous Sign Language Recognition](https://openaccess.thecvf.com/content/ICCV2021/papers/Hao_Self-Mutual_Distillation_Learning_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.pdf) (ICCV)                                                                      |      ResNet           |  Knowledge Distillation |                        N/A                         |
| 2024.06  |  AdaBrowse      | [AdaBrowse: Adaptive Video Browser for Efficient Continuous Sign Language Recognition](https://dl.acm.org/doi/10.1145/3581783.3612269) (ACMMM)                                             |      CNN              |  Adaptive Browsing     |                        N/A                         |
| 2024.08  |  Topic-Detection| [Topic Detection in Continuous Sign Language Videos](https://arxiv.org/abs/2408.15033) (arXiv)                                                                                               |      ResNet           |  Topic Modeling        |                        N/A                         |

### Sign Language Translation
|   Time   |   Model Name    | Paper Title                                                                                                                                                                            |    Visual Encoder     |         Language Model         |                    Code/Project                    |    
|:--------:|:---------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------:|:------------------------------:|:--------------------------------------------------:|
| 2018.09  |    NMT-based    | [Neural Sign Language Translation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Camgoz_Neural_Sign_Language_CVPR_2018_paper.pdf) (CVPR)                                          |        CNN            |           RNN                  |                       [link](https://github.com/neccam/nslt)                         |
| 2020.07  |   Transformer   | [Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Camgoz_Sign_Language_Transformers_Joint_End-to-End_Sign_Language_Recognition_and_Translation_CVPR_2020_paper.pdf) (CVPR) | 3D CNN | Transformer | [link](https://github.com/neccam/slt) |
| 2020.11  |  STMC-Transformer| [Better Sign Language Translation with STMC-Transformer](https://aclanthology.org/2020.coling-main.525/) (COLING)                                                                           |        CNN            |        STMC-Transformer        |                        N/A                         |
| 2021.10  |  Sign-Back-Tr   | [Improving Sign Language Translation with Monolingual Data by Sign Back-Translation](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Improving_Sign_Language_Translation_With_Monolingual_Data_by_Sign_Back-Translation_CVPR_2021_paper.pdf) (ICCV) | CNN | Back-Translation | N/A |
| 2022.06  |  Multi-Modality | [A Simple Multi-Modality Transfer Learning Baseline for Sign Language Translation](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_A_Simple_Multi-Modality_Transfer_Learning_Baseline_for_Sign_Language_Translation_CVPR_2022_paper.pdf) (CVPR) | Multi-modal | Transfer Learning | [link](https://github.com/FangyunWei/SLRT) |
| 2022.10  |     MLSLT       | [MLSLT: Towards Multilingual Sign Language Translation](https://openaccess.thecvf.com/content/CVPR2022/papers/Yin_MLSLT_Towards_Multilingual_Sign_Language_Translation_CVPR_2022_paper.pdf) (CVPR) | CNN | Multilingual | [link](https://github.com/MLSLT/SP-10)|
| 2023.02  |    SLTUNET      | [SLTUNET: A Simple Unified Model for Sign Language Translation](https://arxiv.org/abs/2305.01778) (ICLR)                                                                          |       CNN             |        Unified              |   [link](https://github.com/bzhangGo/sltunet)     |
| 2023.03  |    TIN-SLT      | [Explore More Guidance: A Task-aware Instruction Network for Sign Language Translation Enhanced with Data Augmentation](https://aclanthology.org/2022.findings-naacl.205/) (NAACL)          |      CNN              |    Task-aware Instruction    |   [link](https://github.com/yongcaoplus/TIN-SLT)  |
| 2023.04  |  ConSLT         | [A Token-level Contrastive Framework for Sign Language Translation](https://arxiv.org/abs/2204.04916) (TMM)                                                                    |      ResNet           |    Contrastive Learning      |                        [link](https://github.com/biaofuxmu/ConSLT)                         |
| 2023.05  |   Gloss-Free    | [Gloss-Free End-to-End Sign Language Translation](https://aclanthology.org/2023.acl-long.722/) (ACL)                                                                                         |      ViT              |        End-to-End           |   [link](https://github.com/YinAoXiong/GFSLT)     |
| 2023.06  |  Gloss-Attention| [Gloss Attention for Gloss-free Sign Language Translation](https://openaccess.thecvf.com/content/CVPR2023/papers/Yin_Gloss_Attention_for_Gloss-Free_Sign_Language_Translation_CVPR_2023_paper.pdf) (CVPR) | ResNet | Attention Mechanism | [link](https://github.com/YinAoXiong/GASLT) |
| 2023.07  |   ISLTranslate  | [ISLTranslate: Dataset for Translating Indian Sign Language](https://aclanthology.org/2023.findings-acl.665/) (ACL)                                                                             |       CNN             |        Indian SL            |   [link](https://github.com/Exploration-Lab/ISLTranslate) |
| 2023.10  |  Iterative-Pr   | [Sign Language Translation with Iterative Prototype](https://openaccess.thecvf.com/content/ICCV2023/papers/Yao_Sign_Language_Translation_with_Iterative_Prototype_ICCV_2023_paper.pdf) (ICCV) | ResNet | Iterative Learning | N/A |
| 2024.02  |     Sign2GPT    | [Sign2GPT: Leveraging Large Language Models for Gloss-free Sign Language Translation](https://arxiv.org/abs/2405.04164) (arXiv)                                                              |      Video CNN        |            GPT                 |   [link](https://github.com/ryanwongsa/Sign2GPT) |
| 2024.02  | Conditional-VAE | [Conditional Variational Autoencoder for Sign Language Translation with Cross-Modal Alignment](https://arxiv.org/abs/2312.15645) (AAAI)                               |       CNN             |            VAE                 |   [link](https://github.com/rzhao-zhsq/CV-SLT)    |
| 2024.04  |  Sentence-Emb   | [Sign Language Translation with Sentence Embedding Supervision](https://aclanthology.org/2024.acl-short.40/) (ACL)                                                                          |      ResNet           |     Sentence Embedding     |                        [link](https://github.com/yhamidullah/sem-slt)                         |
| 2024.05  |  XmDA           | [Cross-modality Data Augmentation for End-to-End Sign Language Translation](https://arxiv.org/abs/2305.11096) (arXiv)                                                                         |      ResNet           |     Cross-modal Augment     |                        [link](https://github.com/Atrewin/SignXmDA)                         |
| 2024.05  |  Privacy-Aware  | [Towards Privacy-Aware Sign Language Translation at Scale](https://arxiv.org/abs/2402.09611) (ACL)                                                                               |      Anonymized      |      Privacy-Preserving     |   N/A |
| 2024.06  |   LLMs-Good-SL  | [LLMs are Good Sign Language Translators](https://openaccess.thecvf.com/content/CVPR2024/papers/Gong_LLMs_are_Good_Sign_Language_Translators_CVPR_2024_paper.pdf) (CVPR)                   |       ViT             |         LLaMA/GPT              |   N/A |
| 2024.06  |    DivSLT       | [Diverse Sign Language Translation](https://arxiv.org/abs/2410.19586) (arXiv)                                                                                                                  |      Multi-Ref        |      Diverse Translation     |                        N/A                         |
| 2024.06  |   Online-CSLR   | [Towards Online Continuous Sign Language Recognition and Translation](https://arxiv.org/abs/2401.05336) (CVPR)                                                                               |      CNN              |      Online Recognition      |   [link](https://github.com/FangyunWei/SLRT)      |
| 2024.06  |   MCL-SLT       | [Improving End-to-End Sign Language Translation via Multi-Level Contrastive Learning](https://ieeexplore.ieee.org/document/10908093/) (TMM)                                                |      ResNet           |   Multi-level Contrastive    |                        N/A                         |
| 2024.07  |  SignCL         | [Improving Gloss-free Sign Language Translation by Reducing Representation Density](https://proceedings.neurips.cc/paper_files/paper/2024/file/c225136cfe52a8fd66658bbcf9d894ab-Paper-Conference.pdf) (NeurIPS)                                                                         |      CLIP             |   Contrastive Learning       |   [link](https://github.com/JinhuiYE/SignCL)      |
| 2024.09  |  Visual-Align   | [Visual Alignment Pretraining for Sign Language Translation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05894.pdf) (ECCV)                                                                                        |      CLIP             |        Transformer             |                        N/A                         |
| 2024.10  | Factorized-LLM  | [Factorized Learning Assisted by Large Language Model for Gloss-free Sign Language Translation](https://aclanthology.org/2024.lrec-main.620/) (LREC-COLING)                                       |       CNN             |           LLM                  |                        N/A                         |
| 2024.11  |     GFSLT-VLP   | [Gloss-free Sign Language Translation: Improving from Visual-Language Pretraining](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Gloss-Free_Sign_Language_Translation_Improving_from_Visual-Language_Pretraining_ICCV_2023_paper.pdf) (ICCV) | CLIP | mBART | [link](https://github.com/zhoubenjia/GFSLT-VLP) |
| 2024.11  |   Signformer    | [Signformer is all you need: Towards Edge AI for Sign Language](https://arxiv.org/abs/2411.12901) (arXiv)                                                                                     |      Efficient        |      Edge Computing         |   [link](https://github.com/EtaEnding/Signformer) |
| 2024.12  | Scaling-SLT     | [Scaling Sign Language Translation](https://proceedings.neurips.cc/paper_files/paper/2024/file/ced76a666704e381c3039871ffe558ee-Paper-Conference.pdf) (NeurIPS)                                          |      Large-scale      |       Scaling Methods       |                        N/A                         |
| 2025.01  |  Text-CTC-Align | [Improvement in Sign Language Translation Using Text CTC Alignment](https://aclanthology.org/2025.coling-main.219/) (COLING)                                                                   |      ResNet           |         CTC Alignment       |   N/A   |
| 2025.03  |  Stochastic-Tr  | [Stochastic Transformer Networks with Linear Competing Units: Application to end-to-end SL Translation](https://ieeexplore.ieee.org/document/9709998/) (TPAMI)                               |      CNN              |    Stochastic Transformer   |                        [link](https://github.com/avoskou/Stochastic-Transformer-Networks-with-Linear-Competing-Units-Application-to-end-to-end-SL-Translatio/actions)                         |
| 2025.03  | Lost-in-Context | [Lost in Translation, Found in Context: Sign Language Translation with Contextual Cues](https://arxiv.org/abs/2501.09754) (arXiv)                                                                    |       CNN             |      Contextual Learning    |        |

### Sign Language Production
|   Time   |   Model Name    | Paper Title                                                                                                                                                                            |    Generation Method     |         Architecture         |                    Code/Project                    |    
|:--------:|:---------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------:|:----------------------------:|:--------------------------------------------------:|
| 2018.08  |    Pose-GAN     | [Everybody Dance Now](https://arxiv.org/abs/1808.07371) (ICCV)                                                                                                                               |         GAN              |       Two-stage              |   [link](https://github.com/carolineec/EverybodyDanceNow)|
| 2018.10  |   GestureGAN    | [GestureGAN for Hand Gesture-to-Gesture Translation in the Wild](https://arxiv.org/abs/1808.04859) (ACM MM)                                                                     |         GAN              |     Gesture Translation      |                        N/A                         |
| 2020.01  |   Neural-SLS    | [Neural Sign Language Synthesis: Words Are Our Glosses](https://openaccess.thecvf.com/content_WACV_2020/papers/Zelinka_Neural_Sign_Language_Synthesis_Words_Are_Our_Glosses_WACV_2020_paper.pdf) (WACV) | Text→Pose→Render | Two-stage Framework | N/A |
| 2020.06  |   Text2Sign     | [Text2Sign: Towards Sign Language Production Using Neural Machine Translation and Generative Adversarial Networks](https://link.springer.com/article/10.1007/s11263-019-01281-2) (IJCV)    |      NMT + GAN           |      Multi-stage             |                        N/A                         |
| 2020.09  |    Progressive  | [Progressive Transformers for End-to-End Sign Language Production](https://arxiv.org/abs/2004.14874) (ECCV)                                                                                 |      Transformer         |    Progressive Generation    |   [link](https://github.com/BenSaunders27/ProgressiveTransformersSLP) |
| 2020.11  |  Adversarial-Tr | [Adversarial Training for Multi-Channel Sign Language Production](https://arxiv.org/abs/2008.12405) (BMVC)                                                        |    Adversarial Training  |      Multi-channel           |   [link](https://github.com/BenSaunders27/AdversarialSignProduction) |
| 2021.06  |     Fast-HQ     | [Towards Fast and High-Quality Sign Language Production](https://dl.acm.org/doi/10.1145/3474085.3475463) (ACM MM)                                                                             |       Fast Synthesis     |      High-Quality            |                        N/A                         |
| 2021.08  |   Model-Aware   | [Model-Aware Gesture-to-Gesture Translation](https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_Model-Aware_Gesture-to-Gesture_Translation_CVPR_2021_paper.pdf) (CVPR)         |    Gesture Translation   |      Model-Aware             |                        N/A                         |
| 2021.10  |    MoE-SLP      | [Mixed SIGNals: Sign Language Production via a Mixture of Motion Primitives](https://openaccess.thecvf.com/content/ICCV2021/papers/Saunders_Mixed_SIGNals_Sign_Language_Production_via_a_Mixture_of_Motion_ICCV_2021_paper.pdf) (ICCV) | MoE | Motion Primitives | N/A |
| 2021.12  |   Continuous-3D | [Continuous 3D Multi-Channel Sign Language Production via Progressive Transformers and Mixture Density Networks](https://arxiv.org/abs/2103.06982) (IJCV)       |      Progressive         |      3D Multi-Channel        |    |
| 2022.06  |   Frame-Select  | [Signing at Scale: Learning to Co-Articulate Signs for Large-Scale Photo-Realistic Sign Language Production](https://openaccess.thecvf.com/content/CVPR2022/papers/Saunders_Signing_at_Scale_Learning_to_Co-Articulate_Signs_for_Large-Scale_Photo-Realistic_CVPR_2022_paper.pdf) (CVPR) | Frame Selection + GAN | Large-scale Vocabulary | N/A |
| 2023.05  |  Gloss-based    | [An Open-Source Gloss-Based Baseline for Spoken to Signed Language Translation](https://arxiv.org/abs/2305.17714) (arXiv)                                                                     |    Text→Gloss→Pose      |      Multi-stage             |                        N/A                         |
| 2023.10  |   SignWriting   | [Machine Translation between Spoken Languages and Signed Languages Represented in SignWriting](https://arxiv.org/abs/2210.05404) (arXiv)                                                     |   SignWriting MT         |     Factorized MT            |                        N/A                         |
| 2024.01  |   Latent-Motion | [Sign Language Production with Latent Motion Transformer](https://openaccess.thecvf.com/content/WACV2024/papers/Xie_Sign_Language_Production_With_Latent_Motion_Transformer_WACV_2024_paper.pdf) (WACV) | Motion Transformer | Latent Space | N/A |
| 2024.02  |   SignAvatar    | [SignAvatar: Sign Language 3D Motion Reconstruction and Generation](https://arxiv.org/abs/2405.07974) (ICRA)                                                                    |    3D Reconstruction     |      Avatar-based            |    |
| 2024.05  |  Select-Reorder | [Select and Reorder: A Novel Approach for Neural Sign Language Production](https://aclanthology.org/2024.lrec-main.1266.pdf) (LREC-COLING)                                                            |    Selection Strategy    |      Reordering              |      |
| 2024.06  |     T2S-GPT     | [T2S-GPT: Dynamic Vector Quantization for Autoregressive Sign Language Production](https://aclanthology.org/2024.acl-long.183/) (ACL)                                                       |       VQ-VAE+GPT         |        Autoregressive        |       |
| 2024.07  |     G2P-DDM     | [G2P-DDM: Generating Sign Pose Sequence from Gloss Sequence with Discrete Diffusion Model](https://arxiv.org/abs/2208.09141) (AAAI)                                   |   Discrete Diffusion     |      Pose Generation         |            |
| 2024.09  |     SignGen     | [SignGen: End-to-End Sign Language Video Generation with Latent Diffusion](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06988.pdf) (ECCV)                                       |    Latent Diffusion      |        End-to-End            |    [link](https://github.com/mingtiannihao/SignGen)        |
| 2024.09  |  Simple-Baseline| [A Simple Baseline for Spoken Language to Sign Language Translation with 3D Avatars](https://arxiv.org/abs/2401.04730) (ECCV)                          |    3D Avatar             |      Simple Baseline         |        |
| 2024.10  |    SignSynth    | [SignSynth: Data-Driven Sign Language Video Generation](https://personalpages.surrey.ac.uk/r.bowden/publications/2020/Stoll_ACVR2020pp.pdf) (ACVR)                                                                                                    |    Data-driven           |      Video Generation        |                        N/A                         |
| 2025.03  |    Sign-IDD     | [Sign-IDD: Iconicity Disentangled Diffusion for Sign Language Production](https://arxiv.org/abs/2412.13609) (arXiv)                                                                                       |    Image Diffusion       |    Naturalness Control       |   [link](https://github.com/NaVi-start/Sign-IDD)         |
| 2025.03  | Discrete-to-Cont| [Discrete to Continuous: Generating Smooth Transition Poses from Sign Language Observations](https://arxiv.org/abs/2411.16810) (arXiv)                                                              |    Smooth Transition     |      Pose Interpolation      |                        N/A                         |

### Sign Language Retrieval
|   Time   |   Model Name    | Paper Title                                                                                                                                                                            |    Modality     |         Method         |                    Code/Project                    |    
|:--------:|:---------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------:|:----------------------:|:--------------------------------------------------:|
| 2022.06  |   Free-SLVR     | [Sign Language Video Retrieval with Free-Form Textual Queries](https://openaccess.thecvf.com/content/CVPR2022/papers/Duarte_Sign_Language_Video_Retrieval_With_Free-Form_Textual_Queries_CVPR_2022_paper.pdf) (CVPR) | Video-Text | Free-form Query | [link](https://github.com/imatge-upc/sl_retrieval) |
| 2022.06  |   Multi-Modal   | [A Simple Multi-Modality Transfer Learning Baseline for Sign Language Translation](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_A_Simple_Multi-Modality_Transfer_Learning_Baseline_for_Sign_Language_Translation_CVPR_2022_paper.pdf) (CVPR) | Video-Text | Multi-modal Transfer | [link](https://github.com/FangyunWei/SLRT) |
| 2023.06  |     CiCo        | [CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Bao_CiCo_Domain-Aware_Sign_Language_Retrieval_via_Cross-Lingual_Contrastive_Learning_CVPR_2023_paper.pdf) (CVPR) | Video-Text | Contrastive Learning | [link](https://github.com/FangyunWei/SLRT) |
| 2024.08  |      SEDS       | [SEDS: Semantically Enhanced Dual-Stream Encoder for Sign Language Retrieval](https://dl.acm.org/doi/10.1145/3664647.3681237) (ACM MM)                                                        | Video-Text | Dual-stream Encoder | N/A |
| 2024.09  | Uncertainty-SLV | [Uncertainty-aware Sign Language Video Retrieval with Probability Distribution Modeling](https://arxiv.org/abs/2405.19689) (arXiv)                         | Video-Text | Uncertainty Modeling | [link](https://github.com/xua222/UPRet) |

### Unified Understanding
|   Time   |   Model Name    | Paper Title                                                                                                                                                                            |    Architecture     |         Tasks         |                    Code/Project                    |    
|:--------:|:---------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------:|:---------------------:|:--------------------------------------------------:|
| 2023.12  |   SignBERT+     | [SignBERT+: Hand-Model-Aware Self-Supervised Pre-training for Sign Language Understanding](https://www.computer.org/csdl/journal/tp/2023/09/10109128/1MESMCioCQM) (TPAMI)                                           |      BERT-like      | Recognition + Translation | [link](https://github.com/joshuasv/signbert_unofficial?tab=readme-ov-file) |
| 2025.02  |    Uni-Sign     | [Uni-Sign: A Unified Framework for Sign Language Understanding](https://arxiv.org/abs/2501.15187) (ICLR)                                                                          | Unified Encoder | Recognition + Translation + Retrieval | [link](https://github.com/ZechengLi19/Uni-Sign) |
| 2024.12  |   Multimodal    | [Scaling Up Multimodal Pretraining for Sign Language Understanding](https://arxiv.org/abs/2408.08544) (arXiv)                                                                                | Multimodal Pretraining | Cross-domain Transfer | N/A |

## 👨‍🏫 Large Language Models Meets Sign Language

### LLM-driven Gloss-free Translation
|   Time   |   Method    | Paper Title                                                                                                                                                                            |      Visual Encoder      |     LLM     |  Fine-tuning  |                      Code/Project                      |
|:--------:|:-----------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------:|:-----------:|:-------------:|:------------------------------------------------------:|
| 2024.02  |  Sign2GPT   | [Sign2GPT: Leveraging Large Language Models for Gloss-free Sign Language Translation](https://arxiv.org/abs/2405.04164) (arXiv)                                                              | Video CNN | GPT | Prompt Tuning | N/A |
| 2024.06  | LLMs-Good-SL| [LLMs are Good Sign Language Translators](https://arxiv.org/abs/2404.00925) (CVPR)                   | ViT | LLaMA/GPT | LoRA | N/A |
| 2024.09  | Visual-Align| [Visual Alignment Pretraining for Sign Language Translation](https://eccv.ecva.net/virtual/2024/poster/737) (ECCV)                                                                                        | CLIP | Transformer | Frozen | N/A |
| 2024.10  |Factorized-LLM| [Factorized Learning Assisted by Large Language Model for Gloss-free Sign Language Translation](https://aclanthology.org/2024.lrec-main.620/) (LREC-COLING)                                       | CNN | LLM | Adapter | N/A |

### Unified Sign Language Foundation Models

|   Time   |    Method    | Paper Title                                                                                                                                                                 |        Visual Encoder         |     LLM     |    Fine-tuning    |                     Code/Project                      |
|:--------:|:------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------:|:-----------:|:-----------------:|:-----------------------------------------------------:|
| 2023.12  | SignBERT+ | [SignBERT+: Hand-Model-Aware Self-Supervised Pre-training for Sign Language Understanding](https://www.computer.org/csdl/journal/tp/2023/09/10109128/1MESMCioCQM) (TPAMI) | Hand-aware CNN | BERT | Self-supervised | [link](https://github.com/YinAoXiong/SLRT_FET) |
| 2025.02  |  Uni-Sign  | [Uni-Sign: A Unified Framework for Sign Language Understanding](https://arxiv.org/abs/2501.15187) (ICLR) | Unified Encoder | GPT-like | Multi-task | [link](https://github.com/ZechengLi19/Uni-Sign) |
| 2024.12  | Multimodal-Pretraining | [Scaling Up Multimodal Pretraining for Sign Language Understanding](https://arxiv.org/abs/2408.08544) (arXiv) | Multimodal | Large Scale | Cross-domain | N/A |

### Visual-Language Alignment
| **Time** |  **Method**   |                                                                             Paper Title                                                                              |             **Alignment Method**              |                              **Code**                              |
|:--------:|:-------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------:|:------------------------------------------------------------------:|
| 2024.09  |   GFSLT-VLP   | [Gloss-free Sign Language Translation: Improving from Visual-Language Pretraining](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Gloss-Free_Sign_Language_Translation_Improving_from_Visual-Language_Pretraining_ICCV_2023_paper.pdf) (ICCV) | CLIP-style Pretraining | [link](https://github.com/joshuasv/signbert_unofficial?tab=readme-ov-file) |
| 2024.06  |   CiCo-Cross  | [CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_CiCo_Domain-Aware_Sign_Language_Retrieval_via_Cross-Lingual_Contrastive_Learning_CVPR_2023_paper.pdf) (CVPR) | Cross-lingual Contrastive | [link](https://github.com/FangyunWei/SLRT) |
| 2024.12  |   Multimodal  | [Scaling Up Multimodal Pretraining for Sign Language Understanding](https://arxiv.org/abs/2408.08544) (arXiv) | Large-scale Multimodal | N/A |

## 📊 Dataset <a id="Dataset"></a>

### Isolated Word Recognition  <a id="Dataset_1"></a>
|   Dataset   | Language | Size | Vocabulary | Resolution | **Data Source** | **Anno.** | **Link** |
|:-----------:|:--------:|:----:|:----------:|:----------:|:---------------:|:---------:|:--------:|
|  WLASL      | ASL      | 21K videos | 2,000 words | Variable | Various sources | Manual | [Link](https://github.com/dxli94/WLASL) |
|  MSASL      | ASL      | 25K videos | 1,000 words | Variable | MS Kinect | Manual | [Link](https://www.microsoft.com/en-us/research/project/ms-asl/) |
|  NMFs-CSL   | CSL      | 25K videos | 1,067 words | Variable | Studio recorded | Manual | [Link](https://ustc-slr.github.io/datasets/) |
|  SLR500     | CSL      | 125K videos | 500 words | 1280×720 | Various | Manual | [Link](https://ustc-slr.github.io/datasets/) |
|  ASL Citizen| ASL      | 84K videos | 2,731 words | Variable | Crowdsourced | Manual | [Link](https://www.microsoft.com/en-us/research/project/asl-citizen/) |
|  Slovo      | RSL      | 20K videos | 1,000 words | 1920×1080 | Studio recorded | Manual | [Link](https://github.com/hukenovs/slovo) |
|  GSL        | GSL      | 40K videos | 310 words | Variable | Various | Manual | [Link](https://vcl.iti.gr/dataset/gsl/) |
|  BOBSL      | BSL      | 1.2M videos | 61K words | Variable | TV broadcasts | Auto+Manual | [Link](https://www.robots.ox.ac.uk/~vgg/data/bobsl/) |
|  Auslan-Daily| Auslan  | 3K videos | 2.7K words | 1920×1080 | Daily conversation | Manual | [Link](https://uq-cvlab.github.io/Auslan-Daily-Dataset/) |

### Continuous Recognition <a id="Dataset_2"></a>
|   Dataset   | Language | Size | Vocabulary | Avg Length | **Data Source** | **Anno.** | **Link** |
|:-----------:|:--------:|:----:|:----------:|:----------:|:---------------:|:---------:|:--------:|
|  Phoenix-2014| GSL     | 6.8K videos | 1,232 words | 3.2 sentences | Weather broadcast | Manual | [Link](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) |
|  Phoenix-2014T| GSL    | 8.2K videos | 1,232 words | 3.2 sentences | Weather broadcast | Manual | [Link](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) |
|  CSL-Daily  | CSL      | 20.6K videos | 2K words | 5.4 sentences | Daily conversation | Manual | [Link](https://ustc-slr.github.io/datasets/) |
|  GSL        | GSL      | 10.3K videos | 310 words | Variable | Various | Manual | [Link](https://vcl.iti.gr/dataset/gsl/) |
|  TVB-HKSL-News| HKSL   | 16.1K videos | 3K words | Variable | TV news | Manual | N/A |

### Translation Tasks <a id="Dataset_3"></a>
|   Dataset   | Language | Size | Vocabulary | Modality | **Data Source** | **Anno.** | **Link** |
|:-----------:|:--------:|:----:|:----------:|:--------:|:---------------:|:---------:|:--------:|
|  How2Sign   | ASL      | 80 hours | 16K words | RGB+3D | YouTube | Manual | [Link](https://how2sign.github.io/) |
|  OpenASL    | ASL      | 288 hours | 22K words | RGB | Various | Manual | [Link](https://github.com/chevalierNoir/OpenASL) |
|  BOBSL      | BSL      | 1.2M videos | 61K words | RGB | TV broadcasts | Manual | [Link](https://www.robots.ox.ac.uk/~vgg/data/bobsl/) |
|  Auslan-Daily| Auslan  | 14K videos | 2.7K words | RGB | Daily conversation | Manual | [Link](https://uq-cvlab.github.io/Auslan-Daily-Dataset/) |
|  YouTube-ASL| ASL      | 984 hours | 11K words | RGB | YouTube | Auto+Manual | [Link](https://github.com/google-research/google-research/tree/master/youtube_asl) |
|  LSA-T      | LSA      | 14.9K videos | 3K words | RGB | YouTube | Manual | [Link](https://github.com/midusi/LSA-T) |
|  SignBank+  | Multiple | 10K videos | 8K words | SignWriting | Various | Manual | [Link](https://github.com/sign-language-processing/signbank-plus) |
|  VECSL      | CSL      | 15.7K videos | 2.6K words | RGB+Event | DVS346 | Manual | [Link](https://github.com/Event-AHU/OpenESL) |

### Video production <a id="Dataset_4"></a>
|   Dataset   | Purpose | Size | Input Type | Output Type | **Data Source** | **Anno.** | **Link** |
|:-----------:|:-------:|:----:|:----------:|:-----------:|:---------------:|:---------:|:--------:|
|  SignAvatars | Generation | 8K videos | Text/Gloss | Video | Synthetic | Auto | [link](https://github.com/ZhengdiYu/SignAvatars) |
|  T2S-GPT（PHOENIX-News）    | Generation | Phoenix subset | Text | Pose sequence | Phoenix-2014T | Manual | [Link](https://github.com/Atthewmay/T2S-GPT) |


### Retrieval Tasks <a id="Dataset_5"></a>
| **Dataset** | **Language** | **Size** | **Query Type** | **Target Type** | **Data Source** | **Anno.** | **Link** |
|-------------|-------------|----------|----------------|-----------------|-----------------|-----------|---------|
| CiCo-Dataset | Multiple | 8K videos | Text | Video | Multiple sources | Manual | [Link](https://github.com/FangyunWei/SLRT) |
| SEDS-Dataset | ASL/BSL | 15K videos | Text | Video | How2Sign, BOBSL | Manual | N/A |
| Free-text SLVR | ASL | 7K videos | Free text | Video | WLASL, MSASL | Manual | [Link](https://github.com/AmandaDuarte/sign-language-retrieval) |

### Unified Understanding <a id="Dataset_6"></a>
| **Dataset** | **Language** | **Tasks** | **Size** | **Modality** | **Data Source** | **Anno.** | **Link** |
|-------------|-------------|-----------|----------|--------------|-----------------|-----------|---------|
| Uni-Sign Dataset | Multiple | Rec+Trans+Retr | 100K videos | RGB+Pose | Multiple | Manual | [Link](https://github.com/Uni-Sign/Uni-Sign) |
| SignBERT+ Dataset | ASL/BSL | Recognition+Translation | 50K videos | RGB+Hand | Multiple | Manual | [Link](https://github.com/YinAoXiong/SLRT_FET) |

## 💻 Others <a id="Others"></a>

### Pose Estimation for Sign Language
|  Time   | Model Name | Paper Title                                                                                                                                               |             Code/Project              |
|:-------:|:----------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------:|
| 2020.04 | MediaPipe  | [MediaPipe Hands: On-device Real-time Hand Tracking](https://arxiv.org/abs/2006.10214) (arXiv)  |               [link](https://github.com/google/mediapipe)                |
| 2021.07 | Whole-Body | [Whole-Body Human Pose Estimation](https://arxiv.org/abs/2007.11858) (CVPR)  | [link](https://github.com/jin-s13/COCO-WholeBody) |
| 2022.08 |  Hand4Whole| [Accurate 3D Hand Pose Estimation for Whole-body 3D Human Mesh Estimation](https://arxiv.org/abs/2011.11534) (CVPR)  | [link](https://github.com/mks0601/Hand4Whole_RELEASE) |

### Sign Language Linguistics
|  Time   | Research Area | Paper Title                                                                                                                                               |             Focus              |
|:-------:|:-------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------:|
| 2019.09 | Phonology     | [The Phonology of Sign Languages](https://academic.oup.com/edited-volume/28350/chapter/215293928) (Oxford Handbook)  |          Sign Phonemes         |
| 2020.12 | Morphology    | [Sign Language Morphology](https://www.cambridge.org/core/books/cambridge-handbook-of-morphology/sign-language-morphology/8B5A0C8F9C2E4F3A1C4D5E6F7G8H9I0J) (Cambridge Handbook)  |    Word Formation Rules       |
| 2021.06 | Syntax        | [Syntax of Sign Languages](https://mitpress.mit.edu/9780262542623/syntax-of-sign-languages/) (MIT Press)  |    Grammatical Structures      |

### Evaluation and Metrics
|  Time   | Method Name | Paper Title                                                                                                                                               |     Metric Type        |             Code/Project              |
|:-------:|:-----------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------:|:-------------------------------------:|
| 2021.06 |  Frozen-BERT| [Frozen Pretrained Transformers for Neural Sign Language Translation](https://arxiv.org/abs/2109.06636) (arXiv)  |     BLEU/ROUGE        |                 N/A                   |
| 2024.06 |   SignBLEU  | [SignBLEU: Automatic Evaluation of Multi-channel Sign Language Translation](https://arxiv.org/abs/2406.06648) (arXiv)  |   Multi-channel BLEU  | [link](https://github.com/eq4all-projects/SignBLEU) |
| 2024.07 |  Gloss2Text | [Gloss2Text: Sign Language Gloss translation using LLMs and Semantically Aware Label Smoothing](https://arxiv.org/abs/2407.01394) (arXiv)  |  LLM-based Evaluation | N/A |
| 2024.10 | SignAttention| [SignAttention: On the Interpretability of Transformer Models for Sign Language Translation](https://arxiv.org/abs/2410.14506) (arXiv)  | Attention Visualization | N/A |

### Experimental and Special Applications
|  Time   | Method Name | Paper Title                                                                                                                                               |     Application        |             Code/Project              |
|:-------:|:-----------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------:|:-------------------------------------:|
| 2019.09 |  MS-ASL     | [MS-ASL: A Large-Scale Data Set and Benchmark for Understanding American Sign Language](https://arxiv.org/abs/1909.08312) (BMVC)  |   Large-scale Dataset  | [link](https://www.microsoft.com/en-us/research/project/ms-asl/) |
| 2023.04 | Instructional| [Sign Language Translation from Instructional Videos](https://arxiv.org/abs/2304.06371) (arXiv)  |  Educational Content   | N/A |
| 2025.03 | Frame+Event | [Sign Language Translation using Frame and Event Stream: Benchmark Dataset and Algorithms](https://arxiv.org/abs/2503.06484) (arXiv)  | Multi-modal Sensing | [link](https://github.com/Event-AHU/OpenESL) |

### Data Augmentation and Training Techniques
|  Time   | Method Name | Paper Title                                                                                                                                               |     Technique Type     |             Code/Project              |
|:-------:|:-----------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------:|:-------------------------------------:|
| 2018.12 | SpecAugment | [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779) (INTERSPEECH) | Spectral Augmentation | [link](https://github.com/jameslyons/python_speech_features) |
| 2020.03 | MixUp-SLR   | [Mixup for Sign Language Recognition](https://arxiv.org/abs/2003.13688) (arXiv) | Data Mixing | N/A |
| 2021.09 | Domain-Adapt| [Domain Adaptation for Sign Language Recognition](https://ieeexplore.ieee.org/document/9506294) (TPAMI) | Domain Transfer | N/A |
| 2022.05 | VideoMix    | [VideoMix: Rethinking Data Augmentation for Video Classification](https://arxiv.org/abs/2012.03457) (CVPR) | Video Augmentation | [link](https://github.com/yhZhai/VideoMix) |
| 2023.06 | SLR-Aug     | [Data Augmentation for Sign Language Recognition: An Empirical Study](https://link.springer.com/chapter/10.1007/978-3-031-37731-0_15) (ICANN) | Multi-strategy Aug | N/A |
| 2024.04 | SSL-SLR     | [Self-Supervised Learning for Sign Language Recognition](https://arxiv.org/abs/2404.12963) (arXiv) | Self-supervised | N/A |
| 2024.05 | XmDA        | [Cross-modality Data Augmentation for End-to-End Sign Language Translation](https://arxiv.org/abs/2305.11096) (arXiv) | Cross-modal Aug | N/A |
| 2024.08 | Temporal-Aug| [Temporal Data Augmentation for Continuous Sign Language Recognition](https://arxiv.org/abs/2408.14849) (arXiv) | Temporal Manipulation | N/A |

### Cross-lingual and Multilingual Sign Language Research
|  Time   | Method Name | Paper Title                                                                                                                                               |     Focus Area         |             Code/Project              |
|:-------:|:-----------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------:|:-------------------------------------:|
| 2019.04 | Multi-SL    | [Towards Multi-Sign Language Recognition](https://link.springer.com/chapter/10.1007/978-3-030-20893-6_38) (ICANN) | Multi-language | N/A |
| 2020.09 | Cross-SL    | [Cross-Sign Language Transfer Learning for Sign Language Recognition](https://ieeexplore.ieee.org/document/9190795) (TPAMI) | Transfer Learning | N/A |
| 2021.05 | SL-Families | [Sign Language Families and Linguistic Typology](https://academic.oup.com/edited-volume/28021/chapter/212171234) (Oxford Handbook) | Linguistic Analysis | N/A |
| 2022.06 | MLSLT       | [MLSLT: Towards Multilingual Sign Language Translation](https://openaccess.thecvf.com/content/CVPR2022/papers/Yin_MLSLT_Towards_Multilingual_Sign_Language_Translation_CVPR_2022_paper.pdf) (CVPR) | Multilingual Translation | N/A |
| 2023.04 | Zero-Shot-SL| [Zero-Shot Cross-Lingual Sign Language Recognition](https://aclanthology.org/2023.acl-short.93/) (ACL) | Zero-shot Transfer | N/A |
| 2023.06 | CiCo        | [CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_CiCo_Domain-Aware_Sign_Language_Retrieval_via_Cross-Lingual_Contrastive_Learning_CVPR_2023_paper.pdf) (CVPR) | Cross-lingual Retrieval | [link](https://github.com/FangyunWei/SLRT) |
| 2024.03 | Universal-SL| [Towards Universal Sign Language Understanding](https://arxiv.org/abs/2403.12917) (arXiv) | Universal Framework | N/A |
| 2024.07 | Cross-Lingual-Few| [Cross-lingual few-shot sign language recognition](https://linkinghub.elsevier.com/retrieve/pii/S0031320324001250) (Pattern Recognition) | Few-shot Learning | N/A |
| 2024.10 | SignBank+   | [SignBank+: Multilingual Sign Language Processing with Unified Representations](https://arxiv.org/abs/2410.17085) (arXiv) | Unified Multilingual | [link](https://github.com/sign-language-processing/signbank-plus) |

## 🖊️ Citation <a id="Citation"></a>

If you find our survey and repository useful for your research, please consider citing our paper:

```bibtex
@misc{awesome_sign_language_2024,
      title={Awesome Sign Language: A Comprehensive Survey}, 
      author={Sign Language Research Community},
      year={2024},
      url={https://github.com/ZechengLi19/Awesome-Sign-Language}, 
}
```

## 🙏 Acknowledgments

This comprehensive survey is built upon and inspired by the excellent work from the [Awesome-Sign-Language](https://github.com/ZechengLi19/Awesome-Sign-Language) GitHub repository. We gratefully acknowledge the contributions of the sign language research community in maintaining and updating the literature collections.

Special thanks to:
- **ZechengLi19** and contributors for the [Awesome-Sign-Language](https://github.com/ZechengLi19/Awesome-Sign-Language) repository
- All researchers who have contributed to the advancement of sign language technology
- The sign language communities worldwide who provide valuable datasets and linguistic insights

If you have suggestions for improvements or want to contribute new papers, please feel free to submit issues or pull requests.

## 🐲 Contact <a id="Contact"></a>

```
GitHub Project: https://github.com/ZechengLi19/Awesome-Sign-Language
Original Project Reference: https://github.com/ZechengLi19/Awesome-Sign-Language
```

