<div align="center">

# 🤟 Awesome Sign Language [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

**A Comprehensive Collection of Sign Language Research Papers, Datasets, and Resources**

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](https://github.com/Jim3503/Awesome_Sign/pulls)
[![GitHub stars](https://img.shields.io/github/stars/Jim3503/Awesome_Sign?style=social)](https://github.com/Jim3503/Awesome_Sign/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Jim3503/Awesome_Sign?style=social)](https://github.com/Jim3503/Awesome_Sign/network/members)

[![Last Update](https://img.shields.io/badge/Last%20Update-2026--11--19-green.svg)](https://github.com/Jim3503/Awesome_Sign)

---

*🌟 "From Recognition to Understanding: Bridging the Gap Between Sign Language and AI" 🌟*

</div>

<br />
<p align="center">
  <h1 align="center">Sign Language Research Collection</h1>
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

This repo is used for recording and tracking recent Sign Language research, including Recognition, Translation, Production, and Unified Understanding. If you find any work missing or have any suggestions (papers, implementations, and other resources), feel free to [pull requests](https://github.com/Jim3503/Awesome_Sign/pulls).

### :star: Share us a :star:
Share us a :star: if you're interested in this repo. We will continue to track relevant progress and update this repository.

### 🙌 Suggest a paper

- Suggest relevant work through an issue or pull request, including the paper title, primary source and task.

### 🥳 News

- **2026-11-19 (latest-work follow-up):** Added 14 more works, including November releases and 2026 conference updates. [Read the latest-work table](#latest-work).
- **2026-11-19:** Added 25 verified papers from 2025–2026, a BibTeX file and selected corrections to older entries. [See the update](#verified-additions).

[//]: # (- **2025.5.26**: The first version is available.)

### ✨ Highlight!!

✅ A collection covering recognition, translation, production, retrieval and unified understanding.

✅ Extensive datasets and code links are provided.

✅ Focus on the latest trends: **Gloss-free modeling**, **LLM integration**, and **Unified architectures**.

## 📖 Introduction

**Background and Motivation:**
- Sign Language research tasks include: **Recognition**, **Translation**, **Production**, and **Unified Understanding**.
- Most tasks are multimodal + multilingual problems, facing challenges like: **difficult data acquisition**, **heavy gloss dependency**, **weak semantic understanding**.
- With the advancement of **Large Language Models (like GPT)** and multimodal modeling capabilities, significant progress is being made in **gloss-free approaches**, **unified architectures**, and **multilingual transfer**.

Timeline of Sign Language works:

<!-- ![Alt Text](fig/overview_3.png) -->

## Latest papers and 2026 conference updates <a id="latest-work"></a>

**Follow-up checked on 2026-11-19: 14 additional works**, including a new preprint posted on **2026-11-17**. Together with the [earlier 25-paper update](#verified-additions), these two dated updates add **39 distinct works**; this is not the total size of the repository.

[BibTeX for these 14 works](references_latest_2026-11-19.bib) · [Verification and version notes](UPDATES_2026-11-19.md#latest-work-follow-up)

The date column distinguishes arXiv posting/revision from publisher online publication. A newer revision is not counted as another paper. This selection includes recent releases and additional 2026 conference work; it is not an exhaustive list of everything published before the cutoff. An accepted conference paper is not labelled as already published in proceedings when only an official programme/listing was checked.

| Version / publication date | Paper | Verified status as of 2026-11-19 | Task and contribution | Author resource |
|---|---|---|---|---|
| 2026-11-17 | [SignAligner: Dual-Granularity Span Generation for Sign Language Alignment](https://arxiv.org/abs/2611.22409) | arXiv — preprint | Alignment / spotting: jointly predicts subtitle and word spans, then refines temporal overlap using reward-based training. | — |
| 2026-11-06 (revised) | [Maswao: An automatically collected, prepared, and benchmarked dataset for South African Sign Language](https://arxiv.org/abs/2611.02903) | arXiv — preprint | Dataset / SLT: automatically collects South African signing broadcasts and builds weakly aligned video–text pairs. | — |
| 2026-11-05 | [MV-Sign: Multi-View 3D Motion Integration and Enhanced Text-to-Motion Conditioning for Sign Language Production](https://arxiv.org/abs/2611.07811) | arXiv; WACV 2027 acceptance reported by authors | 3D production: multi-view motion curation, body–hand tokens and retrieval improve text-conditioned generation. | — |
| 2026-11-02 (arXiv); Oct 2026 proceedings | [Znaki: A Russian Fingerspelling Recognition Dataset](https://aclanthology.org/2026.wslp-1.20/) | WSLP 2026 (workshop) | Fingerspelling dataset: Russian signing videos with character and sequence recognition baselines. | — |
| 2026-10-15 (revised) | [DINO Eats Geometry: Self-Supervised Sign Representations that Absorb Articulated Hand Geometry](https://arxiv.org/abs/2609.06296) | arXiv — preprint | Representation learning: SignDEG transfers articulated hand geometry into a video encoder during pretraining; geometry heads are removed at inference. | — |
| 2026-10-14 (online advance) | [Machine Translation for Sign Languages](https://www.annualreviews.org/content/journals/10.1146/annurev-linguistics-040125-043013) | Annual Review of Linguistics — Review in Advance | Survey: connects translation and production methods with linguistic structure, evaluation and Deaf-community participation. | — |
| 2026-10-12 | [From Clips to Streams: A Unified Framework for Streaming Sign Language Translation](https://arxiv.org/abs/2611.00055) | NeurIPS 2026 — accepted; official listing ([programme](https://neurips.cc/virtual/2026/poster/155080)) | Streaming SLT: jointly segments and translates untrimmed video; separates real streaming datasets from synthetic streaming benchmarks. | [Code/data](https://github.com/streamsl/StreamSLST) |
| 2026-09-08 | [SignRefine: Adapting Foundational Video Models for Sign Language Generation](https://arxiv.org/abs/2609.08496) | ECCV 2026 — official accepted-paper list ([listing](https://eccv.ecva.net/Conferences/2026/AcceptedPapers)) | Video generation: adapts a video diffusion transformer with local hand/face refinement and introduces NVSign. | [Project](https://cogvis-cvssp.github.io/papers/signrefine/) |
| 2026-09-02 (revised) | [M3T: Discrete Multi-Modal Motion Tokens for Sign Language Production](https://arxiv.org/abs/2603.23617) | BMVC 2026 — accepted; official programme ([programme](https://bmvc2026.bmva.org/programme/tuesday/)) | 3D production: uses separate body, hand and face tokens, richer facial expression and an auxiliary translation objective. | [Project; code marked coming soon](https://cogvis-cvssp.github.io/papers/m3t/) |
| 2026-09-02 (revised; first posted 2025) | [SignBind-LLM: Multi-Stage Modality Fusion for Sign Language Translation](https://arxiv.org/abs/2509.00030) | ECCV 2026 — official accepted-paper list ([listing](https://eccv.ecva.net/Conferences/2026/AcceptedPapers)) | SLT: combines signing, fingerspelling and lipreading experts through learned temporal alignment and a language decoder. | — |
| 2026-06-27 | [BackTranslation2.0 -- A Linguistically Motivated Metric to Assess Sign Language Production](https://arxiv.org/abs/2606.28673) | ECCV 2026 — official accepted-paper list ([listing](https://eccv.ecva.net/Conferences/2026/AcceptedPapers)) | Production evaluation: combines linguistic and motion tools into interpretable scores and compares them with human ratings. | [Project](https://cogvis-cvssp.github.io/BackTranslation2/) |
| 2026-06-26 | [SIGNET: Motion-Level Knowledge Transfer for Cross-Language Sign Language Translation](https://arxiv.org/abs/2606.28626) | ECCV 2026 — official accepted-paper list ([listing](https://eccv.ecva.net/Conferences/2026/AcceptedPapers)) | Cross-language SLT: combines pretrained sign backbones using hand-prior attention and gated expert fusion. | — |
| 2026-06-25 (revised) | [SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning](https://arxiv.org/abs/2603.10446) | ECCV 2026 — official accepted-paper list ([listing](https://eccv.ecva.net/Conferences/2026/AcceptedPapers)) | Multilingual production: sparse keyframes condition flow-matching generation of dense 3D signing motion. | [Code](https://github.com/JianHe0628/SignSparK) · [Project](https://cogvis-cvssp.github.io/papers/signspark/) |
| 2026-03-19 | [SignAgent: Agentic LLMs for Linguistically-Grounded Sign Language Annotation and Dataset Curation](https://arxiv.org/abs/2603.19059) | arXiv — preprint | Annotation / curation: a tool-using LLM coordinates pseudo-gloss annotation and lexical-variant grouping. | — |

**Version and task cautions:** SignDEG (`arXiv:2609.06296`, current title *DINO Eats Geometry*) is distinct from Gan et al.'s CVPR 2026 SignDINO paper in the earlier update. SignAligner predicts temporal spans rather than translated sentences. SignRefine generates video from pose conditioning; it should not be compared directly to text-to-pose or video-to-text systems. MV-Sign's WACV 2027 acceptance is attributed to the arXiv authors, not independently confirmed proceedings. Project pages and code availability are recorded separately; no linked implementation was executed.

## Verified additions: 2025–2026 <a id="verified-additions"></a>

**Checked on 2026-11-19 · 25 papers newly added to this repository.** Titles and publication status were checked against publisher proceedings, ACL Anthology, CVF Open Access, ICLR proceedings or arXiv. This is a curated update, not an exhaustive search or a full audit of the older tables below.

[BibTeX for these 25 papers](references_2026-11-19.bib) · [Update notes and corrections](UPDATES_2026-11-19.md)

**Task key:** ISLR = isolated sign classification; CSLR = video-to-gloss recognition; SLT = sign-to-written-language translation; SLP = text/gloss-to-sign production. Text-to-gloss and sign retrieval are labelled separately. “Gloss-free” does not mean supervision-free or no external pretraining data.

**Status key:** main conference, Findings and workshop papers are labelled separately. An arXiv link is not evidence of conference acceptance. “—” in the code column means no author-linked implementation was verified in this update, not that none exists.

### Visual pretraining and efficient translation

| Publication / status | Paper | Task | What to read it for | Verified code / resource |
|---|---|---|---|---|
| CVPR 2026 | [Learning Effective Sign Features without Text for Gloss-free Sign Language Translation](https://openaccess.thecvf.com/content/CVPR2026/html/Gan_Learning_Effective_Sign_Features_without_Text_for_Gloss-free_Sign_Language_CVPR_2026_paper.html) | SLT / pretraining | SignDINO learns sign-aware visual features without text during pretraining; downstream translation still uses text supervision. | — |
| ACL 2025 | [SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction](https://aclanthology.org/2025.acl-long.1397/) | Representation learning | Masked cluster prediction over multiple articulator streams learns representations from unlabelled signing videos. | — |
| Findings of ACL 2025 | [SignMusketeers: An Efficient Multi-Stream Approach for Sign Language Translation at Scale](https://aclanthology.org/2025.findings-acl.1157/) | SLT / efficiency | Combines face, hand and body streams with efficient self-supervised frame representations for translation at scale. | — |
| ICCV Workshops 2025 | [SAGE: Segment-Aware Gloss-Free Encoding for Token-Efficient Sign Language Translation](https://openaccess.thecvf.com/content/ICCV2025W/MSLR/html/Low_SAGE_Segment-Aware_Gloss-Free_Encoding_for_Token-Efficient_Sign_Language_Translation_ICCVW_2025_paper.html) | SLT / efficiency | Segment-aware visual encoding and token-level alignment reduce the visual token budget for translation. | — |
| WSLP 2026 (workshop) | [TransAlign-SLT: Complementary Temporal and Semantic Supervision for Gloss-Free Sign Language Translation](https://aclanthology.org/2026.wslp-1.21/) | Gloss-free SLT | Combines auxiliary temporal supervision with training-time semantic alignment; the semantic branch is absent at inference. | — |
| EMNLP 2025 | [PoseStitch-SLT: Linguistically Inspired Pose-Stitching for End-to-End Sign Language Translation](https://aclanthology.org/2025.emnlp-main.698/) | Pose-to-text SLT | Linguistically motivated pose stitching provides synthetic pretraining data; compare its pose-only setting separately from RGB systems. | — |

### LLMs, retrieval, reinforcement learning and context

| Publication / status | Paper | Task | What to read it for | Verified code / resource |
|---|---|---|---|---|
| arXiv 2026 — preprint | [SignRAG: Unified Retrieval-Augmented Gloss-Free Sign Language Translation](https://arxiv.org/abs/2610.11371) | Gloss-free SLT / retrieval | Combines hierarchical pretraining, a training-set retrieval gallery and retrieval-utility-guided reinforcement fine-tuning. | [Author resource](https://github.com/shahelaojieraozhi/SignRAG) |
| Findings of EMNLP 2026 | [Attention-Steered Vision-Language Models for Sign Language Translation](https://aclanthology.org/2026.findings-emnlp.1435/) | SLT / VLM alignment | Steers spatial attention toward hands and faces and learns temporal frame selection with reinforcement learning. | — |
| CVPR Findings 2026 | [RVLF: A Reinforcing Vision-Language Framework for Gloss-Free Sign Language Translation](https://openaccess.thecvf.com/content/CVPR2026F/html/Rao_FVLF_A_Reinforcing_Vision-Language_Framework_for_Gloss-Free_Sign_Language_Translation_CVPRF_2026_paper.html) | Gloss-free SLT / RL | RVLF combines RGB and skeleton features with supervised fine-tuning and reward-guided language generation. | — |
| CVPR 2026 | [BoostSLT: Boosting Sign Language Translation via a Plug-and-Play Diffusion-Based Semantic Enhancer](https://openaccess.thecvf.com/content/CVPR2026/html/Han_BoostSLT_Boosting_Sign_Language_Translation_via_a_Plug-and-Play_Diffusion-Based_Semantic_CVPR_2026_paper.html) | SLT / long inputs | Uses temporal segmentation and diffusion-based semantic reconstruction as an enhancer for translation models. | [Author resource](https://github.com/K1sna/BoostSLT) |
| Findings of ACL 2026 | [Can Small Vision–Language Models Perform Sign Language Translation?](https://aclanthology.org/2026.findings-acl.1609/) | SLT / small VLM evaluation | Evaluates adapted small vision-language models across sign languages, including semantic and entity-sensitive evaluation. | — |
| WACV 2026 | [Large Sign Language Models: Toward 3D American Sign Language Translation](https://openaccess.thecvf.com/content/WACV2026/html/Zhang_Large_Sign_Language_Models_Toward_3D_American_Sign_Language_Translation_WACV_2026_paper.html) | 3D ASL-to-text SLT | Connects 3D signing representations to an instruction-guided language model; input modality differs from RGB-only SLT. | — |

### Production, discourse and bidirectional modelling

| Publication / status | Paper | Task | What to read it for | Verified code / resource |
|---|---|---|---|---|
| Findings of EMNLP 2026 | [SignFLIP: A Unified Model for Sign Language Translation and Generation via Stage-wise Alignment at Scale](https://aclanthology.org/2026.findings-emnlp.828/) | SLT + SLP | Stage-wise alignment supports translation and generation around a shared language model. | — |
| arXiv 2026 — preprint | [SignGPT: Toward LLM-Mediated Sign Language Interaction through Gloss-Free Translation and Generation](https://arxiv.org/abs/2609.21709) | SLT + SLP | Uses part-aware pose representations and a shared language model for bidirectional sign-language interaction. | — |
| EMNLP 2026 | [DiscoSign: Discourse-Aware Text to Sign Language Gloss Translation](https://aclanthology.org/2026.emnlp-main.183/) | Text-to-gloss | Models discourse phenomena such as spatial reference and consistent gloss choices; this is not video-to-text SLT. | — |
| ICCV 2025 | [Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator](https://openaccess.thecvf.com/content/ICCV2025/html/Zuo_Signs_as_Tokens_A_Retrieval-Enhanced_Multilingual_Sign_Language_Generator_ICCV_2025_paper.html) | Multilingual SLP / retrieval | Uses retrieved sign units as tokens for multilingual sign-language generation. | — |
| arXiv 2026; BMVC 2026 acceptance reported by authors | [SignRR: Retrieve and Refine Real Motion for Sign Language Production](https://arxiv.org/abs/2608.28568) | SLP / retrieval | Retrieves real signing segments and refines the full motion sequence with a part-aware residual VQ-VAE. | — |
| arXiv 2026 — preprint | [SemRAG: Semantic-Guided Sentence-Level Retrieval-Augmented Sign Language Production](https://arxiv.org/abs/2611.18189) | SLP / retrieval | Retrieves sentence-level motion templates and uses semantic guidance to localize and regenerate relevant motion. | — |

### Datasets, multilingual learning and evaluation

| Publication / status | Paper | Task | What to read it for | Verified code / resource |
|---|---|---|---|---|
| ACL 2026 | [CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language](https://aclanthology.org/2026.acl-long.1896/) | Chinese SL understanding benchmark | Dictionary-grounded text, image and video tasks examine fine-grained sign understanding in multimodal language models. | [Author resource](https://github.com/rzhao-zhsq/CNSL-bench) |
| NAACL 2025 | [FLEURS-ASL: Including American Sign Language in Massively Multilingual Multitask Evaluation](https://aclanthology.org/2025.naacl-long.314/) | ASL evaluation benchmark | Connects professionally interpreted ASL to multilingual text and speech resources for sentence and discourse evaluation. | — |
| ICLR 2025 | [YouTube-SL-25: A Large-Scale, Open-Domain Multilingual Sign Language Parallel Corpus](https://proceedings.iclr.cc/paper_files/paper/2025/file/cbbb65dc108e8ac2f82cba25bc5992fc-Paper-Conference.pdf) | Multilingual parallel corpus | Provides over 3,000 hours spanning over 25 sign languages; automatically collected captions require alignment and noise checks. | [Author resource](https://github.com/google-research/google-research/tree/master/youtube_sl_25) |
| ACL 2025 (Short Papers) | [Multilingual Gloss-free Sign Language Translation: Towards Building a Sign Language Foundation Model](https://aclanthology.org/2025.acl-short.43/) | Multilingual gloss-free SLT | Studies translation across ten sign languages using a shared model and language-aware auxiliary supervision. | — |
| Findings of EMNLP 2025 | [Rethinking Sign Language Translation: The Impact of Signer Dependence on Model Evaluation](https://aclanthology.org/2025.findings-emnlp.997/) | SLT / evaluation protocol | Shows substantial performance losses under signer-independent evaluation and examines recurring target sentences across splits. | — |
| CVPR Workshops 2026 | [SignEval 2026 Challenge Results](https://openaccess.thecvf.com/content/CVPR2026W/MSLR/html/Hasanaath_SignEval_2026_Challenge_Results_CVPRW_2026_paper.html) | ISLR / CSLR challenge | Reports recognition challenge results for Saudi signing and Italian medical communication datasets; not an SLT leaderboard. | — |
| CVPR Workshops 2026 | [Sign Language Recognition in the Age of LLMs](https://openaccess.thecvf.com/content/CVPR2026W/MSLR/html/Javorek_Sign_Language_Recognition_in_the_Age_of_LLMs_CVPRW_2026_paper.html) | ISLR / VLM evaluation | Compares language-model-based recognition with task-specific models, highlighting limitations of zero-shot sign classification. | — |

**Reading cautions:** SignDINO here is Gan et al.'s CVPR 2026 paper; similarly named projects must not be assumed to be its code. The current CVF title for RVLF differs from the legacy `FVLF` filename in its page URL. SignRAG is retained as a preprint because its claimed future conference status was not independently verified. SignRR's BMVC 2026 acceptance is explicitly author-reported on arXiv. FLEURS-ASL's multilingual text/speech coverage does not imply hundreds of sign languages.

**Comparing results:** record the split, signer overlap, duplicate target sentences, input modality, gloss supervision, external pretraining data and retrieval-gallery construction. Report ISLR accuracy, CSLR WER, SLT text/semantic metrics and SLP motion/comprehension measures separately. A better score in one paper is not evidence of superiority under unmatched training or evaluation conditions.

## 📖 Table of Contents
- [Latest papers and 2026 conference updates](#latest-work)
- [Verified additions: 2025–2026](#verified-additions)
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
  - [Sign Language Production](#Dataset_3)
  - [Unified Understanding](#Dataset_4)
  - ......
- [💻 Others](#Others) 
- [🖊️ Citation](#Citation)

## 📚 Sign Language Tasks and Methods <a id="methods-a-survey"></a>

### Isolated Sign Recognition

The older table also contains general-purpose video backbones used by sign-language systems; those are supporting methods, not dedicated sign-recognition benchmarks. Entries outside the dated update remain subject to verification.
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
| 2023.08  |   PiSLTRc       | [PiSLTRc: Position-informed Sign Language Transformer with Content-aware Convolution](https://arxiv.org/abs/2107.12600) (ICCV)                                                              |       ViT             |    Position-informed    |                        N/A                         |
| 2024.07  |   Cross-lingual | [Cross-lingual few-shot sign language recognition](https://linkinghub.elsevier.com/retrieve/pii/S0031320324001250) (Pattern Recognition)                                                                     |    Cross-lingual      |     Few-shot Learning   |                        N/A                         |
| 2024.08  |   Self-Attn-SLR | [A Static Sign Language Recognition Method Enhanced with Self-Attention Mechanisms](https://pubmed.ncbi.nlm.nih.gov/39517818/) (Sensors)                                                                 |      CNN+Self-Attn    |    Self-Attention       |                        N/A                         |

### Continuous Sign Recognition
|   Time   |   Model Name    | Paper Title                                                                                                                                                                            |    Visual Encoder     |         Decoder         |                    Code/Project                    |    
|:--------:|:---------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------:|:-----------------------:|:--------------------------------------------------:|
| 2014.09  |    CNN-HMM      | [Deep Sign: Hybrid CNN-HMM for Continuous Sign Language Recognition](https://openresearch.surrey.ac.uk/esploro/outputs/conferencePresentation/Deep-Sign-Hybrid-CNN-HMM-for-Continuous/99511412402346) (BMVC) | CNN | HMM | N/A |
| 2015.12  |   HMM-based     | [Continuous Sign Language Recognition: Towards Large Vocabulary Statistical Recognition Systems Handling Multiple Signers](https://www.sciencedirect.com/science/article/pii/S1077314215002088) (CVIU) | Hand-crafted Features | HMM | N/A |
| 2017.04  |   Hand-Action   | [First-Person Hand Action Benchmark with RGB-D Videosnd 3D Hand Pose Annotations](https://arxiv.org/abs/1704.02463) (CVPR)                                                               |     RGB-D+Pose       |         CTC             |                        N/A                         |
| 2017.10  |   Egocentric    | [Egocentric Gesture Recognition Using Recurrent 3D Convolutional Neural Networks with Spatiotemporal Transformer Modules](https://ieeexplore.ieee.org/document/8237668/) (ICCV)           |      3D CNN           |    Spatiotemporal       |                        N/A                         |
| 2017.10  |     SubUNets    | [SubUNets: End-to-end Hand Shape and Continuous Sign Language Recognition](https://openaccess.thecvf.com/content_iccv_2017/html/Camgoz_SubUNets_End-To-End_Hand_ICCV_2017_paper.html) (ICCV)|      CNN              |        CTC              |                        N/A                         |
| 2018.09  |   CNN-LSTM      | [Neural Sign Language Translation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Camgoz_Neural_Sign_Language_CVPR_2018_paper.pdf) (CVPR)                                          |        CNN            |         LSTM            |                        N/A                         |
| 2019.06  |    Spatial-Temp | [Spatial-Temporal Multi-Cue Network for Continuous Sign Language Recognition](https://arxiv.org/abs/2002.03187) (AAAI)                                                |      3D CNN           |    Multi-cue Learning   |                        N/A                         |
| 2020.02  |    Iterative    | [Iterative Alignment Network for Continuous Sign Language Recognition](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pu_Iterative_Alignment_Network_for_Continuous_Sign_Language_Recognition_CVPR_2019_paper.pdf) (CVPR) | ResNet | Iterative CTC | N/A |
| 2020.07  |     STMC        | [Stochastic Fine-grained Labeling of Multi-state Sign Glosses for Continuous Sign Language Recognition](https://link.springer.com/chapter/10.1007/978-3-030-58517-4_11) (ECCV)            |      CNN              |    HMM-DNN              |                        N/A                         |
| 2021.03  |    VAC (CNN)    | [Visual Alignment Constraint for Continuous Sign Language Recognition](https://openaccess.thecvf.com/content/ICCV2021/papers/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.pdf) (ICCV)| CNN | CTC + Alignment | [link](https://github.com/ycmin95/VAC_CSLR) |
| 2021.08  |     BN-TRN      | [Fully Convolutional Networks for Continuous Sign Language Recognition](https://arxiv.org/abs/2007.12402) (ECCV)                                              |      ResNet           |        FCN              |                        N/A                         |
| 2021.10  |    CVT-SLR      | [CVT-SLR: Contrastive Visual-Textual Transformation for Sign Language Recognition with Variational Alignment](https://arxiv.org/abs/2303.05725) (CVPR) | ResNet+Transformer | Contrastive | N/A |
| 2023.06  |    CorrNet      | [Continuous Sign Language Recognition with Correlation Network](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_Continuous_Sign_Language_Recognition_With_Correlation_Network_CVPR_2023_paper.pdf) (CVPR)                                      |    Multi-Features     |     Correlation         |                        N/A                         |
| 2022  |     TwoStream-SLR | [Two-Stream Network for Sign Language Recognition and Translation](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6cd3ac24cdb789beeaa9f7145670fcae-Abstract-Conference.html) (NeurIPS) | RGB+Keypoint (S3D streams) | CTC | [link](https://github.com/FangyunWei/SLRT) |
| 2022.06  |    C2SLR        | [C2SLR: Consistency-enhanced Continuous Sign Language Recognition](https://openaccess.thecvf.com/content/CVPR2022/papers/Zuo_C2SLR_Consistency-Enhanced_Continuous_Sign_Language_Recognition_CVPR_2022_paper.pdf) (CVPR) | ResNet | Consistency Learning | N/A |
| 2022.10  |   TLP-CTC       | [Temporal Lift Pooling for Continuous Sign Language Recognition](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950506.pdf) (ECCV)                                                                                    |      ResNet           |    Temporal Pooling     |                        [link](https://github.com/hulianyuyy/Temporal-Lift-Pooling)                         |
| 2021.10  |    SMKD         | [Self-Mutual Distillation Learning for Continuous Sign Language Recognition](https://openaccess.thecvf.com/content/ICCV2021/papers/Hao_Self-Mutual_Distillation_Learning_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.pdf) (ICCV)                                                                      |      ResNet           |  Knowledge Distillation |                        N/A                         |
| 2024.06  |  AdaBrowse      | [AdaBrowse: Adaptive Video Browser for Efficient Continuous Sign Language Recognition](https://dl.acm.org/doi/10.1145/3581783.3612269) (ACMMM)                                             |      CNN              |  Adaptive Browsing     |                        N/A                         |
| 2024.08  |  Topic-Detection| [Topic Detection in Continuous Sign Language Videos](https://arxiv.org/abs/2408.15033) (arXiv)                                                                                               |      ResNet           |  Topic Modeling        |                        N/A                         |
| 2025  |   MSKA-SLR      | [Multi-Stream Keypoint Attention Network for Sign Language Recognition and Translation](https://doi.org/10.1016/j.patcog.2025.111602) (Pattern Recognition)                                                            |    Multi-stream       |   Keypoint Attention    |   [link](https://github.com/sutwangyan/MSKA)      |


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
| 2024  |     Sign2GPT    | [Sign2GPT: Leveraging Large Language Models for Gloss-free Sign Language Translation](https://openreview.net/forum?id=LqaEEs3UxU) (ICLR)                                                              | See paper | See paper |   [link](https://github.com/ryanwongsa/Sign2GPT) |
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
| 2023.10  |     GFSLT-VLP   | [Gloss-free Sign Language Translation: Improving from Visual-Language Pretraining](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Gloss-Free_Sign_Language_Translation_Improving_from_Visual-Language_Pretraining_ICCV_2023_paper.pdf) (ICCV) | CLIP | mBART | [link](https://github.com/zhoubenjia/GFSLT-VLP) |
| 2024.11  |   Signformer    | [Signformer is all you need: Towards Edge AI for Sign Language](https://arxiv.org/abs/2411.12901) (arXiv)                                                                                     |      Efficient        |      Edge Computing         |   [link](https://github.com/EtaEnding/Signformer) |
| 2024.12  | Scaling-SLT     | [Scaling Sign Language Translation](https://proceedings.neurips.cc/paper_files/paper/2024/file/ced76a666704e381c3039871ffe558ee-Paper-Conference.pdf) (NeurIPS)                                          |      Large-scale      |       Scaling Methods       |                        N/A                         |
| 2025.01  |  Text-CTC-Align | [Improvement in Sign Language Translation Using Text CTC Alignment](https://aclanthology.org/2025.coling-main.219/) (COLING)                                                                   |      ResNet           |         CTC Alignment       |   N/A   |
| 2025.03  |  Stochastic-Tr  | [Stochastic Transformer Networks with Linear Competing Units: Application to end-to-end SL Translation](https://ieeexplore.ieee.org/document/9709998/) (TPAMI)                               |      CNN              |    Stochastic Transformer   |                        [link](https://github.com/avoskou/Stochastic-Transformer-Networks-with-Linear-Competing-Units-Application-to-end-to-end-SL-Translatio/actions)                         |
| 2025.06  | Lost-in-Context | [Lost in Translation, Found in Context: Sign Language Translation with Contextual Cues](https://openaccess.thecvf.com/content/CVPR2025/html/Jang_Lost_in_Translation_Found_in_Context_Sign_Language_Translation_with_CVPR_2025_paper.html) (CVPR)                                                                    |       CNN             |      Contextual Learning    |        |
| 2020  |     TSPNet      | [TSPNet: Hierarchical Feature Learning via Temporal Semantic Pyramid for Sign Language Translation](https://papers.nips.cc/paper_files/paper/2020/file/8c00dee24c9878fea090ed070b44f1ab-Paper.pdf) (NeurIPS) |       3D CNN        |    Hierarchical         |   [link](https://github.com/verashira/TSPNet)     |


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
| 2025  |    Uni-Sign     | [Uni-Sign: Toward Unified Sign Language Understanding at Scale](https://proceedings.iclr.cc/paper_files/paper/2025/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html) (ICLR)                                                                          | Pose GCN + RGB fusion | ISLR + CSLR + SLT | [link](https://github.com/ZechengLi19/Uni-Sign) |
| 2024.12  |   Multimodal    | [Scaling Up Multimodal Pretraining for Sign Language Understanding](https://arxiv.org/abs/2408.08544) (arXiv)                                                                                | Multimodal Pretraining | Cross-domain Transfer | N/A |

## 👨‍🏫 Large Language Models Meets Sign Language

### LLM-driven Gloss-free Translation
|   Time   |   Method    | Paper Title                                                                                                                                                                            |      Visual Encoder      |     LLM     |  Fine-tuning  |                      Code/Project                      |
|:--------:|:-----------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------:|:-----------:|:-------------:|:------------------------------------------------------:|
| 2024  |  Sign2GPT   | [Sign2GPT: Leveraging Large Language Models for Gloss-free Sign Language Translation](https://openreview.net/forum?id=LqaEEs3UxU) (ICLR)                                                              | See paper | See paper | See paper | [link](https://github.com/ryanwongsa/Sign2GPT) |
| 2024.06  | LLMs-Good-SL| [LLMs are Good Sign Language Translators](https://arxiv.org/abs/2404.00925) (CVPR)                   | ViT | LLaMA/GPT | LoRA | N/A |
| 2024.09  | Visual-Align| [Visual Alignment Pretraining for Sign Language Translation](https://eccv.ecva.net/virtual/2024/poster/737) (ECCV)                                                                                        | CLIP | Transformer | Frozen | N/A |
| 2024.12 | LLaVA-SLT | LLaVA-SLT: Visual Language Tuning for Sign Language Translation (arXiv) | （DINOv2 ViT-B/14 — CLIP-style contrastive pre-training） | （Qwen-2.5-Instruct 3 B / 7 B / 14 B） | （视觉编码器与 LLM 全部 Frozen，仅两层 MLP 连接器可训练） | N/A | 
| 2024.10  |Factorized-LLM| [Factorized Learning Assisted by Large Language Model for Gloss-free Sign Language Translation](https://aclanthology.org/2024.lrec-main.620/) (LREC-COLING)                                       | CNN | LLM | Adapter | N/A |
| 2025.4  |SpaMo| [An Efficient Gloss-Free Sign Language Translation Using Spatial Configurations and Motion Dynamics with LLMs](https://aclanthology.org/2025.naacl-long.197/) (NAACL)                                       | CNN | LLM | Adapter | N/A |


### Unified Sign Language Foundation Models

|   Time   |    Method    | Paper Title                                                                                                                                                                 |        Visual Encoder         |     LLM     |    Fine-tuning    |                     Code/Project                      |
|:--------:|:------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------:|:-----------:|:-----------------:|:-----------------------------------------------------:|
| 2023.12  | SignBERT+ | [SignBERT+: Hand-Model-Aware Self-Supervised Pre-training for Sign Language Understanding](https://www.computer.org/csdl/journal/tp/2023/09/10109128/1MESMCioCQM) (TPAMI) | Hand-aware CNN | BERT | Self-supervised | [link](https://github.com/YinAoXiong/SLRT_FET) |
| 2025  |  Uni-Sign  | [Uni-Sign: Toward Unified Sign Language Understanding at Scale](https://proceedings.iclr.cc/paper_files/paper/2025/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html) (ICLR) | Pose GCN + RGB fusion | mT5-Base | Multi-task | [link](https://github.com/ZechengLi19/Uni-Sign) |
| 2024.12  | Multimodal-Pretraining | [Scaling Up Multimodal Pretraining for Sign Language Understanding](https://arxiv.org/abs/2408.08544) (arXiv) | Multimodal | Large Scale | Cross-domain | N/A |

### Visual-Language Alignment
| **Time** |  **Method**   |                                                                             Paper Title                                                                              |             **Alignment Method**              |                              **Code**                              |
|:--------:|:-------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------:|:------------------------------------------------------------------:|
| 2023.10  |   GFSLT-VLP   | [Gloss-free Sign Language Translation: Improving from Visual-Language Pretraining](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Gloss-Free_Sign_Language_Translation_Improving_from_Visual-Language_Pretraining_ICCV_2023_paper.pdf) (ICCV) | CLIP-style Pretraining | [link](https://github.com/zhoubenjia/GFSLT-VLP) |
| 2024.06  |   CiCo-Cross  | [CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_CiCo_Domain-Aware_Sign_Language_Retrieval_via_Cross-Lingual_Contrastive_Learning_CVPR_2023_paper.pdf) (CVPR) | Cross-lingual Contrastive | [link](https://github.com/FangyunWei/SLRT) |
| 2024.12  |   Multimodal  | [Scaling Up Multimodal Pretraining for Sign Language Understanding](https://arxiv.org/abs/2408.08544) (arXiv) | Large-scale Multimodal | N/A |

## 📊 Dataset <a id="Dataset"></a>

### Isolated Word Recognition  <a id="Dataset_1"></a>
|   Dataset   | Lang | Vocab | Videos | Signers | Res | Source | Anno | Link |
|:-----------:|:----:|:-----:|:------:|:-------:|:---:|:------:|:----:|:----:|
|  Purdue RVL-SLLL | ASL | 184 | 3.6K | 14 | Var | SR | M | — |
|  RWTH-BOSTON-50 | ASL | 50 | 483 | 3 | Var | SR | M | — |
|  GSLC | GSL | — | — | 4 | Var | SR | M | — |
|  Boston ASLLVD | ASL | 3.3K+ | 9.8K | 6 | Var | SR | M | — |
|  IIITA-ROBITA | ISL | 22 | 605 | — | Var | SR | M | — |
|  Auslan dataset | Auslan | — | — | 100 | Var | Var | M | — |
|  MSR Gesture 3D | ASL | 12 | 336 | 10 | Var | Kinect | M | — |
|  DGS Kinect 40 | DGS | 40 | 3.0K | 15 | Var | Kinect | M | — |
|  Dicta-Sign | Multi | — | — | 14-16/SL | Var | Var | M | — |
|  PSL Kinect 30 | PSL | 30 | 300 | 10 | Var | Kinect | M | — |
|  ASLG-PC12 | ASL | — | — | — | Var | Var | M | — |
|  CUNY ASL | ASL | — | — | 8 | Var | SR | M | — |
|  DEVISIGN-G | CSL | 36 | 300 | 8 | Var | SR | M | — |
|  DEVISIGN-D | CSL | 500 | 2.0K | 8 | Var | SR | M | — |
|  DEVISIGN-L | CSL | 2.0K | 24.0K | 8 | Var | SR | M | — |
|  PSL ToF 84 | PSL | 84 | 1.7K | 1 | Var | ToF | M | — |
|  LSA64 | LSA | 64 | 3.2K | 10 | Var | SR | M | — |
|  Indian Kinect | ISL | 140 | 5.0K | 18 | Var | Kinect | M | — |
|  BosphorusSign | TSL | 855 | — | 10 | Var | SR | M | — |
|  USTC CSL dataset | CSL | 178 | — | 50 | Var | SR | M | — |
|  WLASL | ASL | 2.0K | 21.0K | 119 | Var | Var | M | [🔗](https://github.com/dxli94/WLASL) |
|  WLASL100 | ASL | 100 | 2.0K | 97 | Var | Var | M | — |
|  WLASL300 | ASL | 300 | 5.1K | 109 | Var | Var | M | — |
|  WLASL1000 | ASL | 1.0K | 13.2K | 116 | Var | Var | M | — |
|  MS-ASL | ASL | 1.0K | 25.5K | 222 | Var | Kinect | M | [🔗](https://www.microsoft.com/en-us/research/project/ms-asl/) |
|  ASL-100-RGBD | ASL | 100 | 4.2K | 22 | RGB-D | RGBD | M | — |
|  DGS Korpus | DGS | — | — | 330 | Var | Var | M | — |
|  BosphorusSign22k | TSL | 744 | 22.5K | 6 | Var | SR | M | — |
|  AUTSL | TSL | 226 | 36.3K | 43 | Var | SR | M | — |
|  K-RSL | RSL | 600 | — | 11 | Var | Var | M | — |
|  GSL | GSL | 310 | 40.0K | 7 | Var | Var | M | [🔗](https://vcl.iti.gr/dataset/gsl/) |
|  NMFs-CSL | CSL | 1.1K | 32.0K | 10 | Var | SR | M | [🔗](https://ustc-slr.github.io/datasets/) |
|  SLR500 | CSL | 500 | 125.0K | — | 1280×720 | Var | M | [🔗](https://ustc-slr.github.io/datasets/) |
|  ASL Citizen | ASL | 2.7K | 84.0K | — | Var | Crowd | M | [🔗](https://www.microsoft.com/en-us/research/project/asl-citizen/) |
|  Slovo | RSL | 1.0K | 20.0K | — | 1920×1080 | SR | M | [🔗](https://github.com/hukenovs/slovo) |
|  BOBSL | BSL | 61.0K | 1.2M | >100 | Var | TV | A+M | [🔗](https://www.robots.ox.ac.uk/~vgg/data/bobsl/) |
|  Auslan-Daily | Auslan | 2.7K | 3.0K | — | 1920×1080 | Daily | M | [🔗](https://uq-cvlab.github.io/Auslan-Daily-Dataset/) |
|  BVCSL3D | ISL | 200 | 20.0K | 10 | Var | 3D | M | — |
|  KSL DATASET | KSL | 77 | 1.2K | 20 | Var | SR | M | — |
|  KArSL | ArSL | 502 | 75.3K | 3 | Var | SR | M | — |
|  AUSLAN | Auslan | 1.0K | 1.1K | 6 | Var | SR | M | — |
|  BSL Corpus | BSL | 5.0K | — | 100 | Var | Var | M | — |
|  PSL Kinect | PSL | 30 | 300 | 6 | Var | Kinect | M | — |
|  LSE-sign | LSA | 2.4K | 24.4K | 6 | Var | SR | M | — |
|  USTC-ICSL | CSL | 500 | — | 50 | Var | SR | M | — |
|  DISFA | DGS | 30 | 300 | 6 | Var | SR | M | — |
|  SMILE | DGS | 1.0K | 32.0K | 10 | Var | SR | M | — |
|  CSSL5000 | CSL | 1.0K | 100.0K | 50 | Var | SR | M | — |
|  BSL-1K | BSL | 1.0K | 1.0K | 10 | Var | SR | M | — |
|  INCLUDE | CSL | 263 | 4.3K | 7 | Var | SR | M | — |
|  WLASL-LEX | ASL | 800 | 10.0K | 119 | Var | Var | M | — |

**Abbreviations:**
- **Lang**: Language (ASL=American SL, BSL=British SL, CSL=Chinese SL, DGS=German SL, GSL=Greek SL, ISL=Indian SL, KSL=Korean SL, PSL=Polish SL, RSL=Russian SL, TSL=Turkish SL, LSA=Argentinian SL, ArSL=Arabic SL)
- **Vocab**: Vocabulary Size (K=thousand)
- **Videos**: Number of Videos (K=thousand, M=million)
- **Signers**: Number of Signers
- **Res**: Resolution (Var=Variable, RGB-D=RGB+Depth)
- **Source**: Data Source (SR=Studio Recorded, TV=TV Broadcast, Crowd=Crowdsourced, Daily=Daily Conversation, ToF=Time-of-Flight Camera, 3D=3D Camera)
- **Anno**: Annotation (M=Manual, A=Automatic, A+M=Automatic+Manual)
- **Link**: 🔗=Available, —=Not Available

### Continuous Recognition <a id="Dataset_2"></a>
|   Dataset   | Lang | Vocab | Videos | Signers | Length | Source | Anno | Link |
|:-----------:|:----:|:-----:|:------:|:-------:|:------:|:------:|:----:|:----:|
|  GSLC | GSL | — | — | 4 | Var | SR | M | — |
|  SIGNUM | DGS | 450 | 780 | 20 | Var | SR | M | — |
|  Corpus NGT | NGT | — | — | 92 | Var | Var | M | — |
|  RWTH-BOSTON-104 | ASL | 168 | 201 | 3 | Var | SR | M | — |
|  RWTH-BOSTON-400 | ASL | 593 | 843 | 4 | Var | SR | M | — |
|  Auslan dataset | Auslan | — | — | 100 | Var | Var | M | — |
|  RWTH-Phoenix-Weather | DGS | 911 | 2.0K | 7 | Var | Weather | M | — |
|  Dicta-Sign | Multi | — | — | 14-16/SL | Var | Var | M | — |
|  BSL Corpus | BSL | — | — | 249 | Var | Var | M | — |
|  S-pot | FinSL | 1.2K | 5.4K | 5 | Var | Var | M | — |
|  Phoenix-2014 | DGS | 1.2K | 6.8K | 50 | 3.2 sent | Weather | M | [🔗](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) |
|  RWTH-Phoenix-Weather-2014 | DGS | 1.1K | 6.8K | 9 | Var | Weather | M | [🔗](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) |
|  Phoenix-2014T | DGS | 1.2K | 8.2K | 9 | 3.2 sent | Weather | M | [🔗](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) |
|  RWTH-Phoenix-2014T | DGS | 1.2K | 6.3K | 9 | Var | Weather | M | [🔗](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) |
|  How2Sign | ASL | 16.0K | 60.0K | 11 | Var | YouTube | M | [🔗](https://how2sign.github.io/) |
|  CSL-Daily | CSL | 2.0K | 20.7K | 10 | 5.4 sent | Daily | M | [🔗](https://ustc-slr.github.io/datasets/) |
|  GSL | GSL | 310 | 10.3K | 7 | Var | Var | M | [🔗](https://vcl.iti.gr/dataset/gsl/) |
|  TVB-HKSL-News | HKSL | 3.0K | 16.1K | — | Var | TV | M | — |
|  SignsWorld Atlas | ArSL | 264+ | 535+ | 10 | Var | SR | M | — |
|  MS-ASL100 | ASL | 100 | 5.7K | 189 | Var | Kinect | M | — |
|  MS-ASL200 | ASL | 200 | 9.7K | 196 | Var | Kinect | M | — |
|  MS-ASL500 | ASL | 500 | 17.8K | 2.2K | Var | Kinect | M | — |
|  MS-ASL1000 | ASL | 1.0K | 25.5K | 222 | Var | Kinect | M | — |
|  TheRuSLan Kinect | RSL | 164 | — | 13 | Var | Kinect | M | — |
|  Boston-104 | ASL | 128 | 214 | 6 | Var | SR | M | — |
|  USTC-CCSL | CSL | 1.2K | 25.0K | 50 | Var | SR | M | — |
|  RCSD | CSL | 242 | 4.2K | 10 | Var | SR | M | — |
|  KETI | KSL | 419 | 14.7K | 14 | Var | SR | M | — |
|  MEDIAPI-SKEL | BSL | 14.4K | 36.9K | >100 | Var | TV | M | — |
|  OpenASL | ASL | 22.0K | 288h videos | — | Var | Var | M | [🔗](https://github.com/chevalierNoir/OpenASL) |
|  Auslan-Daily | Auslan | 2.7K | 14.0K | — | Var | Daily | M | [🔗](https://uq-cvlab.github.io/Auslan-Daily-Dataset/) |
|  YouTube-ASL | ASL | 11.0K | 984h videos | — | Var | YouTube | A+M | [🔗](https://github.com/google-research/google-research/tree/master/youtube_asl) |
|  LSA-T | LSA | 3.0K | 14.9K | — | Var | YouTube | M | [🔗](https://github.com/midusi/LSA-T) |
|  VECSL | CSL | 2.6K | 15.7K | — | Var | DVS346 | M | [🔗](https://github.com/Event-AHU/OpenESL) |
|  Event-SLT | CSL | 3.0K | 20.0K | — | Var | DVS346 | M | [🔗](https://github.com/Event-AHU/Event-SLT) |

**Abbreviations:**
- **Lang**: Language (NGT=Dutch SL, FinSL=Finnish SL, HKSL=Hong Kong SL, others same as above)
- **Vocab**: Vocabulary Size (K=thousand)
- **Videos**: Number of Videos (K=thousand)
- **Signers**: Number of Signers (K=thousand)
- **Length**: Average Length (sent=sentences, Var=Variable)
- **Source**: Data Source (Weather=Weather Broadcast, YouTube=YouTube Videos, DVS346=Event Camera, others same as above)
- **Anno**: Annotation (M=Manual, A=Automatic)
- **Link**: 🔗=Available, —=Not Available

### Sign Language Production <a id="Dataset_3"></a>
|   Dataset   | Purpose | Size | Input Type | Output Type | Source | Anno | Link |
|:-----------:|:-------:|:----:|:----------:|:-----------:|:------:|:----:|:----:|
|  SignAvatars | Generation | 8K videos | Text/Gloss | Video | Synthetic | A | [🔗](https://github.com/ZhengdiYu/SignAvatars) |
|  T2S-GPT | Generation | Phoenix subset | Text | Pose sequence | Phoenix-2014T | M | [🔗](https://t2sgpt-demo.yinaoxiong.cn/) |


**Abbreviations:**
- **Purpose**: Dataset Purpose (Generation=Sign Generation, Multi-task=Multiple Tasks)
- **Size**: Dataset Size (K=thousand)
- **Input Type**: Input Modality (Text=Natural Language Text, Gloss=Sign Glosses)
- **Output Type**: Output Modality (Video=Sign Language Video, Pose=Pose Sequences, SignWriting=Sign Writing System)
- **Source**: Data Source (Synthetic=Synthetically Generated, Phoenix-2014T=Phoenix Weather Dataset, Var=Various Sources)
- **Anno**: Annotation (M=Manual, A=Automatic)
- **Link**: 🔗=Available, —=Not Available


### Retrieval Tasks <a id="Dataset_5"></a>
| **Dataset** | **Language** | **Size** | **Query Type** | **Target Type** | **Data Source** | **Anno.** | **Link** |
|-------------|-------------|----------|----------------|-----------------|-----------------|-----------|---------|


### Unified Understanding <a id="Dataset_4"></a>
| **Dataset** | **Language** | **Tasks** | **Vocabulary Size** | **Number of Videos** | **Number of Signers** | **Modality** | **Data Source** | **Anno.** | **Link** |
|-------------|-------------|-----------|--------------------:|---------------------:|----------------------:|--------------|-----------------|-----------|---------|
| CSL-News (Uni-Sign) | CSL | ISLR + CSLR + SLT pretraining | — | 1,985 hours (not a clip count) | — | RGB+Pose | TV news | See paper | [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html) |



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
| 2021.05 | Domain-Adapt| [Data Augmentation for Sign Language Gloss Translation](https://arxiv.org/abs/2105.07476) (AT4SSL) | Domain Transfer | N/A |
| 2022.05 | VideoMix    | [VideoMix: Rethinking Data Augmentation for Video Classification](https://arxiv.org/abs/2012.03457) (CVPR) | Video Augmentation | [link](https://github.com/yhZhai/VideoMix) |
| 2023.10 | SLR-Aug     | [Preliminary Study on Data Augmentation Methods for Sign Language Recognition](https://www.researchgate.net/publication/375785218_Preliminary_Study_on_Data_Augmentation_Methods_for_Sign_Language_Recognition) (ICCAS) | Multi-modal Aug | N/A |
| 2024.05 | XmDA        | [Cross-modality Data Augmentation for End-to-End Sign Language Translation](https://arxiv.org/abs/2305.11096) (arXiv) | Cross-modal Aug | [link](https://github.com/Atrewin/SignXmDA) |
| 2025.06 | SignAug     | [Using Sign Language Production as Data Augmentation to enhance Sign Language Translation](https://arxiv.org/html/2506.09643v1) (arXiv) | Generative Aug | N/A |


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

To cite this repository (not a published survey paper):

```bibtex
@misc{jim3503_awesome_sign_2026,
      title={Awesome Sign Language: Research Papers, Datasets, and Resources},
      author={{Jim3503}},
      year={2026},
      howpublished={GitHub repository},
      url={https://github.com/Jim3503/Awesome_Sign}
}
```

## 🙏 Acknowledgments

This research collection is built upon and inspired by the excellent work from the [Awesome-Sign-Language](https://github.com/ZechengLi19/Awesome-Sign-Language) GitHub repository. We gratefully acknowledge the contributions of the sign language research community in maintaining and updating the literature collections.

Special thanks to:
- **ZechengLi19** and contributors for the [Awesome-Sign-Language](https://github.com/ZechengLi19/Awesome-Sign-Language) repository
- All researchers who have contributed to the advancement of sign language technology
- The sign language communities worldwide who provide valuable datasets and linguistic insights

If you have suggestions for improvements or want to contribute new papers, please feel free to submit issues or pull requests.




