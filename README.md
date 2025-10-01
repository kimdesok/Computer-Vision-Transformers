## Multimodal AI for Pathology: Vision and Language Models (in revision) <br>

![Workflow of weakly supervised WSI classification and analysis](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/71b9a546-fc3d-43d2-b422-c010d4c2abaa) <br>

**Figure 1.** Proposed multimodal pathology framework.  High-resolution WSIs are processed through transformer-based models (ViT and related architectures) to capture global and local histological features. Model outputs are visualized as heatmaps and quantitative morphometric analyses, which are then paired with NLP-driven processing of pathology reports. This integration bridges image-derived evidence with clinical text, enhancing interpretability, supporting pathologists’ decision-making, and improving workflow efficiency.


### A. CNN vs. ViT Models

* Convolutional neural networks (CNNs) primarily capture **local patterns**, while transformer-based vision models (ViTs) leverage **self-attention** to integrate **global context** across the entire image. As a result, the heatmaps derived from these models often highlight different histological features.
* Within the transformer family, models such as ViT, DeiT, Swin, PVT, CrossViT, and T2T-ViT provide complementary architectural advantages (e.g., hierarchical structure, efficient training, cross-scale representation) and are considered strong candidates for cancer tissue image classification tasks.
* Preliminary benchmarking supports this: a ViT model trained on the Food-101 dataset achieved **86% accuracy**, compared to **77% with ResNet-50** (see [Pilot Project 1](#pilot-project-1--food-image-classification-with-vision-transformers)). This aligns with reported gains in other domains where transformers outperform end-to-end CNNs.


### B. NLP of Pathology Reports  

While transformer-based vision models provide powerful tools for analyzing histological images, the full clinical utility emerges when these **image-derived insights are integrated with clinical text**. Pathology reports contain rich diagnostic context, and natural language processing (NLP) enables structured extraction and summarization of this information.  

NLP techniques, particularly those enabled by large-scale language models, can play a significant role in:  
- Interpreting and discussing model inference results,  
- Associating quantitative image features (e.g., nuclear morphology, tissue boundaries, lymphocyte infiltration, microvasculature, presence of mitotic, necrotic, and apoptotic cells) with patient-level outcomes,  
- Reporting these findings in a format that directly supports the pathologist’s decision-making workflow.  

This combined approach forms the basis of our proposed framework (see figure above): weakly labeled high-resolution WSI datasets (e.g., TCGA projects) are used to train ViT models for cancer-type classification, while NLP modules generate pathology report summaries that contextualize the image-based inferences. Together, these components enhance interpretability, strengthen clinical evidence, and ultimately improve workflow efficiency in pathology practice. 

## Overview of vision transformer:

![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/8a97f8fa-09ac-490d-ac6a-adb5c1846d6b)
**Figure 2.** High-level overview of the Vision Transformer (ViT) architecture* MLP: Multi-Layer Perceptron. Taken from the original paper [1] <br>

The vision transformer (ViT) can be understood as the **BERT of images**. Instead of processing words, ViT splits an image (e.g., 224×224) into a sequence of small patches (16×16 or 32×32), projects each into an embedding space, and adds a special [CLS] token for classification. Absolute position embeddings encode spatial information, and the sequence is passed into a Transformer encoder.  

Key mechanisms:  
- **Multi-Head Self-Attention (MHSA):** lets each patch attend to every other patch, capturing long-range dependencies and global context.  
- **Feed-Forward Networks (MLP):** interpret and combine these representations for the final prediction.  

By contrast, **CNNs** rely on progressively larger receptive fields from local convolutions, making them excellent at extracting edges, textures, and localized features, but less effective at capturing global relational structures.  

In histopathology, this distinction matters:  
- CNNs emphasize **local interactions** (edges, nuclei, blobs).  
- ViTs emphasize **global structure**, modeling interconnections between distant patches (e.g., stromal-tumor interactions, immune infiltration patterns).  

Thus, ViTs are particularly suited for **complex WSI structures**, where malignant tissues often disrupt normal architecture.  

---

### Summary Table: CNN vs. ViT vs. NLP in Pathology

| Component | Core Strength | Limitation | Role in Proposed Framework |
|-----------|--------------|------------|----------------------------|
| **CNN (ResNet, etc.)** | Strong at capturing local features (edges, nuclei, textures) | Limited global context, may underperform on complex tissue patterns | Baseline model, benchmark for patch-level accuracy |
| **ViT & Variants** | Self-attention integrates global + local features; scalable; interpretable heatmaps | Data-hungry, sensitive to weak supervision | Primary engine for WSI classification |
| **NLP (Pathology Reports)** | Extracts and structures clinical context; bridges human-readable insights | Requires domain-specific adaptation | Summarizes inference results and supports workflow efficiency |

---

### Brief Discussion → Experimental Hypothesis
ResNet-50 consistently shows lower accuracy (~0.70–0.80) compared to ViTs, likely due to weaker regularization and reduced generalization across datasets. We hypothesize that **ViT models trained on weakly labeled WSI datasets (TCGA)** will outperform CNN baselines in classification, and when coupled with NLP-generated clinical reporting, will yield a robust multimodal framework for pathology. [See the above model diagram][1].

---
### Note on Pilot Projects
Pilot Projects 1 and 2 are **preliminary studies** designed to demonstrate technical readiness, validate our workflows (data preprocessing, training, evaluation), and build familiarity with transformer-based models in both vision and language domains. These pilots should not be regarded as the main body of the proposed work.  

The central focus of this proposal is the **development of a multimodal framework** that combines:  
1. Vision Transformer (ViT)–based classification of weakly labeled WSI datasets (e.g., TCGA), and  
2. NLP-driven generation of pathology report summaries.  

The pilot projects serve as a foundation, showing feasibility and our ability to execute, but the main R&D effort will be dedicated to delivering this integrated multimodal pipeline during the project period.

## Pilot Project 1: Food Image Classification with Vision Transformers
<br>
As a **preliminary feasibility study**, we evaluated a transformer-based model (ViT) against a CNN baseline (ResNet-50) on the Food-101 dataset. The goal of this pilot was not to achieve state-of-the-art performance in food recognition, but rather to test our workflow for fine-tuning Hugging Face models, assess training efficiency, and establish a baseline comparison between transformer and CNN architectures.

**Summary of Pilot Project 1 (Food-101):**

| Aspect                | Details                                                                 |
|------------------------|-------------------------------------------------------------------------|
| **Dataset**           | Food-101 (101 categories, Hugging Face repository)                      |
| **Models compared**   | ViT (google/vit-base-patch16-224-in21k) vs. ResNet-50 (microsoft/resnet-50) |
| **Training setup**    | Fine-tuning pretrained models with identical preprocessing (augmentation, TF conversion) |
| **Accuracy**          | ViT: **86.2%** vs. ResNet-50: **76.7%**                                 |
| **Training time**     | ViT: **15.6 hrs** vs. ResNet-50: **30.9 hrs**                           |
| **Observations**      | ViT converged faster, achieved higher accuracy; ResNet showed signs of overfitting and would benefit from hyperparameter tuning. |

Here are some preliminary results of inference: <br>
![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/81f7622e-0fc0-413a-8751-064750942445)

![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/d2044dde-4632-40dd-94be-2159bdefe1c5)

## Pilot Project 2: Classification of Metastatic Breast Cancer Histologic Images
As a **second exploratory study**, we fine-tuned ViT and ResNet-50 models on the PatchCamelyon dataset. The purpose of this pilot was to extend our evaluation of CNNs vs. transformers into the domain of histopathology images. While the results are preliminary (limited epochs and constrained hyperparameter tuning), they highlight dataset-dependent performance patterns and reinforce the need for our proposed multimodal framework that integrates WSI classification with pathology report analysis.

**Summary of Pilot Project 2 (PatchCamelyon):**

| Aspect                | Details                                                                 |
|------------------------|-------------------------------------------------------------------------|
| **Dataset**           | PatchCamelyon (327,680 RGB images, 96×96 patches, Hugging Face: laurent/PatchCamelyon) |
| **Labeling**          | Positive if metastatic cells found in central 32×32 patch; otherwise negative |
| **Models compared**   | ViT vs. ResNet-50 (both Hugging Face pretrained models)                  |
| **Training setup**    | Fine-tuning for 5 epochs, batch sizes up to 64                           |
| **Preliminary results** | ViT slightly outperformed ResNet at larger batch sizes (≥64), but ResNet was stronger at batch size 32 |
| **Observations**      | Transformer advantage not guaranteed; performance may depend on dataset and optimization settings. Confirms need for extended experiments in WSIs. |

As a baseline model, a ResNet-50 model available as "microsoft/resnet-50" at Huggingface,  was trained in a fine tuning manner. 
The training dataset is available at Huggingface's Datasets titled as ['laurent/PatchCamelyon'](https://huggingface.co/datasets/1aurent/PatchCamelyon) It consists of 327,680 RGB color images with the size of 96x96 depicting the lymph node sections of locally metastasized breast cancers. Patch images were labeled as positive if the metastatic cells were found in the 32x32 square located in the center of the patch(See the figure below. The positive patch was highlighted in cyan in the center square).  Otherwise, they were labeled as negative.

PatchCamelyon images
![Image](https://github.com/basveeling/pcam/raw/master/pcam.jpg)

### Preliminary Results

Upon fine tuning the pretrained models with the PatchCamelyon dataset, it appears that the ViT model performs slightly better than in a number of benchmark criteria such as batch size as large as 64 or larger.  However ResNet 50 resulted in more accurate inference when batch size was 32 (See the table below).  Due to the time limitation, the training was performed only within five epochs. At this point, it seems that transformer models do not seem to perform always better than ResNet 50 models but its performance could be dependent on the dataset.
											
Performance of Image Classsification Models											
![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/7e9b5fa2-c3f0-435e-ad15-b9ad8dc658b5)

[Web application implementation]<br>
[Current Website:](https://c693-114-110-129-200.ngrok-free.app/) note: It may not load models listed below depending on the stage of the project.
- A web app. has been built to illustrate how accurately inference models can predict cancer images.
- The number of images can be selected to randomly select the input images from the test set of the patchcamelyon dataset.
- Models provided were VGG16, ResNet50, ResNet101, Xception, etc at the moment.
- Most .keras models were compiled successfully to .rbln models that are running on the ATOM NPUs.
- The compilation of ViT models was failed. Those huggingface provided models should be archived with some caution.
- The app provides the performance of the model in terms of accuracy, precision, recall, and f1-score.  In addition, throughput related values are also provided. 
- The output window provides the whole test images with the inference results. If the test image was incorrectly predicted, the index in the figure was highlighted in red.
![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/c44dcc69-e3de-427a-a1fb-d23faae2d9ca)
- The above app is provided for the illustration purpose only. The current version may be slightly different to the one introduced in this section.

### From Pilots to Proposed Work
Pilot Projects 1 and 2 serve as **proof-of-concept exercises**: they demonstrate our ability to train and evaluate both CNN and transformer models, implement preprocessing pipelines, and deploy inference systems on NPUs. However, they should not be interpreted as the primary contribution of this proposal.  

The **core R&D effort** will focus on developing a multimodal pipeline that:  
1. Trains ViT-based models on weakly labeled WSI datasets (e.g., TCGA), and  
2. Integrates these results with NLP-driven pathology report generation to provide clinically interpretable outputs.  

Thus, the pilots establish readiness, while the proposed work aims at delivering the integrated framework during the project period.

### Technical Background (Supporting Detail)
The transformer encoder is the core component of the ViT model. It processes the sequence of image patch embeddings (which include both the linear projections of the flattened patches and their positional embeddings) through multiple layers of self-attention and feed-forward neural networks.  

Multi-Head self-attention is a mechanism that allows the model to focus on different parts of the input sequence (the patch embeddings) simultaneously. It's "multi-head" because this process occurs in parallel for multiple heads, each head potentially focusing on different relationships in the data and learning to pay attention to different parts of the input. This parallel attention mechanism enables the model to capture a richer representation resulting from various types of interactions between patches, such as difference patterns or features of the input data.

While multi-head mechanism refers to the parallel processing parts in the self-attention mechanism, multi-layer perceptron(MLP) is to interpret the complex representations output by the transformer encoder and make a final prediction.

At the moment, it seems that the ResNet-50 shows consistently lower accuracy values ranging from 0.7 to 0.8 depending on the hyperparameter settings.  This lower performance could be perhaps due to the inefficient regularization technique or its sensitivity to some difference existing between the training set and the evaluation set(not generalizing well).

*Transformers* require a tokenization of data as in NLP applications and embedding of these tokens into a high dimensionial feature space so the data with semantically similar meanings tend to gather closely.  <br>
This eventually lets it to predict the next word of a given sentence (upon training with a massive amount of text data). 

In computer vision application of transformers, similarily, an image is divided ('tokenized') into small patches with the size 16x16 or 32x32 and these patches are projected('embedded') into a high dimensional feature space that represents how relevant those patches are for the task the model is trained on, such as image classification.  For example, two patches might be placed closer in the feature space because they share features critical for distinguishing between classes in the dataset.  Mind that the closeness in the projected space of patches does not necessarily mean visual similarity in the conventional sense. 

In both NLP and computer vision transformers, positional encodings are added to the embeddings to provide positional context of each token (word or patch) in the sequence. For images, this means conveying the spatial location of each patch, as the transformer architecture does not inherently capture the order of input.

In summary, word embeddings focus on semantic and syntactic relationships, while patch projections in vision transformers are more about encoding a wide range of visual features relevant to the task at hand.

*Histologic images* are complex and appear similar to the eyes of non-experts.  What distinguishes experts from laymen could be how they selectively choose where to look (focus) and extract anatomical features (compute) from the selected fields.  

CNN based models are designed to be good at computing important local features that are pooled through their multiscale convolutions to capture wider context.  Although this architecture mimics the increasing receptive fields, CNNs are inherently emphasizing local interactions between edges or blobs in understanding the image.  Thus, they are believed to be somewhat limited in preferentially selecting the fields of interest and more importantly interpreting the connections of these important fields to come up with a global understanding of the image.

Transformer based models appear to be somewhat different to CNN based ones from this perspective.  In NLP, self-attention is a mechanism that allows each word in a sentence to consider all other words in the same sentence. In computer vision tasks, this self-attention mechanism allows a patch to directly interact with every other patch in the image, regardless of their spatial distance. Thus, this ability to learn the inter-relationship between distant patches could be uniquely beneficial to get a big picture in histologic images often highly complicated in their structures (often disrupted in malignant cases) and also mixed with many different types of cells.  
(will be continued...)
### References:
1) An image is worth 16 x 16 images: Transformers for image recognition at scale. https://arxiv.org/pdf/2010.11929.pdf
2) ### Open source codes:
To prepare the dataset for training : https://huggingface.co/docs/transformers/tasks/image_classification <br>
To set up the Keras based training at SageMaker : https://www.philschmid.de/image-classification-huggingface-transformers-keras <br>
To set up the Keras bsed training process https://github.com/huggingface/notebooks/blob/main/examples/image_classification-tf.ipynb <br>


