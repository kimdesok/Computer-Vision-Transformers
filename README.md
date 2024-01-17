# Computer-Vision-Transformers
Image classification tasks performed with ViT and other transformer based deep learning models

### Overview:

![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/8a97f8fa-09ac-490d-ac6a-adb5c1846d6b)

The Vision Transformer (ViT) could be understood as a BERT that applied to images. To input images to the model, each image is divided into a sequence of patches (16x16 or 32x32) and linearly embedded. A [CLS] token is added to at the beginning of the sequence to that enables the classifification of images. Then, absolute position embeddings are added to provide this sequence to the Transformer encoder [See the above model diagram][1].

To investigate the advantages of Transformer based deep learning models for computer vision tasks, a ViT model "google/vit-base-patch16-224-in21k" selected from Huggingface's model repository.  The Python script has been basically adopted from the list of open source codes[2] and is still under development.  The first source code loads a dataset called 'food-101' available at Huggingface's dataset repository. 

As a baseline model, a ResNet-50 model, "microsoft/resnet-50",  was used, also available at Huggingface's model repository.  

### Preliminary Results

Upon fine tuning the pretrained model with the food-101 dataset, it was immediately realized that the ViT model should be superior to a CNN based model such as ResNet-50.  The accuracy and loss data were as below:

![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/025afbaf-aba1-486a-bc71-2f1f33a5b2bf)

The above results were obtained with the same preprocessing such as data augmentation, converting the data to a tensorflow format, etc.  
At the moment, each model should be further tuned with adjusting hyperparameters and trying for different optimizers, especially for the ResNet-40 model since the accuracy of the SOTA reached 
about 90%(top 1 accuracy).

![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/03074ed8-d126-4ad1-8a32-650c22c1cdcb)

The brief demonstration was good enough to pursue further with the ViT model to get its performance on histologic images.

Transformers require a tokenization of data as in NLP applications and embedding of these tokens into a high dimensionial feature space so the data with semantically similar meanings tend to gather closely.  This eventually lets it to predict the next word of a given sentence (upon training with a massive amount of text data). 

In computer vision application of transformers, similarily, an image is divided ('tokenized') into a small patches with the size 16x16 or 32x32 and projected('embedded') into a a high dimensional feature space that represents how relevant those patches are for the task the model is trained on, such as image classification so that the closeness in the projected space of patches does not necessarily mean visual similarity in the conventional sense. For example, two patches might be close in the feature space because they share features critical for distinguishing between classes in the dataset.

In both NLP and computer vision transformers, positional encodings are added to embeddings to provide positional context of each token (word or patch) in the sequence. For images, this means conveying the spatial location of each patch, as the transformer architecture does not inherently capture the order of input.

In summary, word embeddings focus on semantic and syntactic relationships, while patch projections in vision transformers are more about encoding a wide range of visual features relevant to the task at hand.

Histologic images are complex and appear similar to the eyes of non-experts.  What distinguishes experts from laymen could be how they selectively choose where to look (focus) and extract anatomical features (compute) from the selected fields.  

CNN based models are designed to be good at computing important local features through their multiscale convolutions so that edges or blobs are playing an important role in understanding the image.  However, they are somewhat limited in preferentially selecting the fields of interest and more importantly interpreting the connections of these important fields to come up with global understanding of the image.

Transformer based models are somewhat different in this sense.  In NLP, self-attention is a mechanism that allows each word in a sentence to consider (or "attend to") all other words in the same sentence.  As in NLP, this mechanism is applied to represent the inter-relation between distant patches in the image and this attribute might be a key to understand how the ViT model works (better than CNN-based models).

(will be continued...)

Here are some preliminary results of inference:

![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/4e6f7079-de2d-4c54-abe7-6e16201ce625)

![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/436fe46d-8f8a-4f3d-b4aa-309e7c400a4b)


### References:
1) An image is worth 16 x 16 images: Transformers for image recognition at scale. https://arxiv.org/pdf/2010.11929.pdf
2) ### Open source codes:
To prepare the dataset for training : https://huggingface.co/docs/transformers/tasks/image_classification
To set up the Keras based training at SageMaker : https://www.philschmid.de/image-classification-huggingface-transformers-keras
To set up the Keras bsed training process https://github.com/huggingface/notebooks/blob/main/examples/image_classification-tf.ipynb


