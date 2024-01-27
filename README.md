# Computer-Vision-Transformers
Image classification tasks performed with ViT and other transformer based deep learning models

### Overview:

![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/8a97f8fa-09ac-490d-ac6a-adb5c1846d6b)
*High-level overview of the Vision Transformer (ViT) architecture* MLP: Multi-Layer Perceptron. Taken from the original paper [1] <br>

The Vision Transformer (ViT) could be understood as a BERT that applied to images. To input images to the model, each image is divided into a sequence of patches (16x16 or 32x32) and linearly embedded. A [CLS] token is added to at the beginning of the sequence to that enables the classifification of images. Then, absolute position embeddings are added to provide this sequence to the Transformer encoder [See the above model diagram][1].

The transformer encoder is the core component of the ViT model. It processes the sequence of image patch embeddings (which include both the linear projections of the flattened patches and their positional embeddings) through multiple layers of self-attention and feed-forward neural networks.  

Multi-Head self-attention is a mechanism that allows the model to focus on different parts of the input sequence (the patch embeddings) simultaneously. It's "multi-head" because this process occurs in parallel for multiple heads, each head potentially focusing on different relationships in the data and learning to pay attention to different parts of the input. This parallel attention mechanism enables the model to capture a richer representation resulting from various types of interactions between patches, such as difference patterns or features of the input data.

While multi-head mechanism refers to the parallel processing parts in the self-attention mechanism, multi-layer perceptron(MLP) is to interpret the complex representations output by the transformer encoder and make a final prediction(.

### Methods and Materials

To investigate the advantages of Transformer based deep learning models for computer vision tasks, a ViT model "google/vit-base-patch16-224-in21k" selected from Huggingface's model repository.  The Python script has been basically adopted from the list of open source codes[2] and is still under development.  The first source code loads a dataset called 'food-101' available at Huggingface's dataset repository. 

As a baseline model, a ResNet-50 model, "microsoft/resnet-50",  was used, also available at Huggingface's model repository.  

### Preliminary Results

Upon fine tuning the pretrained models with the food-101 dataset, it is becoming obvious that the ViT model should be superior to a CNN based model such as ResNet-50 in a number of benchmark criteria.  The ViT model seemed better in accuracy and loss and converged faster than the Resnet-50: 86.2% vs. 76.7% in accuracy and 15.6 hrs vs. 30.9 hrs taken for the training.  (See the table below, still in preparation):
									
![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/ab5fdbac-0f49-4b10-a6b4-ee42a51b57b6)

The above results were obtained with the same preprocessing such as data augmentation, converting the data to a tensorflow format, etc.  
At the moment, each model should be further tuned with adjusting hyperparameters and trying for different optimizers, especially for the ResNet-50 model since it seemed overfitted (The accuracy of the SOTA ResNet-50 reached 
about 90% as top 1 accuracy).

![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/c38bd6bc-28fc-4e20-997e-0dde8019932f)

![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/8d5d8c66-d369-4ade-96dd-c183d5c73cbc)


### Brief Discussion (which will lead to an experimental hypothesis)

The simple demonstration was good enough to warrant more experimental works on the ViT model with anticipating its superior performance on histologic images (without requiring more powerful compute resources).

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

Here are some preliminary results of inference: <br>
![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/81f7622e-0fc0-413a-8751-064750942445)

![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/d2044dde-4632-40dd-94be-2159bdefe1c5)

### References:
1) An image is worth 16 x 16 images: Transformers for image recognition at scale. https://arxiv.org/pdf/2010.11929.pdf
2) ### Open source codes:
To prepare the dataset for training : https://huggingface.co/docs/transformers/tasks/image_classification <br>
To set up the Keras based training at SageMaker : https://www.philschmid.de/image-classification-huggingface-transformers-keras <br>
To set up the Keras bsed training process https://github.com/huggingface/notebooks/blob/main/examples/image_classification-tf.ipynb <br>


