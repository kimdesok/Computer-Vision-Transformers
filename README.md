# Computer-Vision-Transformers
Image classification tasks performed with ViT and other transformer based deep learning models

### Overview:

![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/8a97f8fa-09ac-490d-ac6a-adb5c1846d6b)

The Vision Transformer (ViT) could be understood as a BERT that applied to images. To input images to the model, each image is divided into a sequence of patches (16x16 or 32x32) and linearly embedded. A [CLS] token is added to at the beginning of the sequence to that enables the classifification of images. Then, absolute position embeddings are added to provide this sequence to the Transformer encoder [See the above model diagram][1].

To investigate the advantages of Transformer based Deep learning models for computer vision tasks, ViT model "google/vit-base-patch16-224-in21k" from Huggingface's model repository.  The Python script is still under development but it has been basically adopted from the list of open source codes[2].  The first source code loads the dataset called 'food-101' which is also available at Huggingface's dataset repository. 

As a baseline model, a ResNet-50 model, "microsoft/resnet-50",  was used, also available at Huggingface.

### Preliminary Results

Upon fine tuning the pretrained model with the food-101 dataset, it was immediately realized that the ViT model should be superior to a CNN based model such as ResNet-50.  The accuracy and loss data were as below:

![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/025afbaf-aba1-486a-bc71-2f1f33a5b2bf)

The brief demonstration was good enough to pursue the more thorough investigation of the ViT performance on histologic images such as WsI(Whole Slide Images).

Here are more preliminary result with inference examples

![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/03074ed8-d126-4ad1-8a32-650c22c1cdcb)

![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/4e6f7079-de2d-4c54-abe7-6e16201ce625)

![image](https://github.com/kimdesok/Computer-Vision-Transformers/assets/64822593/436fe46d-8f8a-4f3d-b4aa-309e7c400a4b)


### References:
1) An image is worth 16 x 16 images: Transformers for image recognition at scale. https://arxiv.org/pdf/2010.11929.pdf
2) ### Open source codes:
To prepare the dataset for training : https://huggingface.co/docs/transformers/tasks/image_classification
To set up the Keras based training at SageMaker : https://www.philschmid.de/image-classification-huggingface-transformers-keras
To set up the Keras bsed training process https://github.com/huggingface/notebooks/blob/main/examples/image_classification-tf.ipynb


