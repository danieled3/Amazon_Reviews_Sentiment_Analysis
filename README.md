# Amazon Reviews: Sentiment Analysis

## Table of Content
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspects](#technical-aspects)
  * [Results](#result)
  * [Technologies](#technologies)
  * [To Do](#to-do)
  * [File List](#file-list)
  * [Credits](#credits)


  
## Overview <a name="overview" />
In this project, I loaded and processed titles of Amazon reviews to predict the review scores. I compared the performances of 3 models (Tensorflow): 
* classification model with an LTSM layer
* regression model with an LTSM layer 
* regression model with a 1D convolutional layer

In the end, the classifications made by these 3 models were compared with the classification made by a human being through AWS Sagemaker Ground Truth to understand whether the models were further improvable.

## Motivation <a name="motivation" />
The book [Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) written by the author of Keras and most scientific articles recommend using classification models to perform sentiment analysis or score prediction. However, by doing so, we consider each score as a distinct class and lose the information of the sorting. To preserve this information I wanted to train a regression model and a classification model, with similar complexity, on the same data to see the difference in accuracy and MAE.

Moreover, I was curious to understand the potential of Tensorflow models with LTSM layers. In other words, I wanted to understand whether a simple model with few layers was able to reach the same performance of a human being in the identification of score ratings.

## Technical Aspect <a name="technical-aspects" />
In this project, I used the classical Keras functions for text preprocessing and tokenization i.e. *Tokenizer* and *pad_sequence*. Models were built in Tensorflow and performances visualization was made by *sklearn.confusion_matrix*.

The most original aspects are :
* The custom Keras metric *ordinal_accuracy* made to monitor accuracy during the training phase of regression models
* The output layer of regression models built with Lamba layer. It allows to bound output between -0.5 and 4.5.
* The *make_confusion_matrix* function used to visualize and understand confusion matrixes

## Result <a name="result" />
The confusion matrixes obtained from the predictions of the 4 analyzed models are the following:

<img src="https://user-images.githubusercontent.com/29163695/122113334-3fd3ff00-ce22-11eb-80e2-741cc13019e5.png" height="400">
<img src="https://user-images.githubusercontent.com/29163695/122112158-d0114480-ce20-11eb-85b8-47b4912d23ca.png" height="400">

<img src="https://user-images.githubusercontent.com/29163695/122112221-e0292400-ce20-11eb-8703-a550ec62404e.png" height="400">
<img src="https://user-images.githubusercontent.com/29163695/122112292-f0d99a00-ce20-11eb-9e06-88a16a05e469.png" height="400">

I noticed that:
1. The classification model provides a higher accuracy even if some predictions are heavily wrong (i.e. a lot of "5" in place of "1" or vice versa). The MAE is the highest because classes are independent and sorting information is not used.
2. Even if the regression models have the same complexity of the classification model, they do not provide so high accuracy. However, thanks to the optimization function used in the training phase, prediction errors are often lower.
3. Both classification and regression models, even if they are very simple, allow reaching the same precision of a human being. It is a proof of the potential of LSTM layers and convolutional layers in neural networks.

## Technologies <a name="technologies" />
I used *nltk* library for text preprocessing, *Tensorflow* for model building and *AWS Ground Truth* to make data classified by human beings.

<img src="https://user-images.githubusercontent.com/29163695/122077900-726b0100-cdfc-11eb-90d4-9e45d3a3f53f.png" height="200">
<img src="https://user-images.githubusercontent.com/29163695/122078058-94fd1a00-cdfc-11eb-93d4-fe4159a0675a.png" height="200">
<img src="https://user-images.githubusercontent.com/29163695/122078294-c675e580-cdfc-11eb-95d6-bdd137cf2847.png" height="200">


## To Do <a name="to-do" />
* Since the accuracy of the model is similar to the accuracy of a human classifier, we may try to create a model only for reviews of a particular kind of object to obtain better performance.
* We may analyze full reviews and see whether performances improve or not.
* We may train the model on the full dataset (the number of reviews to load has been limited to speed up model training)

## File List <a name="file-list" />
* **main.py** Data loading, data preprocessing, model training and model evaluation.
* **my_utils.py** Useful functions to load data and plot confusion matrixes

## Credits <a name="credits" />
* [DTrimarchi10](https://github.com/DTrimarchi10) - Thanks for the confusion_matrix function I took inspiration from
* [Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) - Thansk of the authors of the book for the advice about model building
* [Xiang Zhang](https://figshare.com/articles/dataset/Amazon_Reviews_Full/13232537/1) - Thanks for the complete dataset of Amazon reviews
