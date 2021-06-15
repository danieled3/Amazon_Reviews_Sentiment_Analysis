# Amazon Reviews: Sentiment Analysis

## Table of Content
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspects)
  * [Result](#result)
  * [Technologies](#technologies)
  * [To Do](#to-do)
  * [File List](#file-list)
  * [Credits](#credits)


  
## Overview <a name="overview" />
In this project I loaded and processed titles of Amazon reviews to predict the review scores. I compared the performances of 3 models (Tensorflow): 
* classification model with a LTSM layer
* regression model with a LTSM layer 
* regression model with a 1D convolutional layer
In the end the classifications made by these 3 models were compared with the classification made by a human being through AWS Sagemaker Ground Truth to understand whether the models were further improvable.

## Motivation <a name="motivation" />
The book [Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) written by the author of Keras and most scientific articles recommend using classification models to perform sentiment analysis or score prediction. However, by doing so, we consider each score as a distinct class and lose the information that the class 4 for example is most similar to class 5 than to class 1. In order to preserve this information I wanted to train a regression model and a classification model, with similar complexity, on the same data to see the difference in accuracy and MAE.
Moreover, I was curious to understand the potential of LTSM layers. In other words I wanted to understand wheter a simple model with few layers was able to reach the same performance of a human being in identification of score ratings.

## Technical Aspect <a name="technical-aspects" />
The main issue of this project were:
1. Organizing raw data in a "tidy version"
2. Finding interpretable charts to visualize results

## Result <a name="result" />
The main result, that you can also find in UL_Population_Study.ppt, are the following:
1. Uk population is constantly growing (growth rate of 500k people/year in the last 5 years).
2. The total population growth is mainly due to the growth of England population.
![image](https://user-images.githubusercontent.com/29163695/121808797-7e1ec200-cc5a-11eb-94b7-36f82c999790.png)

3. UK population is slowly leaving Scotland (it may have a lower bith rate or higher emigration rate)
4. England population, on the contrary, is increasing 
![image](https://user-images.githubusercontent.com/29163695/121808980-48c6a400-cc5b-11eb-84d2-c3e61877d58a.png)

5. Population is ageing because people older than 50 are strongly growing in recent years ( because of either high emigration of young people or low birth rate)
![image](https://user-images.githubusercontent.com/29163695/121811058-66980700-cc63-11eb-8865-7fdad80f82db.png)

6. Wales has the largest percentage of over 50 year old people

![image](https://user-images.githubusercontent.com/29163695/121809055-98a56b00-cc5b-11eb-8de1-b40b7269b79d.png)

## Technologies <a name="technologies" />
I used nltk libery for text preprocessing, tensorflow for model building and AWS Ground Truth to make data classified by human beings.
<img src="https://user-images.githubusercontent.com/29163695/122077900-726b0100-cdfc-11eb-90d4-9e45d3a3f53f.png" height="200">
<img src="https://user-images.githubusercontent.com/29163695/122078058-94fd1a00-cdfc-11eb-93d4-fe4159a0675a.png" height="200">
<img src="https://user-images.githubusercontent.com/29163695/122078294-c675e580-cdfc-11eb-95d6-bdd137cf2847.png" height="200">


## To Do <a name="to-do" />
* Since the accuracy of model is similar to the accuracy of human classifier, we may try to create a model only for reviews of a particular kind of objects to obtain better peformances.
* We may analyze full review and see whether performances improve or  not.
* We may train model on full dataset (the number of reviews to load has been limited to speed up model training)

## File List <a name="file-list" />
* **UK_Population_Study.ipynb** Completed and commented data exploration analisys made thorught R Markdown in RStudio
* **UK_Population_Study.pdf** Extraction in pdf of the previous analysis and of code output
* **UK_Population_Presentation.ppt** Presentations of main insights

## Credits <a name="credits" />
* [DTrimarchi10](https://github.com/DTrimarchi10) - Thanks for the confusion_matrix function I took inspiration from
* [Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) - Thansk of the authors of the book for the advice about model building
* [Xiang Zhang](https://figshare.com/articles/dataset/Amazon_Reviews_Full/13232537/1) - Thanks for the complete dataset of Amazon reviews

