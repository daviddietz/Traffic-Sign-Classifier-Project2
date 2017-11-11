#**Traffic Sign Recognition** 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/data-visualization/image1.png "Training Data Classes"
[image2]: ./images/data-visualization/image2.png "Validation Data Classes"
[image3]: ./images/data-visualization/image3.png "Test Data Classes"
[image4]: ./images/data-visualization/beforeGray.png "Before Grayscale"
[image5]: ./images/data-visualization/afterGray.png "After Grayscale"
[image6]: ./images/data-visualization/bumpyRoad.png "Bumpy Road"
[image7]: ./images/data-visualization/childrenCrossing.png "Children Crossing"
[image8]: ./images/data-visualization/noEntry.png "No Entry"
[image9]: ./images/data-visualization/noPassing.png "No Passing"
[image10]: ./images/data-visualization/speed80.png "Speed Limit 80"
[image11]: ./images/data-visualization/speed100.png "Speed Limit 100"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/dcDun/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410 (not printed in the project but calculated)
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Each bar chart shows how many example there are per class for each dataset: Training, Validation, Test

![Training Data Classes][image1]
![Validation Data Classes][image2]
![Test Data Classes][image3]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I decided to convert the images to grayscale for simplicity (my modal did not need color to identify signs) and data reduction (minimized how much data needs to be processed. 
Here is an example of a traffic sign image before and after grayscaling.

![Before Grayscaling][image4]
![After Grayscaling][image5]

I then normalized the image data in order to reduce the variations in the input data ranges. This helps when training (using gradient decent) and decreases the changes of getting stuck in a local optimum.

For the next step I updated the shape of the images in order to add layers in the model architecture. After grayscaling the shape of an image is (32,32) and the updated shape is (32,32,1)

Finally I shuffled the training data set in order to reduce the likelihood of batches of correlated examples.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1X1 stride, valid padding, outputs 10x10x36   
| RELU					|         									|
| Max Pooling			| 2x2 stride, valid padding, outputs 5x5x36         								|
| Flatten				| input 5x5x36, output 900									|
| Fully Connected		| input 900, output 450											| 
| RELU					| 											| 	| 										
| Dropout				| training keep probability 50%											| 
| Fully Connected		| input 450, output 225										| 	| 										
| RELU				     | 										| 
| Fully Connected		 | input 90, output 43											| 

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used TensorFlow's AdamOptimizer with a batch size of 100, learning rate of 0.001, with 100 epochs and a dropout rate of 0.5. TensorFlow's softmax_cross_entropy_with_logits function was used to compute the cross entropy between the logits and labels (one hot encoded). This helps ensure the image ends up with only one label (e.g. a sign can't be a stop sign and speed limit sign). Then the minimum loss (or loss function) is calculated to determine how wrong the model is and use values that are the least wrong.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.955
* test set accuracy of 0.928

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
		I chose a LeNet architecture implemented with TensorFlow. 
* What were some problems with the initial architecture?
		The first iteration of the architecture produced training accuracy that  was lower than desired (around 82%). 
* How was the architecture adjusted and why was it adjusted? 
		The first thing I did to adjust the architecture was to add pooling (max pooling), dropout after the first activation, and added an additional fully connected layer. I also increased the depth (number of filters) settling on 36. These changes helped the network achieve a much better training rate but it still suffered from overfitting (Training was consistently more accurate than validation).  After many smaller adjustments (adding more layers, increasing/decreasing the dropout rate, adjusting the image shape), I settled on the architecture producing 99.5% training accuracy and .95.5% validation accuracy.
* Which parameters were tuned? How were they adjusted and why?
I tuned the number parameters including: epochs, batch size, learning rate and dropout rate. The epochs and batch size were set as a hyperparameters and changed depending on the environment I was running the model (I added more epochs and decreased the batch size when training in aws instance with a gpu) and looking at the output to see if the accuracy was still increasing before the training ended. The dropout rate was only applied to the training data set and was adjusted to help with overfitting. The learning rate was changed a bit between iterations (I tried 0.005 and 0.01) and finally settled on 0.001
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
I chose the convolution neural network LeNet as a way to extract image features and classify the images because of it has been proven to be effective in classifying images/characters and detecting objects that might have little or no meaning to humans. The dropout layer helped with overfitting the images, however this proved only slightly effective. Further enhancement could include penalizing large weights (using l2 loss for example).
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Bumpy Road][image6] ![Children Crossing][image7] ![No Entry][image8] 
![No Passing][image9] ![Speed Limit 80][image10] ![Speed Limit 100][image11]

The first image might be difficult to classify because it was taken at an angle and there are other high contrast areas of the image which could lead to local minimums. The next 5 images are preprocessed (cropped and resized) which should have made them easier to classify. The resolution of the images are not very high. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy Road      		| Keep right   									| 
| Children Crossing     | Speed limit (70km/h) 										|
| No entry				| No entry											|
| No Passing	      	| Speed limit (70km/h)					 				|
| Speed Limit 80		| Speed limit (70km/h)      							|
| Speed Limit 100		| Speed limit (70km/h)      							|


The model was able to correctly guess 1 of the 6 traffic signs, which gives an accuracy of 16%. This compares unfavorably to the accuracy on the test set of 92.8% with 43 classes

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 
The code for making predictions on my final model is located in the 9th cell  (In [129] of the Ipython notebook.

This section I don't think was implemented correctly. The top five softmax probabilities for each image was the same and the do not seem to make sense. Largest number for each image is 1 and the rest are very small (for example 1.95571337e-12)

----------
