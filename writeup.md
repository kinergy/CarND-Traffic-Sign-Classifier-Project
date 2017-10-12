# **Traffic Sign Recognition**

## Kimon Roufas


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

<!-- [image1]: ./examples/visualization.jpg "Visualization" -->
[image1]: ./writeup/training-set-histogram.png "Histogram"
[image2]: ./writeup/example-image.png "Grayscaling"
[image4]: ./five-traffic-signs/no-passing-32.png "Traffic Sign 1"
[image5]: ./five-traffic-signs/priority-road-32.png "Traffic Sign 2"
[image6]: ./five-traffic-signs/rough-road-32.png "Traffic Sign 3"
[image7]: ./five-traffic-signs/speed-limit-80-32.png "Traffic Sign 4"
[image8]: ./five-traffic-signs/stop-32.png "Traffic Sign 5"
[image9]: ./writeup/top-5-guesses.png "Top 5 Guesses"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [https://github.com/kinergy/CarND-Traffic-Sign-Classifier-Project](https://github.com/kinergy/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34,799**
* The size of the validation set is **4,410**
* The size of test set is **12,630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

One can quickly see that the training set is biased due to the varying number of training examples for each class. I expect this will have consequences in the quality of my model being as accurate as possible for the under-represented classes.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I initially explored converting the images to HSV and then experimented with using each of the channels. My theory was that I may be able to get better contrast of the features. Ultimately, this did not work as well and I switched to doing a simple grayscale processing step.

Here is an example of a traffic sign image before and after grayscaling:

![alt text][image2]

As the next step, I calculated the mean and standard deviation on the training set only and then saved those values. I used them to normalize the validation and test sets, and later, also the 5 new images. I did this to get the training data values into a range that helps the training converge better.

Training set mean = 81
Training set standard deviation = 66

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer             |     Description                   |
|:---------------------:|:---------------------------------------------:|
| Input             | 32x32x1 Grayscale image                 |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x6  |
| RELU          |                       |
| Dropout         | keep probability = 0.5                        |
| Max pooling         | 2x2 stride, outputs 14x14x6         |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU          |                       |
| Dropout         | keep probability = 0.5                        |
| Max pooling         | 2x2 stride, outputs 5x5x16        |
| Flatten   | Input 5x5x16, output 400                          |
| Fully connected   | Input 400, output 120                         |
| RELU          |                       |
| Dropout         | keep probability = 0.5                        |
| Fully connected   | Input 120, output 84                          |
| RELU          |                       |
| Dropout         | keep probability = 0.5                        |
| Fully connected   | Input 84, output 43 logits                        |
| Softmax       | normalize to max of 1 by squishing the logits                   |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer with a learning rate of 0.001 to minimize the average cross entropy loss. I experimented with the number of epochs, and ended up calculating out to 100 which provided enough opportunity for training to reach the point where validation accuracy stabilized. I could have kept the best model along the way in case by random chance the final result wasn't as good as some intermediate, however, it wasn't clear to me that doing this would eventually produce a better or worse test result.

I didn't keep track of a list of the experiments I did, however, I did try decreasing the learning rate in situations where it seemed that the validation accuracy was jumping around too much. After adding the dropout layers though, the whole model seemed to stabilize much better and I stuck with a learning rate of 0.001. A recommended value for keep probability in the dropout layers is 0.5, I started there and it worked well enough so I didn't modify further.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The architecture I chose to use is LeNet, it is well-known and seems to perform well for classification of images that are 32x32 pixels. I experimented with a version that used all three RGB (red, green, blue) layers from the training set but found that not to work as well as when I preprocessed the images into grayscale. I was able to achieve validation accuracy results almost always just under 0.93. Not good enough, there was too much chance in exceeding the required minimum of 0.93. The model was not robust enough. The natural next step was to start experimenting with regularization, so I added dropout. With some further experimentation of the hyperparameters as discussed above, I was able to converge on my final results.

The training set accuracy is higher than that of the validation set, but not by a whole lot. There may be some overfitting. However the validation accuracy was in an acceptable range. I only ran the test set through the model once and was satisfied with the results so I ended my experimentation there.

My final model results were:
* training set accuracy of **0.994**
* validation set accuracy of **0.961**
* test set accuracy of **0.944**

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

These images have varying backgrounds and different angles, however, there is reasonably good contrast and the angles are not too severe. The signs were cropped and then scalled to be similarly proportioned as to the data set images.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image             |     Prediction                    |
|:---------------------:|:---------------------------------------------:|
| No passing         | No passing                     |
| Priority road         | Priority road                    |
| Rough road         | Rough road |
| Speed limit 80            | Speed limit 80        |
| Stop     | Stop                   |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.4%. If I had selected more images to test on, I'm sure the results may not have been so perfect. Also, it seems that this model works on signs that are nicely proportined in the image, but I am skeptical about other sizes.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For all images the model is quite sure about the classification. I didn't expect this so I spent some time reviewing everything to make sure I wasn't inadvertently passing in the correct answers somehow... In each case the next best guess for classifying the image is more than an order of magnitude less than the correct prediction. Here is a visual of the top 5 guesses for each of the signs:

![alt text][image9]

The actual top guess had a high probability in every test image:

| Probability           |     Prediction                    |
|:---------------------:|:---------------------------------------------:|
| 1.00               | No passing                     |
| 1.00            | Priority road                    |
| 1.00         | Rough road                     |
| 1.00             | Speed limit 80                  |
| 1.00           | Stop                   |

