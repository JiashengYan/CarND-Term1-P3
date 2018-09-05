# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report





## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My overall strategy for deriving a model architecture was to experiment on various architectures with different level of complexity and find out a capble model that not is not over complex.

My first step was to use convolution neural network models similar to the Lenet and more complex NVIDIA model. I thought these two model might be appropriate because both of them are powerful networks. The lenet was successfully used to classify signs in previous project and the NVIDIA is also intended for much difficult tasks.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (splitrate 0.2). The best performance of each models were measued according to the best validation loss before overfitting occur. Overfitting can be identified when the the train loss keeps reducing while the validation loss begins to fluctuate around a value or even increasing again.

After training, both networks showed samilar performance of validation loss (around 0.025), therefroe I choose to continue with the lenet model as it's simpler. 

Then I tried to reduce parameters further by reduce the depth of convolutional layers and finally got the architecture shown above, which keeps same accuracy.

To combat the overfitting, I add maxpooling layers between convolutional layers and dropout layer between fully connected layers.

The final step was to run the simulator to see how well the car was driving around track one. There one spot right after the bridge where the vehicle fell off the track as the lane boundary at that part is not as clear as other parts. To improve the driving behavior in these cases, I collected more data of the vehicle driving through that part of roads.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. An appropriate model architecture has been employed

My model consists of three convolution layers and three fully connected layers (code line 100 - 121):

* Convolutional Layer 1 : 5x5 Filter with depth 3, activation function:elu, Maximalpooling(4x4)
* Convolutional Layer 2 : 3x3 Filter with depth 6, activation function:elu, Maximalpooling(4x4)
* Convolutional Layer 2 : 3x3 Filter with depth 12, activation function:elu, Maximalpooling(2x2)
* Fully Connected Layer 1 : n = 64, activation function : elu
* Dropout Layer : Dropout Value = 0.5
* Fully Connected Layer 1 : n = 32, activation function : elu
* Dropout Layer : Dropout Value = 0.5
* Fully Connected Layer 1 : n = 16, activation function : elu
* Dropout Layer : Dropout Value = 0.5
* Output Layer : n = 1

The model includes ELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer. 


#### 3. Attempts to reduce overfitting in the model

The model contains two dropout layers between fully connected layers in order to reduce overfitting (model.py lines 116, 118). 

The dataset was splited into training data and validation data, so that the model was trained and validated on different data sets to ensure that the model was not overfitting (code line 26). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 4. Creation of the Training Set & Training Process

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (both in clockwise and anti-clockwise direction), recovering from the left and right sides of the road.

To capture good driving behavior, I first recorded two laps on track one in clockwise dirction and two laps in anti-clockwise direction using center lane driving. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return back to the center lane for the cases that it drive away from center by accidents. 

To augment the data sat, I also flipped images and angles thinking that this would help to generalize differnet cases and avoid the network to remember the map. 

The images from left-mounted and righ-mounted cameras are also utilized from trainning to improve the network's ability to return to center from sides. A correction angle was added to the actual steering angle to pair with the images from left sied and right side:
* images form left side: steering_angle + correction
* images form right side: steering_angle - correction

After the collection process, I had 38250 number of images. I then preprocessed this data by conerting images to grayscale and crop the upper and lower part off as both parts contains little relavant information.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 as evidenced by the following image of learning curve. 

The model used an adam optimizer, so the learning rate was not tuned manually (default learning rate 0.001 shows best performance).

![sample_dataset](https://github.com/JiashengYan/CarND-Term1-P3/blob/master/Train_Loss_Curve.png)