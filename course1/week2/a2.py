'''
Assignment 1.2.2
0 - preprocess image data
0.1 - horizontal flip image
0.2 - resize image dpi
0.3 - image and label to h5
Logistic Regression with a Neural Network mindset
1 - Packages
2 - Overview of the Problem set
feature x = (m, w, h, 3)
label y = (1, m)
flatten x = (w*h*3, m) ← x.reshape(m,-1).T
3 - General Architecture of the learning algorithm
- Initialize the parameters of the model
- Learn the parameters for the model by minimizing the cost
- Use the learned parameters to make predictions (on the test set)
- Analyse the results and conclude
4 - Building the parts of our algorithm
4.1 - Helper functions
4.2 - Initializing parameters
4.3 - Forward and Backward propagation
4.4 - Optimization
超参数
    - learning rate
    - iterations of gradient descent
    - w initialization
5 - Merge all functions into a model
6 - Further analysis (optional/ungraded exercise)
7 - Test with your own image (optional/ungraded exercise)
'''

import numpy as np
from sklearn import preprocessing as pre
import time
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import h5py
import os
from course1.week2 import activitive_utils as acu
from course1.week2 import lr_utils as lru
import time

def extract_image(filename,dir,key):
    with h5py.File(filename, 'r') as f:
        images = f[key][:]
    image_num = len(images)
    for i in range(image_num):
        img = images[i,:,:,:]
        if not os.path.exists(dir):
            os.makedirs(dir)
        file = dir + str(i + 1) + '.jpg'
        img = img.astype('uint8')
        cv2.imwrite(file, img)

# extract_image('datasets/train_catvnoncat.h5','images/train/','train_set_x')
# extract_image('datasets/test_catvnoncat.h5','images/test/','test_set_x')

def resize_image(file_in, width, height, file_out):
    image = Image.open(file_in)
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    resized_image.save(file_out)

def standard_dpi(path,spath,width,height):
    for child_dir in os.listdir(path):
        child_path = os.path.join(path, child_dir)
        std_child = os.path.join(spath, child_dir)
        for dir_image in os.listdir(child_path):
            if not os.path.exists(std_child):
                os.makedirs(std_child)
            resize_image(os.path.join(child_path, dir_image), width, height, os.path.join(std_child, '_' + dir_image))

# standard_dpi('images','standard',600,600)

def horizontal_flip(image):
    image = cv2.flip(image,1)
    return image

def images_horizontal_flip(path):
    for child_dir in os.listdir(path):
        child_path = os.path.join(path, child_dir)
        for dir_image in os.listdir(child_path):
            img = cv2.imread(os.path.join(child_path, dir_image))
            hrz = horizontal_flip(img)
            names = dir_image.split('.')
            filename = names[0] +'_hrz.'+ names[1]
            filename = os.path.join(child_path, filename)
            cv2.imwrite(filename, hrz)

# images_horizontal_flip('standard')

def save_image_to_h5py(path):
    img_list = []
    label_list = []
    dir_counter = 0
    for child_dir in os.listdir(path):
        child_path = os.path.join(path, child_dir)
        for dir_image in os.listdir(child_path):
            img = cv2.imread(os.path.join(child_path, dir_image))
            # img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #单通道，分辨率会下降
            print(img.shape,img.dtype,img.size)
            img_list.append(img)
            label_list.append(dir_counter)
        dir_counter += 1
    img_np = np.array(img_list)
    # for i in range(len(img_np)):
    #     print(img_np[i].shape)
    label_np = np.array(label_list).reshape(1,len(label_list))
    f = h5py.File('standard\\hdf5_file.h5', 'w')
    f['image'] = img_np
    f['labels'] = label_np
    f.close()

# save_image_to_h5py('standard')

def load_image_data(h5file):
    dataset = h5py.File(h5file, "r")
    x = np.array(dataset["image"][:])
    y = np.array(dataset["labels"][:])
    print(x.shape)
    print(y.shape)
    return x,y

# load_image_data('standard\\hdf5_file.h5')

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lru.load_dataset()
# print(classes)
# # Example of a picture
# index = 25
# plt.imshow(train_set_x_orig[index]) # 实现热图绘制
# plt.show()
# # np.squeeze 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
# print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
# ### START CODE HERE ### (≈ 3 lines of code)
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
# ### END CODE HERE ###
# print ("Number of training examples: m_train = " + str(m_train))
# print ("Number of testing examples: m_test = " + str(m_test))
# print ("Height/Width of each image: num_px = " + str(num_px))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_set_x shape: " + str(train_set_x_orig.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("test_set_x shape: " + str(test_set_x_orig.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))

# Reshape the training and test examples
### START CODE HERE ### (≈ 2 lines of code)
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T
### END CODE HERE ###
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

# standardize feature dataset 0-1 归一化
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
print ("sanity check after reshaping: " + str(train_set_x[0:5,0]))

# GRADED FUNCTION: initialize_with_zeros
# dim = m
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros(shape=(dim, 1), dtype=float)
    b = 0.0
    ### END CODE HERE ###

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b

w,b = initialize_with_zeros(train_set_x.shape[0])

# GRADED FUNCTION: propagate
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    m = X.shape[1]
    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    A = acu.sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = -1.0 / m * (np.dot(Y, np.log(A.T+1e-5)) + np.dot(1 - Y, np.log((1 - A).T+1e-5)))  # compute cost 修改精度防止溢出
    # z = np.dot(w.T,X)+b
    # A = acu.sigmoid(z)
    # cost = acu.cost(A,Y) # scalar
    ### END CODE HERE ###

    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    # dw = 1.0 / m * (np.dot(X, (A - Y).T))
    # db = 1.0 / m * np.sum(A - Y, axis=1, keepdims=True)
    dw = np.sum(np.multiply(X,A-Y),axis=1,keepdims=True)/m
    db = np.sum(A-Y,axis=1,keepdims=True)/m
    assert (dw.shape == w.shape)
    ### END CODE HERE ###
    # print('cost:',np.squeeze(cost))
    # print('dw:',dw)
    # print('db:',np.squeeze(db))
    grads = {'dw':dw,'db':db}
    return grads,cost

# grads,cost = propagate(w,b,train_set_x,train_set_y)

# GRADED FUNCTION: optimize
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False,log_step=100):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ###
        grads,cost = propagate(w, b, X, Y)
        ### END CODE HERE ###

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate * dw
        b = b - learning_rate * db
        ### END CODE HERE ###

        # Record the costs
        if i % log_step == 0:
            costs.append(np.squeeze(cost))

        # Print the cost every 100 training examples
        if print_cost and i % log_step == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

iterations = 1000
learning_rate = 0.35
tic = time.time()
params, grads, costs = optimize(w, b, train_set_x, train_set_y, iterations, learning_rate, print_cost=False,log_step=10)
toc = time.time()
print('time spend:',str(toc-tic))
print(np.array(costs))
plt.plot(np.array(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(learning_rate))
plt.show()

# GRADED FUNCTION: predict
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    m = X.shape[1]
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = acu.sigmoid(np.dot(w.T, X) + b)
    ### END CODE HERE ###


    # Convert probabilities A[0,i] to actual predictions p[0,i]
    ### START CODE HERE ### (≈ 4 lines of code)
    round_vec = np.vectorize(acu.round_prob)
    Y_prediction = round_vec(A).reshape(1,m)
    ### END CODE HERE ###

    assert (Y_prediction.shape == (1, m))

    return Y_prediction

test_y = predict(params['w'],params['b'],test_set_x)
print(test_y)

# GRADED FUNCTION: model
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """
    m_train = X_train.shape[1]
    ### START CODE HERE ###

    # initialize parameters with zeros (≈ 1 line of code)
    dim = X_train.shape[0]
    w, b = initialize_with_zeros(dim)

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    ### END CODE HERE ###

    # Print train/test Errors
    # np.mean 均值
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

tic = time.time()
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = iterations, learning_rate = learning_rate, print_cost = True)
toc = time.time()
print('time spend:',str(toc-tic))
# Example of a picture that was wrongly classified.
index = 19
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[d["Y_prediction_test"][0,index]].decode("utf-8") +  "\" picture.")
# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
print(costs)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
# START CODE HERE ## (PUT YOUR IMAGE NAME)
my_image = "cat_in_iran.jpg"   # change this to the name of your image file
# END CODE HERE ##
# We preprocess the image to fit your algorithm.
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)
# plt.imshow(image)
plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 5500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
