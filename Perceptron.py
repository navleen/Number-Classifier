import numpy as np
from random import choice
from numpy import array, dot, random
from scipy import misc
from matplotlib import pyplot as plt
import itertools
np.set_printoptions(threshold=7000)

img_1_0 = misc.imread('numbers/smaller/1-0.jpg',mode='L')
img_1_0 = img_1_0.flatten()
img_1_1 = misc.imread('numbers/smaller/1-1.jpg',mode='L')
img_1_1 = img_1_1.flatten()
img_2_0 = misc.imread('numbers/smaller/2-0.jpg',mode='L')
img_2_0 = img_2_0.flatten()
img_2_1 = misc.imread('numbers/smaller/2-1.jpg',mode='L')
img_2_1 = img_2_1.flatten()
img_3_0 = misc.imread('numbers/smaller/3-0.jpg',mode='L')
img_3_0 = img_3_0.flatten()
img_3_1 = misc.imread('numbers/smaller/3-1.jpg',mode='L')
img_3_1 = img_3_1.flatten()
img_4_0 = misc.imread('numbers/smaller/4-0.jpg',mode='L')
img_4_0 = img_4_0.flatten()
img_4_1 = misc.imread('numbers/smaller/4-1.jpg',mode='L')
img_4_1 = img_4_1.flatten()
img_5_0 = misc.imread('numbers/smaller/5-0.jpg',mode='L')
img_5_0 = img_5_0.flatten()
img_5_0[img_5_0<101]=1
img_5_0[img_5_0>100]=0
img_5_1 = misc.imread('numbers/smaller/5-1.jpg',mode='L')
img_5_1 = img_5_1.flatten()
img_5_1[img_5_1<101]=1
img_5_1[img_5_1>100]=0
X = np.array([img_1_0,img_1_1,img_2_0,img_2_1,img_3_0,img_3_1,img_4_0,img_4_1])
y = np.array([0,1,0,1,0,1,0,1])
#y = np.array([-1,1,-1,1,-1,1,-1,1])
X[X<101]=1
X[X>100]=0

img_1_0 = misc.imread('numbers/small/1-0.jpg',mode='L')
img_1_0 = img_1_0.flatten()
img_1_1 = misc.imread('numbers/small/1-1.jpg',mode='L')
img_1_1 = img_1_1.flatten()
img_2_0 = misc.imread('numbers/small/2-0.jpg',mode='L')
img_2_0 = img_2_0.flatten()
img_2_1 = misc.imread('numbers/small/2-1.jpg',mode='L')
img_2_1 = img_2_1.flatten()
img_3_0 = misc.imread('numbers/small/3-0.jpg',mode='L')
img_3_0 = img_3_0.flatten()
img_3_1 = misc.imread('numbers/small/3-1.jpg',mode='L')
img_3_1 = img_3_1.flatten()
img_4_0 = misc.imread('numbers/small/4-0.jpg',mode='L')
img_4_0 = img_4_0.flatten()
img_4_1 = misc.imread('numbers/small/4-1.jpg',mode='L')
img_4_1 = img_4_1.flatten()
img_5_01 = misc.imread('numbers/small/5-0.jpg',mode='L')
img_5_01 = img_5_01.flatten()
img_5_01[img_5_01<101]=1
img_5_01[img_5_01>100]=0
img_5_11 = misc.imread('numbers/small/5-1.jpg',mode='L')
img_5_11 = img_5_11.flatten()
img_5_11[img_5_11<101]=1
img_5_11[img_5_11>100]=0
X1 = np.array([img_1_0,img_1_1,img_2_0,img_2_1,img_3_0,img_3_1,img_4_0,img_4_1])
y1 = np.array([0,1,0,1,0,1,0,1])
#y1 = np.array([-1,1,-1,1,-1,1,-1,1])
X1[X1<101]=1
X1[X1>100]=0

img_1_0 = misc.imread('numbers/medium/1-0.jpg',mode='L')
img_1_0 = img_1_0.flatten()
img_1_1 = misc.imread('numbers/medium/1-1.jpg',mode='L')
img_1_1 = img_1_1.flatten()
img_2_0 = misc.imread('numbers/medium/2-0.jpg',mode='L')
img_2_0 = img_2_0.flatten()
img_2_1 = misc.imread('numbers/medium/2-1.jpg',mode='L')
img_2_1 = img_2_1.flatten()
img_3_0 = misc.imread('numbers/medium/3-0.jpg',mode='L')
img_3_0 = img_3_0.flatten()
img_3_1 = misc.imread('numbers/medium/3-1.jpg',mode='L')
img_3_1 = img_3_1.flatten()
img_4_0 = misc.imread('numbers/medium/4-0.jpg',mode='L')
img_4_0 = img_4_0.flatten()
img_4_1 = misc.imread('numbers/medium/4-1.jpg',mode='L')
img_4_1 = img_4_1.flatten()
img_5_02 = misc.imread('numbers/medium/5-0.jpg',mode='L')
img_5_02 = img_5_02.flatten()
img_5_02[img_5_02<101]=1
img_5_02[img_5_02>100]=0
img_5_12 = misc.imread('numbers/medium/5-1.jpg',mode='L')
img_5_12 = img_5_12.flatten()
img_5_12[img_5_12<101]=1
img_5_12[img_5_12>100]=0
X2 = np.array([img_1_0,img_1_1,img_2_0,img_2_1,img_3_0,img_3_1,img_4_0,img_4_1])
y2 = np.array([0,1,0,1,0,1,0,1])
#y2 = np.array([-1,1,-1,1,-1,1,-1,1])
X2[X2<101]=1
X2[X2>100]=0

img_1_0 = misc.imread('numbers/1-0.jpg',mode='L')
img_1_0 = img_1_0.flatten()
img_1_1 = misc.imread('numbers/1-1.jpg',mode='L')
img_1_1 = img_1_1.flatten()
img_2_0 = misc.imread('numbers/2-0.jpg',mode='L')
img_2_0 = img_2_0.flatten()
img_2_1 = misc.imread('numbers/2-1.jpg',mode='L')
img_2_1 = img_2_1.flatten()
img_3_0 = misc.imread('numbers/3-0.jpg',mode='L')
img_3_0 = img_3_0.flatten()
img_3_1 = misc.imread('numbers/3-1.jpg',mode='L')
img_3_1 = img_3_1.flatten()
img_4_0 = misc.imread('numbers/4-0.jpg',mode='L')
img_4_0 = img_4_0.flatten()
img_4_1 = misc.imread('numbers/4-1.jpg',mode='L')
img_4_1 = img_4_1.flatten()
img_5_03 = misc.imread('numbers/5-0.jpg',mode='L')
img_5_03 = img_5_03.flatten()
img_5_03[img_5_03<101]=1
img_5_03[img_5_03>100]=0
img_5_13 = misc.imread('numbers/5-1.jpg',mode='L')
img_5_13 = img_5_13.flatten()
img_5_13[img_5_13<101]=1
img_5_13[img_5_13>100]=0
X3 = np.array([img_1_0,img_1_1,img_2_0,img_2_1,img_3_0,img_3_1,img_4_0,img_4_1])
y3 = np.array([0,1,0,1,0,1,0,1])
#y3 = np.array([-1,1,-1,1,-1,1,-1,1])
X3[X3<101]=1
X3[X3>100]=0

def output(w,x):
    #weight vector has length of x+1
    #first element is the bias
    #the rest is the weights for each x feature
    return np.dot(x, w[1:]) + w[0]

def predict(w,X):
    guess = output(w,X)
    if guess>= 0.0:
        return 1
    else:
        return 0


def perceptron(X, y, learning_rate, epochs=10):
    weights = np.zeros(X.shape[1]+1)
    errors = []

    for _ in range(epochs):
        total_error = 0
        for xi, target in zip(X,y):
            #o = activation_fn(weights,xi)
            o = predict(weights,xi)
            update = learning_rate*(target - o)
            weights[1:] += update * xi
            weights[0] += update
            total_error += int(update != 0.0)
        errors.append(total_error)
    return weights, errors

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def noisy(weights):
    target = 0 #switch to 1 to see for the other case
    img_xs = misc.imread('numbers/smaller/5-0.jpg',mode='L')
    img_xs = img_xs.flatten()
    img_xs[img_xs<101]=1
    img_xs[img_xs>100]=0
    img_s = misc.imread('numbers/small/5-0.jpg',mode='L')
    img_s = img_s.flatten()
    img_s[img_s<101]=1
    img_s[img_s>100]=0
    img_m = misc.imread('numbers/medium/5-0.jpg',mode='L')
    img_m = img_m.flatten()
    img_m[img_m<101]=1
    img_m[img_m>100]=0
    img_l = misc.imread('numbers/5-0.jpg',mode='L')
    img_l = img_l.flatten()
    img_l[img_l<101]=1
    img_l[img_l>100]=0
    images = np.array([img_xs,img_s,img_m,img_l])
    

    img_sizes = np.array([len(img_xs),len(img_s),len(img_m),len(img_l)])
    img_names = np.array(['X-Small','Small','Medium','Large'])
    colors =np.array(['red','blue','green','cyan'])
    linestyles = np.array(['-','--','-.',':'])
    incorrect_percentages = []
    
    print('\nNoisy predictions for ',target,' image')
    print('-----------------')
    noise_percent = np.arange(0,100)
    plt.figure(3)
    for img, w, c, l in zip(images, weights, colors, linestyles):
        noise = random.random_integers(0,high=len(img)-1,size=len(img))
        prediction = []
        img_copy = np.copy(img)
        prev = target
        find = True
        for n_p in noise_percent:
            noise_ind_max = int(n_p*len(img)/100)
            ni = 0
            while ni < noise_ind_max:
                #p = noise[ni]
                p = ni
                img_copy[p] = 0 if img_copy[p]==1 else 1
                #img_copy[p] = 1 if img_copy[p]==1 else 0 #for the other case
                ni+=1
           

            predict_out = predict(w,img_copy)
            prediction.append(predict_out)
            if find and predict_out!=prev:
                incorrect_percentages.append(n_p)
                print('Incorrect guess for ',(str(len(img))+' pixel'),' image starts at ',n_p,'%')
                find = False
            
        plt.plot(noise_percent, prediction, color=c, linestyle=l, label=(str(len(img))+' pixel'))
        plt.legend()
        plt.xlabel('Percentage of Noise')
        plt.ylabel('Prediction given an image of 0')
        #plt.ylabel('Prediction given an image of 1') #for the other case
        plt.gca().set_ylim([0,1])

    plt.figure(4)
    plt.plot(img_sizes[:len(incorrect_percentages)],incorrect_percentages,marker='o')
    plt.xlabel('Pixels')
    plt.ylabel('% Percentage of Noise')
    plt.gca().set_ylim([0,102])
    plt.gca().set_xlim([0,7000])
        
                       
def main():
    w, errors = perceptron(X, y, learning_rate=0.5, epochs=10)
    pxs0 = predict(w,img_5_0)
    pxs1 = predict(w,img_5_1)
    w1, errors1 = perceptron(X1, y1, learning_rate=0.5, epochs=10)
    ps0 = predict(w1,img_5_01)
    ps1 = predict(w1,img_5_11)
    w2, errors2 = perceptron(X2, y2, learning_rate=0.5, epochs=10)
    pl0 = predict(w2,img_5_02)
    pl1 = predict(w2,img_5_12)
    w3, errors3 = perceptron(X3, y3, learning_rate=0.5, epochs=10)
    pxl0 = predict(w3,img_5_03)
    pxl1 = predict(w3,img_5_13)
    weights = np.array([w,w1,w2,w3])
    
    #print('Weights: %s' % w)
    #print('Weights: %s' % w1)
    #print('Weights: %s' % w2)
    print('Given Image of of 0',)
    print('-------------')
    print('Prediction for ',len(img_5_0),'pixel size image is: ',pxs0)
    print('Prediction for ',len(img_5_01),'pixel size image is: ',ps0)
    print('Prediction for ',len(img_5_02),'pixel size image is: ',pl0)
    print('Prediction for ',len(img_5_03),'pixel size image is: ',pxl0)
    print('\nGiven Image of of 1',)
    print('-------------')
    print('Prediction for ',len(img_5_1),'pixel size image is: ',pxs1)
    print('Prediction for ',len(img_5_11),'pixel size image is: ',ps1)
    print('Prediction for ',len(img_5_12),'pixel size image is: ',pl1)
    print('Prediction for ',len(img_5_13),'pixel size image is: ',pxl1)

    w_10, errors_10 = perceptron(X3, y3, learning_rate=1, epochs=10)
    w_8, errors_8 = perceptron(X3, y3, learning_rate=0.8, epochs=10)
    w_4, errors_4 = perceptron(X3, y3, learning_rate=0.4, epochs=10)
    w_1, errors_1 = perceptron(X3, y3, learning_rate=0.1, epochs=10)

    
    plt.figure(1)
    plt.plot(range(1, len(errors)+1), errors, marker='o', label='150 pixels')
    plt.plot(range(1, len(errors1)+1), errors1, color='green', marker='+', label='500 pixels', linestyle=':')
    plt.plot(range(1, len(errors2)+1), errors2, color='red', marker='*', label='1050 pixels', linestyle='--')
    plt.plot(range(1, len(errors3)+1), errors3, color='pink', marker='v', label='6800 pixels', linestyle='-.')
    plt.xlabel('Iterations')
    plt.ylabel('Missclassifications')
    plt.legend()

    plt.figure(2)
    plt.plot(range(1, len(errors_1)+1),errors_1, marker='o', label='learning rate = 0.1')
    plt.plot(range(1, len(errors_4)+1),errors_4, color='green', linestyle='--', marker='x', label='learning rate = 0.4')
    plt.plot(range(1, len(errors)+1),errors_8, color='red', linestyle=':', marker='v', label='learning rate = 0.5')
    plt.plot(range(1, len(errors_8)+1),errors3, color='pink', linestyle='-.', marker='*', label='learning rate = 0.8')
    plt.plot(range(1, len(errors_10)+1),errors_10, color='cyan', linestyle='-', marker='+', label='learning rate = 1.0')
    xlabel = 'Iterations for ' + str(len(img_5_03)) + ' pixel image'
    plt.xlabel(xlabel) 
    plt.ylabel('Missclassifications')
    plt.legend()

    noisy(weights)
    
    plt.show()
      
main()
    
