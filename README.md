###### Created by: Eason Jin

# DEVS x FSAE Machine Learning Workshop 2025

[Slides]()

# Introduction

This workshop is designed to introduce the basics of machine learning. We will be looking at regression techniques and neural networks. The goal is to get you familiar with the concepts and how to implement them in Python.

Machine learning can be split into two major categories: **regression** and **classification**.

![Regression vs Classification](/assets/image-6.jpg)

**Regression** is used to predict a continuous value. It tries to fit a line (or curve) through the points and use this line to predict the value of some unseen point (this point is not in the dataset but lies somewhere on the line).

**Classification** is used to predict a discrete value. It also tries to fit a line (or curve) through the points but instead of aiming to go through all the points, it aims to use this line to separate the points into two regions. Each region represents a class. The model will predict the class of some unseen point based on which side of the line it lies on.

Before we start, make sure you have the following libraries installed:

```bash
pip install pandas numpy matplotlib torch opencv-python sklearn
```

# Regression

## Introduction

We will start off with a simple regression program. An FSAE car is able races around a track automonously. It achieves this by recognising cones on either side of the track using cameras or LiDARs and produces a serise of coordinates that the car should drive to. However due to noise and errors, this series of points may not be smooth. In this oversimplified example, we will be using regression to predict the next point based on previous ones to ensure a smooth trajectory.

![An FSAE car](/assets/fsae47.jpg)

## Step 0: Dataset

We will use F1 track layouts as our dataset, and we can use computer vision techniques to convert the images into a set of coordinates. The dataset can be found [here](https://www.shutterstock.com/image-vector/complete-set-circuits-f1-2017-season-730184434).

#### Some example images:

![tracks](/assets/tracks.png)

## Step 0.5: Image Processing

We will use OpenCV to process the images. We first convert everything to grayscale, then find the contours, and finally writing to a CSV file. This is implemented in `regression/process_image.py`. We will not go into much detail, you can check out my image processing workshop [here](https://github.com/devsuoa/computer-vision-workshop-2025/blob/main/README.md).

## Step 1: Fit a Line to the Points

A general form of the regression equation is as follows:

$$
\mathbf{y}=a_0 + a_1x_1 + a_2x_2 + ... + a_nx_n
$$

where $y$ is the output vector, $x_i$ are the inputs, and $a_i$ are the weights. The goal of regression is to find the best values for $a_i$. We can also simplify the same equation into vectors:

$$
\mathbf{y} = a_0 + 
\begin{pmatrix} a_1 & a_2 & \cdots & a_n \end{pmatrix}
\begin{pmatrix}
\begin{array}{c}
x_1 \\ x_2 \\ \vdots \\ x_n
\end{array}
\end{pmatrix}
= \mathbf{a}^T \mathbf{x} + b
$$

where $\mathbf{a}$ is a vector of weights, $\mathbf{x}$ is a vector of inputs and $b$ is the bias term ($a_0$ of the first equation).

`numpy` has a handy function `ployfit` that fits a polynomial function to the data with a given degree. It first sets up a matrix of the form:

$$
\begin{pmatrix}x_1^n&...&x_1^2&x_1^1&1\\ x_2^n&...&x_2^2&x_2^1&1\\ ...&...&...&...&...\\ x_m^n&...&x_m^2&x_m^1&1\end{pmatrix}
$$

where $m$ is the number of data points and $n$ is the degree of the polynomial. For example, if we have `x = np.array([1, 2, 3, 5])` and `degree = 2`, the matrix will look like this:

$$
\begin{pmatrix}1^2&1^1&1^0\\ 4^2&2^1&1^0\\ 9^2&3^1&1^0&\\ 25^2&5^1&1^0\end{pmatrix}=\begin{pmatrix}1&1&1\\ 4&2&1\\ 9&3&1&\\ 25&5&1\end{pmatrix}
$$

Note that the matrix will always have 1 more column than the degree to account for the bias term.

It then uses the least squares method to find the best fit line. The least squares method tries to minimize the sum of the squared differences between the predicted values and the actual values. In practice, we first find the square root of the sum of the squares, meaning that:

First column: $\sqrt{1^2+4^2+9^2+25^2}\approx26.89$\
Second column: $\sqrt{1^2+2^2+3^2+5^2}\approx6.24$\
Thrid column: $\sqrt{1^2+1^2+1^2+1^2}=2$

So these are our scales: `[26.89, 6.24, 2]`, we divide each column by their respective scale to get the normalized matrix:

$$
A=\begin{pmatrix}\frac{1}{26.89}&\frac{1}{6.24}&\frac{1}{2}\\\frac{4}{26.89}&\frac{2}{6.24}&\frac{1}{2}\\\frac{9}{26.89}&\frac{3}{6.24}&\frac{1}{2}&\\\frac{25}{26.89}&\frac{5}{6.24}&\frac{1}{2}\end{pmatrix}\approx\begin{pmatrix}0.0371&0.1602&0.5\\ 0.1488&0.3205&0.5\\ 0.3347&0.4808&0.5&\\ 0.9297&0.9013&0.5\end{pmatrix}
$$

Now we solve for:

$$
A\times \begin{pmatrix}c_0\\ c_1\\ c_2\end{pmatrix}=\mathbf{y}
$$

where $c_0, c_1, c_2$ are the coefficients of the polynomial. This step will be handled by `lstsq` from the numpy library.

## Step 2: Predict with given x

Now that we have the coefficients of the regression equation, prediction is rather easy. We can just plug the value of x into the equation and calculate the value of y. `numpy` has a function `polyval` that exactly does this.

## Step 3: Construct the Regression Class

We define `HISTORY` as the number of historic points we consider to perform regression and predict the next point.

```python
class Regression:
    def __init__(self, points, degree):
        if len(points) != HISTORY:
            raise ValueError(f"Exactly {HISTORY} points are required.")

        self.x = np.array([p[0] for p in points])
        self.y = np.array([p[1] for p in points])
        self.degree = degree

        self.coefficients = self._fit()

    def _fit(self):
        return np.polyfit(self.x, self.y, self.degree)

    def predict(self, x_value):
        y_value = np.polyval(self.coefficients, x_value)
        return y_value
```

# Neural Network

https://www.kaggle.com/datasets/samwelnjehia/simple-tire-wear-and-degradation-simulated-dataset?resource=download
