###### Created by: Eason Jin

# DEVS x FSAE Machine Learning Workshop 2025

[Recording]()

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

We will use F1 track layouts as our dataset, and we can use computer vision techniques to convert the images into a set of coordinates. The dataset can be found [here](https://www.shutterstock.com/image-vector/complete-set-circuits-f1-2017-season-730184434). The computer vision workshop recordings can be found [here](https://github.com/devsuoa/computer-vision-workshop-2025).

#### Some example images:

![tracks](/assets/tracks.png)

## Step 0.5: Image Processing

We will use OpenCV to process the images. We first convert everything to grayscale, then find the contours, and finally writing to a CSV file. This is implemented in `regression/process_image.py`. We will not go into much detail.

## Step 1: Fit a Line to the Points

A general form of the regression equation is as follows:

$$
\mathbf{y}=a_0 + a_1x_1 + a_2x_2 + ... + a_nx_n
$$

where $y$ is the output vector, $x_i$ are the inputs, and $a_i$ are the weights. The goal of regression is to find the best values for $a_i$. We can also simplify the same equation into vectors:

$$
\mathbf{y}=a_0+\begin{pmatrix}a_1&a_2&...&a_n\end{pmatrix} \begin{pmatrix}x_1 \\\ x_2 \\\ ...\\\ x_n \end{pmatrix}=\mathbf{a}^T\mathbf{x}+b
$$

where $\mathbf{a}$ is a vector of weights, $\mathbf{x}$ is a vector of inputs and $b$ is the bias term ($a_0$ of the first equation).

`numpy` has a handy function `ployfit` that fits a polynomial function to the data with a given degree. It first sets up a matrix of the form:

$$
\begin{pmatrix}x_1^n&...&x_1^2&x_1^1&1 \\\ x_2^n&...&x_2^2&x_2^1&1 \\\ ...&...&...&...&...\\\ x_m^n&...&x_m^2&x_m^1&1\end{pmatrix}
$$

where $m$ is the number of data points and $n$ is the degree of the polynomial. For example, if we have `x = np.array([1, 2, 3, 5])` and `degree = 2`, the matrix will look like this:

$$
\begin{pmatrix}1^2&1^1&1^0 \\\ 2^2&2^1&2^0 \\\ 3^2&3^1&3^0& \\\ 5^2&5^1&5^0\end{pmatrix}=\begin{pmatrix}1&1&1 \\\ 4&2&1 \\\ 9&3&1& \\\ 25&5&1\end{pmatrix}
$$

Note that the matrix will always have 1 more column than the degree to account for the bias term.

It then uses the least squares method to find the best fit line. The least squares method tries to minimize the sum of the squared differences between the predicted values and the actual values. In practice, we first find the square root of the sum of the squares, meaning that:

First column: $\sqrt{1^2+4^2+9^2+25^2}\approx26.89$\
Second column: $\sqrt{1^2+2^2+3^2+5^2}\approx6.24$\
Thrid column: $\sqrt{1^2+1^2+1^2+1^2}=2$

So these are our scales: `[26.89, 6.24, 2]`, we divide each column by their respective scale to get the normalized matrix:

$$
A=\begin{pmatrix}\frac{1}{26.89}&\frac{1}{6.24}&\frac{1}{2} \\\ \frac{4}{26.89}&\frac{2}{6.24}&\frac{1}{2} \\\ \frac{9}{26.89}&\frac{3}{6.24}&\frac{1}{2}& \\\ \frac{25}{26.89}&\frac{5}{6.24}&\frac{1}{2}\end{pmatrix}\approx\begin{pmatrix}0.0371&0.1602&0.5 \\\ 0.1488&0.3205&0.5 \\\ 0.3347&0.4808&0.5&\\\ 0.9297&0.9013&0.5\end{pmatrix}
$$

Now we solve for:

$$
A\times \begin{pmatrix}c_0 \\\ c_1 \\\ c_2\end{pmatrix}=\mathbf{y}
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

## Step 4: Passing the Points to the Class

Refer to code at `regression/main.py`

## Step 5: Evaluate the Model

Since this is a regression task, we can calculate the mean square error (MSE) between the predicted value and the actual value. The MSE is defined as:

$$
MSE=\frac{1}{n}\sum_{i=1}^n(y_i-\hat{y}_i)^2
$$

We can also check the R-squared score, which is a statistical measure of how close the data are to the fitted regression line. An R-squared of 1 means that the line perfectly fits the data (this may not be good because the model is potentially overfitting). An R-squared of 0 or negative means the line does not fit the data at all.

# Neural Network

## Introduction

Neural network is a more advanced form of regression. Instead of calculating a curve to fit, it learns the non-linear relationship between the input and output. You have probably all seen this diagram before:
![Neural Network](/assets/Colored_neural_network.svg.jpg)

Each input node represents a feature(column) of the input data. Each arrow between nodes has its own weight value. The node where the arrow ends takes the sum of the value of each node multiplied by the weight of the respective arrow. The weights are updated in every training cycle. Naturally, the output node will also be a number, and the meaning of that number is up to the developer to interpret.

## Step 0: Dataset

We will use a simple dataset that records the tire degradation while the race car is running. We aim to predict the degradation of the tire given some other car and environment parameters. This original dataset is deleted from the repository because it is very large, you can download it [here](https://www.kaggle.com/datasets/samwelnjehia/simple-tire-wear-and-degradation-simulated-dataset?resource=download).

## Step 0.5: Data Processing

Inspecting the dataset, we can see that it has the following columns:

```bash
['Lap', 'Motorsport_Type', 'Track', 'lap_time', 'Throttle', 'Brake', 'Steering_Position', 'Speed', 'Surface_Roughness', 'Ambient_Temperature', 'Humidity', 'Wind_Speed', 'Lateral_G_Force', 'Longitudinal_G_Force', 'Tire_Compound', 'Tire_Friction_Coefficient', 'Tire_Tread_Depth', 'Tire_wear', 'cumilative_Tire_Wear', 'Driving_Style', 'force_on_tire', 'Event', 'front_surface_temp', 'rear_surface_temp', 'front_inner_temp', 'rear_inner_temp', 'Tire degreadation']
```

We also observe that `Motorsport_Type`, `Track`, `Tire_Compound`, `Driving_Style` and `Event` are categorical variables (i.e. they are strings and not numbers). Since neural networks can only process numbers, we can create a map that converts each categorical variable into a number. The code to do this can be found under `neural_network/process_data.py`. What each variable is mapped to is recorded in `neural_network/keys.txt`. It looks something like this:

```bash
{'DTM': 0, 'WEC': 1, 'F1': 2, 'Sports Car': 3, 'SUV': 4, 'Sedan': 5, 'Hot Hatch': 6, 'Sub-Urban': 7, 'Crossover': 8}
{'Nürburgring Nordschleife': 0, 'Monza': 1, 'Red Bull Ring': 2, 'Monaco': 3}
{'C1': 0, 'C2': 1, 'C3': 2, 'C4': 3, 'C5': 4}
{'Normal': 0, 'Aggressive': 1}
{'Lock-up': 0, 'Normal': 1, 'Loss of Traction': 2, 'puncture': 3}
```

These numbers will replace the string values in the dataset.

We will also drop `Team` and `Driver` columns by assuming all drivers exhibit the same behaviour for simplicity.

The dataset contains a lot of columns, but not all of them are necessarily useful for our prediction. All features influence the target in one way or another, but only a few of them are the deciding factors. A causal illustration can be seen below (this is not exactly the same but gets the idea across):
![Causal Graph](/assets/draw_graph1.jpg)

Each node is a variable and arrow represents the causation (or how much one influences another). We can see that some numbers are significantly larger than the others. The `pandas` `Dataframe` has a method called `corr` which finds the Pearson correlation between each of the columns.

```python
correlations = df.corr()[TARGET].drop(TARGET).abs()
```

For example we have two columns $X$ and $Y$, and we want to find the Pearson correlation between them. The formula for Pearson correlation ($r$) is as follows:

$$
r=\frac{Cov(X, Y)}{\sigma_X\sigma_Y}, r[-1, 1]
$$

where $Cov(X, Y)$ means the covariance between $X$ and $Y$, and $\sigma$ means the standard deviation.

The covariance is defined as:

$$
Cov(X, Y)=\frac{1}{n-1}\sum_{i=1}^n(X_i-\bar{X})(Y_i-\bar{Y})
$$

where $\bar{X}$ and $\bar{Y}$ are the means of $X$ and $Y$ respectively. In simple words, the covariance is the average of the product of the difference between each value and the mean.

The standard deviation of $X$ is defined as:

$$
\sigma_X=\sqrt{\frac{1}{n-1}\sum_{i=1}^n(X_i-\bar{X})^2}
$$

it basically represents the spread of the data.

In Pearson correlation, $r=1$ means that the two variables are perfectly correlated, $r=-1$ means that the two variables are perfectly inversely correlated, and $r=0$ means the two variables are not correlated at all. The closer the value is to 1 or -1, the stronger the correlation.

We also drop the column of `TARGET` because the correlation of a variable on itself is always 1.

Then we sort the correlation in descending order:

```python
most_influential = correlations.sort_values(ascending=False)
```

and we get the following:

```bash
Speed                        0.967303
force_on_tire                0.922719
Tire_wear                    0.913166
cumilative_Tire_Wear         0.762723
Wind_Speed                   0.549569
Track                        0.371106
lap_time                     0.257100
Lap                          0.256574
Throttle                     0.241156
Tire_Friction_Coefficient    0.240725
Tire_Compound                0.240725
Humidity                     0.235376
front_inner_temp             0.225665
front_surface_temp           0.221546
rear_surface_temp            0.216876
rear_inner_temp              0.215619
Brake                        0.210878
Motorsport_Type              0.127249
Surface_Roughness            0.111238
Ambient_Temperature          0.092125
Tire_Tread_Depth             0.002019
Driving_Style                0.000390
Event                        0.000196
Longitudinal_G_Force         0.000191
Lateral_G_Force              0.000187
Steering_Position            0.000003
```

We see that variables such as `Speed` and `force_on_tire` are very influential, while `Lateral_G_Force` and `Steering_Position` are not. Thus, we decide a reasonable threshold of 0.1 and drop all variables below that.

```bash
Dropped columns: ['Ambient_Temperature', 'Tire_Tread_Depth', 'Driving_Style', 'Event', 'Longitudinal_G_Force', 'Lateral_G_Force', 'Steering_Position']
```

Now we can save the processed dataset as `neural_network/data.csv` and use that to train our model.

## Step 1: Construct the Neural Network Class

We can use `torch` to construct the model. We can define the layers as follows:

```python
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, layer_1_size=128, layer_2_size=64):
        super(NeuralNetwork, self).__init__()
        # Input layer
        self.layer1 = nn.Linear(input_dim, layer_1_size)
        self.activation1 = nn.LeakyReLU()
        # Hidden layer
        self.layer2 = nn.Linear(layer_1_size, layer_2_size)
        self.activation2 = nn.LeakyReLU()
        # Output layer
        self.output = nn.Linear(layer_2_size, 1)
```

We set up a basic neural network with 1 input layer, 1 hidden layer, and 1 output layer. The input layer has number of nodes equal to the number of features in the dataset. The output layer has 1 node because this is a regression task we only need to predict 1 value. If a model is constructed for classification, then we need to have as many nodes as the number of classes. We arbitarily set the input layer to produce 128 nodes and the hidden layer to produce 64 nodes. In practice, these numbers can be changed until you are satisfied with the performance of the model.

> How do we determine the number of nodes of the hidden layer?
>
> Literally just **guess**.

At each node, it produces the sum of all inputs multiplied by their respective weights:

$$
y=\sum_iw_ix_i+b
$$

We then pass this sum through an activation function. Activation functions are usually non-linear, thus it helps the model to learn the non-linear relationships between inputs and outputs. We use `LeakyReLU` for our mode. The plot for `LeakyReLU` looks like:

![LeakyReLU](/assets/images.jpeg)

This means that if the sum is negative, the function will output a small negative number to avoid the output of a neuron being 0, meaning that it will never be activated and thus never learn anything. A positive number will be passed through unchanged.

![Activation](/assets//neuron-activation-monkey.png)

A `forward` method is defined within class, which will be called automatically when we use the model. It describes how the input flows through the model.

```python
def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.output(x)
        return x
```

To update the weights, we need to calculate the loss of the prediction. We use `nn.MSELoss()` to calculate the mean square error between the predicted value and the actual value, which is defined as:

$$
MSE=\frac{1}{n}\sum_{i=1}^n(y_i-\hat{y}_i)^2
$$

where $y_i$ is the actual value recorded in the dataset and $\hat{y}_i$ is the predicted value from the model.

After we know the loss, we can use `torch.optim.Adam` to update the weights. Adam is a popular optimization algorithm that uses the gradient descent method to update the weights. It is an adaptive learning rate method, meaning that it adjusts the learning rate based on the first and second moments of the gradients. The learning rate is a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated. An illustration of the gradient descent method and learning rate can be seen below (taken from COMPSYS306 course book):

![Gradient Descent](/assets//Screenshot%202025-04-20%20130827.png)

In the code, this is done by calling:
```python
optimiser.step()
```
What happens behind the scenes is that when we calculate the loss (or cost) $C$ at the output node, this loss is propagated back to the input node using the chain rule.

![Chain rule](/assets/Screenshot%202025-04-20%20213637.png)

In this diagram, $w$ is our weight, $z$ is the value of the node, $y$ is the value after the activation function, and $C$ is the cost. We aim to minimise the cost, meaning that the partial derivative of $C$ w.r.t. $w$ should be minimal. We know that $C$ is a function of $y$, $y$ is a function of $z$, and $z$ is a function of $w$. Thus, we can apply the chain rule to find $\frac{\partial C}{\partial w}$:

$$
\frac{\partial C}{\partial w} = \frac{\partial C}{\partial y}\times \frac{\partial y}{\partial w} = \frac{\partial C}{\partial y}\times \frac{\partial y}{\partial z}\times \frac{\partial z}{\partial w}
$$

If the gradient of $\frac{\partial C}{\partial w}$ is positive, we need to decrease the weight. If the gradient is negative, we need to increase the weight. The magnitude of the gradient decides how much the weight is changed. The new weight will be calculated as:

$$
w'=w-\eta\frac{\partial C}{\partial w}
$$

where $\eta$ is the learning rate. The learning rate can be changed. A large learning rate will train the model faster but may overshoot the minimum. A small learning rate will more likely to reach global minimum at the cost of time and the risk of stuck in a local mimimum.

Now the model is ready for the next training iteration.

## Step 2: Prepare the Training and Testing Data

To ensure the reliability of our evaluation, we will split the dataset into `training` and `testing` sets. The model will learn the `training` set and we will evaluate the performance of the model against the `testing` set. We use the `train_test_split` method from `sklearn` and set a ratio of 80:20:

```python
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
)
```

To properly use the `torch` library, we need to convert the data into `tensor`s. A `tensor` is a multi-dimensional array that can be used to perform mathematical operations. It is similar to a numpy array, but it can be used on a GPU to speed up the calculations.

```python
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
```

We need to unsqueeze the `y` tensor to add an extra dimension, because the output layer of the model expects a 2D tensor of shape (batch_size, 1). I don't know why but this is how the library is defined.

We use the `DataLoader` class to create batches of input data. A batch is a small subset of data. We do this because by default, the model only updates its weights after a full pass of the dataset, which will lead to very slow learning. By using batches, the model updates its weight every batch, making the training a lot more efficient. We arbitarily set the batch size to be 32:

```python
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

## Step 3: Create the Model and Train

```python
model = NeuralNetwork(input_dim=X.shape[1])
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# The model will pass through the entire training dataset 10 times
for epoch in range(10):
    # Iterate through each batch
    for batch_X, batch_y in train_loader:
        # Make prediction (calls forward)
        preds = model(batch_X)
        # Calculate loss
        loss = loss_fn(preds, batch_y)
        # Clear previous gradients
        optimizer.zero_grad()
        # Backpropagation
        loss.backward()
        # Update weights
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## Step 4: Evaluate the Model

Similarly, we can also use MSE and R-squared score to evaluate the model.

```python
model.eval()
with torch.no_grad():
    # Run the model on the testing dataset
    test_preds = model(X_test_tensor)
    test_preds = test_preds.numpy()
    y_test_np = y_test_tensor.numpy()

mse = mean_squared_error(y_test_np, test_preds)
r2 = r2_score(y_test_np, test_preds)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")
```
