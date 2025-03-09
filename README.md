# Wine-Quality-Predication-Model
Wine Quality Predication Model Made Using RNN (Recurrent Neural Network) 
In the wine industry, assessing the quality of wine is a critical task that requires expertise and time. Automating this process with machine learning models can help winemakers and distributors maintain quality standards efficiently.

This project explores the use of a Recurrent Neural Network (RNN) for predicting the quality of red wine based on its physicochemical properties. Unlike traditional models that only rely on independent feature analysis, RNNs are capable of capturing sequential dependencies, making them suitable for datasets with potential temporal or correlated structures.

The primary objective is to predict the wine quality (on a scale of 0–10) using attributes such as alcohol content, pH levels, acidity, and other physicochemical properties. This enables wine producers to focus on improving key characteristics that influence quality, ensuring consistency and customer satisfaction.

Problem Statement
The wine quality prediction problem can be framed as follows:

Objective:
To design and implement a Recurrent Neural Network (RNN) that predicts the quality of red wine based on a given dataset containing its physicochemical attributes.

Key Features:
Input attributes include pH, alcohol content, sulfur dioxide levels, etc.
Target output is the wine quality score, an integer rating.

Goals:
Develop a preprocessing pipeline to clean and normalize the data.
Train an RNN model to learn complex patterns between physicochemical features and wine quality.
Evaluate the model's performance using metrics like Mean Absolute Error (MAE) for regression or accuracy for classification.
This problem's solution can help automate quality control in winemaking processes, improving operational efficiency and customer satisfaction.

Methodology
1. Data Preprocessing
Dataset:
The dataset contains physicochemical features of red wines and a corresponding quality rating.
Steps Taken:
Missing values were handled by removing rows with null entries.
Features were normalized to scale values between 0 and 1 for optimal neural network performance.
The target variable, "quality," was processed as:
Classification: One-hot encoded for categorical labels.
Regression: Normalized as a continuous variable.

3. Model Design
Architecture:
A Recurrent Neural Network (RNN) with LSTM layers was used.
Input Layer: Accepts normalized features with shape (samples, timesteps, features).
LSTM Layers: Extract sequential patterns and correlations between features.
Dense Layers: Map learned features to the target output.
Output:
Regression: Single node with linear activation for wine quality prediction.
Classification: Multiple nodes with softmax activation for quality classification.

5. Implementation
The model was implemented in Python using TensorFlow and Keras. The training data was split into training and test sets (80:20 ratio).

Model Details:

Input shape: (samples, 1, num_features)
Hidden Layers: Two LSTM layers with dropout for regularization.
Output Layer:
Regression: Single node, loss='mse'.
Classification: Nodes equal to classes, loss='categorical_crossentropy'.
Optimizer: Adam.
Training: 50 epochs with batch size of 32.

4. Evaluation
For regression tasks, Mean Squared Error (MSE) and Mean Absolute Error (MAE) were used to evaluate performance.
For classification, accuracy, precision, and recall metrics were used.

