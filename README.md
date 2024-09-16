# neural-network-challenge-1

neural network study

https://bootcampspot.instructure.com/courses/6442/assignments/80401

student_loans_with_deep_learning.ipynb

https://colab.research.google.com/drive/1YaFO2sQDpRw_s_Tar3L8IDRQTr5PB-J9#scrollTo=E-hZaeSn6q61

student_loans_with_deep_learning_performance.ipynb

https://colab.research.google.com/drive/1tdKl2_3Qdd7w7-3ZlJcW9U48dkxYFB2Z#scrollTo=ogm3_9wo4HAD&uniqifier=1

### Background

You work at a company that specializes in student loan refinancing. If the company can predict whether a borrower will repay their loan, it can provide a more accurate interest rate for the borrower. Your team has asked you to create a model to predict student loan repayment.

The business team has given you a CSV file that contains information about previous student loan recipients. With your knowledge of machine learning and neural networks, you decide to use the features in the provided dataset to create a model that will predict the likelihood that an applicant will repay their student loans. The CSV file contains information about these students, such as their credit ranking.

* Prepare the data for use on a neural network model.
* Compile and evaluate a model using a neural network.
* Predict loan repayment success by using your neural network model.
* Discuss creating a recommendation system for student loans

#### Part 1: Prepare the data for use on a neural network model

Using your knowledge of Pandas and scikit-learn’s `StandardScaler()`, preprocess the dataset so that you can use it to compile and evaluate the neural network model later.

Open the starter code file and complete the following data preparation steps:

1. Read the data from [https://static.bc-edx.com/ai/ail-v-1-0/m18/lms/datasets/student-loans.csv**Links to an external site.**](https://static.bc-edx.com/ai/ail-v-1-0/m18/lms/datasets/student-loans.csv) into a Pandas DataFrame. Review the DataFrame, looking for columns that could eventually define your features and target variables.
2. Create the features (`X`) and target (`y`) datasets. The target dataset should be defined by the “credit_ranking” column. The remaining columns should define the features dataset.
3. Split the features and target sets into training and testing datasets.
4. Use scikit-learn's `StandardScaler` to scale the features data.

#### Part 2: Compile and Evaluate a Model Using a Neural Network

Use your knowledge of TensorFlow to design a deep neural network model. This model should use the dataset’s features to predict the credit quality of a student based on the features in the dataset. Consider the number of inputs before determining the number of layers that your model will contain or the number of neurons on each layer. Then, compile and fit your model. Finally, evaluate the model to calculate its loss and accuracy.

To do so, complete the following steps:

1. Create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using TensorFlow’s Keras.
   **hint  You can start with a two-layer deep neural network model that uses the `relu` activation function for both layers.**
2. Compile and fit the model using the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` evaluation metric.
   **hint  When fitting the model, start with a small number of epochs, such as 50 or 100.**
3. Evaluate the model using the test data to determine the model’s loss and accuracy.
4. Save and export your model to a keras file, and name the file `student_loans.keras`.
   **note:  **Remember to download your saved model from Colab so you can upload it to your GitHub repo.

#### Part 3: Predict loan repayment success by using your neural network model

Use the model you saved in the previous section to make predictions on your reserved testing data.

To do so, complete the following steps:

1. Reload your saved model.
2. Make predictions on the testing data, saving them to a DataFrame and rounding the predictions to binary values.
3. Generate a classification report with the predictions and testing data.

#### Part 4: Discuss creating a recommendation system for student loans

Briefly answer the following questions in the space provided:

1. Describe the data that you would need to collect to build a recommendation system to recommend student loan options for students. Explain why this data would be relevant and appropriate.
2. Based on the data you chose to use in this recommendation system, would your model be using collaborative filtering, content-based filtering, or context-based filtering? Justify why the data you selected would be suitable for your choice of filtering method.
3. Describe two real-world challenges that you would take into consideration while building a recommendation system for student loans. Explain why these challenges would be of concern for a student loan recommendation system.



----------------------------------------------------------------------------------------------------------------------------------------------

Scoring:

#### Prepare the Data for Use on a Neural Network Model (15 points)

* [ ] Two datasets were created: a target (`y`) dataset, which includes the "credit_ranking" column, and a features (`X`) dataset, which includes the other columns. (5 points)
* [ ] The features and target sets have been split into training and testing datasets. (5 points)
* [ ] Scikit-learn's `StandardScaler` was used to scale the features data. (5 points)

#### Compile and Evaluate a Model Using a Neural Network (30 points)

* [ ] A deep neural network was created with appropriate parameters. (10 points)
* [ ] The model was compiled and fit using the `accuracy` loss function, the `adam` optimizer, the `accuracy` evaluation metric, and a small number of epochs, such as 50 or 100. (10 points)
* [ ] The model was evaluated using the test data to determine its loss and accuracy. (5 points)
* [ ] The model was saved and exported to a keras file named `student_loans.keras`. (5 points)

#### Predict Loan Repayment Success by Using your Neural Network Model (25 points)

* [ ] The saved model was reloaded. (5 points)
* [ ] The reloaded model was used to make binary predictions on the testing data. (10 points)
* [ ] A classification report is generated for the predictions and the testing data. (10 points)

#### Discuss creating a recommendation system for student loans (30 points)

**For Question 1:**

* [ ] The response describes the data that should be collected to build a recommendation system for student loan options. (4 points)
* [ ] The response explains why they think that data should be collected. (4 points)
* [ ] The type of data described is appropriate for a recommendation system for student loan options. (2 points)

**For Question 2:**

* [ ] The response chose a filtering method. (4 points)
* [ ] The student justified the choice of their filtering method. (4 points)
* [ ] The choice of filtering method was appropriate for the data selected in the previous question. (2 points)

**For Question 3:**

* [ ] The response lists two real-world challenges with building a recommendation system for student loans. (4 points)
* [ ] The response explains why these challenges would be of concern for a student loan recommendation system. (6 points)
