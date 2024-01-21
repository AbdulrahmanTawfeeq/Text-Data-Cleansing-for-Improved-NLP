from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb


# pip install xgboost


class Models:
    @staticmethod
    def train_nb(train_features, train_labels, test_features, test_labels, alpha=1):
        """
        Trains a Naive Bayes model using the specified features and labels, and prints out the accuracy of the model on
        the test data.

        Parameters:
        train_features (sparse matrix): The features of the training data, as a sparse matrix
        train_labels (array-like): The labels of the training data
        test_features (sparse matrix): The features of the test data, as a sparse matrix
        test_labels (array-like): The labels of the test data
        """

        # Create a Naive Bayes model
        nb_model = MultinomialNB(alpha=alpha, fit_prior=False)

        # Train the Naive Bayes model using the training data
        nb_model.fit(train_features, train_labels)

        # Predict the labels for the test data using the trained Naive Bayes model (testing)
        test_predictions = nb_model.predict(test_features)

        Models.print_evaluation_metrics(test_labels, test_predictions)

    @staticmethod
    def train_svm(train_features, train_labels, test_features, test_labels, kernel='linear', c=1.0):
        """
        Trains an SVM model using the specified features and labels, and prints out the accuracy of the model on the
        test data.

        Parameters:
        train_features (sparse matrix): The features of the training data, as a sparse matrix
        train_labels (array-like): The labels of the training data
        test_features (sparse matrix): The features of the test data, as a sparse matrix
        test_labels (array-like): The labels of the test data
        kernel (str): The type of kernel to use for the SVM. Default is 'linear'.
        C (float): The regularization parameter for the SVM. Default is 1.0.
        """

        # Create an SVM model with the specified kernel and regularization parameter
        svm_model = SVC(kernel=kernel, C=c, gamma='auto')

        # Train the SVM model using the training data
        svm_model.fit(train_features, train_labels)

        # Predict the labels for the test data using the trained SVM model
        test_predictions = svm_model.predict(test_features)

        Models.print_evaluation_metrics(test_labels, test_predictions)

    @staticmethod
    def train_dt(train_features, train_labels, test_features, test_labels, max_depth=None, max_features=None,
                 criterion="gini"):
        """
        Trains a Decision Tree model using the specified features and labels, and prints out the accuracy of the model
        on the test data.

        Parameters:
        train_features (sparse matrix): The features of the training data, as a sparse matrix
        train_labels (array-like): The labels of the training data
        test_features (sparse matrix): The features of the test data, as a sparse matrix
        test_labels (array-like): The labels of the test data
        """

        # Create a Decision Tree model
        dt_model = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features, criterion=criterion)

        # Train the Decision Tree model using the training data
        dt_model.fit(train_features, train_labels)

        # Predict the labels for the test data using the trained Decision Tree model
        test_predictions = dt_model.predict(test_features)

        Models.print_evaluation_metrics(test_labels, test_predictions)

    @staticmethod
    def train_lr(train_features, train_labels, test_features, test_labels):
        # Create a logistic regression model
        lr_model = LogisticRegression()

        # Train the logistic regression model using the training data
        lr_model.fit(train_features, train_labels)

        # Predict the labels for the test data using the trained logistic regression model (testing)
        test_predictions = lr_model.predict(test_features)

        Models.print_evaluation_metrics(test_labels, test_predictions)

    @staticmethod
    def train_xgb(train_features, train_labels, test_features, test_labels):
        # Create a dictionary to map the label values
        label_map = {
            -1: 0,
            0: 1,
            1: 2
        }

        # Use list comprehension to map the labels
        train_labels = [label_map[label] for label in train_labels]
        test_labels = [label_map[label] for label in test_labels]

        # n_estimators: The number of trees in the forest. A larger number of trees may improve the performance of
        # the model, but also increase the training time.

        # max_depth: The maximum depth of each tree. A deeper tree may capture more complex relationships in the data,
        # but also increase the risk of overfitting.

        # Overfitting happens when a model is too complex and fits the training data's noise or random fluctuations,
        # not the underlying patterns that generalize well to new data. This results in good performance on the training
        # data but poor performance on new data. The model becomes too specialized to the training data and loses its
        # ability to generalize to new data.

        # learning_rate: The step size shrinkage used to prevent overfitting. A smaller learning rate may require more
        # trees to fit the data, but may result in better generalization.

        # subsample: The fraction of samples used for each tree. A smaller subsample may reduce the risk of overfitting,
        # but may also increase the variance of the model.
        #
        # colsample_bytree: The fraction of features used for each tree. A smaller fraction may reduce the risk of
        # overfitting, but may also decrease the predictive power of the model.
        # Create an XGBoost classifier
        xgb_model = xgb.XGBClassifier(n_estimators=100,
                                      max_depth=5,
                                      learning_rate=0.1,
                                      subsample=0.8,
                                      colsample_bytree=0.8)

        # Train the XGBoost classifier using the training data
        xgb_model.fit(train_features, train_labels)

        # Predict the labels for the test data using the trained XGBoost classifier (testing)
        test_predictions = xgb_model.predict(test_features)

        Models.print_evaluation_metrics(test_labels, test_predictions)

    @staticmethod
    def print_evaluation_metrics(test_labels, test_predictions):
        # Calculate the evaluation metrics
        accuracy = accuracy_score(test_labels, test_predictions)
        precision = precision_score(test_labels, test_predictions, average='weighted')
        recall = recall_score(test_labels, test_predictions, average='weighted')
        f1 = f1_score(test_labels, test_predictions, average='weighted')
        micro_f1 = f1_score(test_labels, test_predictions, average='micro')
        macro_f1 = f1_score(test_labels, test_predictions, average='macro')

        # Print the evaluation metrics
        print("Accuracy on test data: {:.2f}%".format(accuracy * 100))
        print("Precision on test data: {:.2f}%".format(precision * 100))
        print("Recall on test data: {:.2f}%".format(recall * 100))
        print("F1 score on test data (weighted): {:.2f}%".format(f1 * 100))
        print("Micro F1 score on test data: {:.2f}%".format(micro_f1 * 100))
        print("Macro F1 score on test data: {:.2f}%".format(macro_f1 * 100))
        print("")
