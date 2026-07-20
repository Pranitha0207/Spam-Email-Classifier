# Spam-Email-Classifier
This project is a machine learning-based Spam Email Classifier developed using Python. It analyzes email messages and classifies them as either Spam or Not Spam (Ham).

The project uses the Spam Email dataset and processes the email text using CountVectorizer, which converts text messages into numerical feature vectors. The data is divided into training and testing sets using train_test_split.

A Multinomial Naive Bayes algorithm is trained on the email data to identify patterns commonly found in spam messages. After training, the model predicts the class of test emails and evaluates its performance using accuracy score and classification report.

Finally, a custom email message is tested by the trained model, and the system displays whether the email is Spam Email or Not Spam.

Technologies Used: Python, Pandas, NumPy, Scikit-learn, Machine Learning, Natural Language Processing (NLP).
