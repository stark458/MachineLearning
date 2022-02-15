
#importing decision tree 
from sklearn.metrics import accuracy_score
from decision_trees_builder import DecisionTreeClassifier
import pandas as pd
import numpy as np
def random_selector(dataframe):
    nrows = len(dataframe)
    selected = dataframe.sample(n=nrows)
    return selected
def get_accuracy(data):

    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1,1)
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.1, random_state=1)
    random_tree = np.random.randint(1,len(data))
    classifier = DecisionTreeClassifier(minimum_samples=3, max_depth=random_tree)
    print("done")
    import time
    start_time = time.time()
    classifier.fit(X_train,Y_train)
    training_time = time.time() - start_time
    print("train time: = ",training_time)
    # classifier.print_tree()
    Y_pred = classifier.predict(X_test) 
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(Y_test, Y_pred)
    print("the accuracy of decision tree is:\t \t ",accuracy)
    print("successful!")

    return accuracy

if __name__ == '__main__':

    data = pd.read_csv('blobs.csv', index_col = 0)
    data_groups,accuracy = [],[]
    for i in range(0,5):
        #bootstrapping 
        data_groups.append(random_selector(data))
        

        accuracy.append(get_accuracy(data))
    print(accuracy_score)

    print("Accuracy of random forest is: \t",np.mean(accuracy))
