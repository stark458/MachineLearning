
#importing decision tree 
from sklearn.metrics import accuracy_score, classification_report
from decision_trees_builder import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def random_selector(dataframe):
    nrows = len(dataframe)
    selected = dataframe.sample(n=nrows)
    return selected
def get_accuracy(data,classification_type = None,iterval=10,f=None):

    if classification_type is None:

        X = data.iloc[:, :-1].values
        Y = data.iloc[:, -1].values.reshape(-1,1)
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.9, random_state=1)
        itervalue = iterval*7
        minim = np.random.randint(10,20)
        with open("test_result_blobs.txt", "a") as f: 
            f.write(f"\nWhen Tree depth = {itervalue}, minimum value for sample to further split = {minim} and train_data_size = {len(X_train)} and test data size = {len(X_test)}")
            
        classifier = DecisionTreeClassifier(minimum_samples=minim, max_depth=itervalue)
        print("done")
        import time
        start_time = time.time()
        classifier.fit(X_train,Y_train)
        training_time = time.time() - start_time
        print("train time: = ",training_time)
        with open("test_result_blobs.txt", "a") as f: 
            f.write(f"\t training time= {np.round(training_time,4)} seconds")   
        # classifier.print_tree()
        Y_pred = classifier.predict(X_test) 
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(Y_test, Y_pred)
        with open("test_result_blobs.txt", "a") as f: 
            f.write(f"and the accuracy for the decision tree {iterval} = {np.round(accuracy,4)} %.")
        print("the accuracy of decision tree is:\t \t ",accuracy)
        print("successful!")
    
    else:
        from text_data import get_text
        #for text data classification
        data_text,label  = get_text()
        # X = data.iloc[:, :-1].values
        # Y = data.iloc[:, -1].values.reshape(-1,1)
        X = data_text
        Y = label
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5, random_state=1)
        itervalue = iterval*3
        with open("test_result_blobs.txt", "a") as f: 
            f.write(f"\nWhen Tree depth = {iterval}, and train_data_size = {len(X_train)} and test data size = {len(X_test)}")
        classifier = DecisionTreeClassifier(minimum_samples=10, max_depth=itervalue,file=f)
        print("done")
        import time
        start_time = time.time()
        classifier.fit(X_train,Y_train)
        training_time = time.time() - start_time
        print("train time: = ",training_time)
        with open("  test_result_blobs.txt", "a") as f: 
            f.write(f"\n Training time= {np.round(training_time,4)} seconds")        
        # classifier.print_tree()
        Y_pred = classifier.predict(X_test) 
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(Y_test, Y_pred)
        with open("test_result_blobs.txt", "a") as f: 
            f.write(f"\n The accuracy for the decision tree {iterval} = {np.round(accuracy,4)} seconds.")
        print("the accuracy of decision tree is:\t \t ",accuracy)
        print("successful!")

    return accuracy

if __name__ == '__main__':

    data = pd.read_csv('spirals.csv', index_col = 0)
    with open('test_result_blobs.txt', 'w') as f:
        f.write("Random forest classification result in spam/ham classification dataset.")

    data_groups,accuracy = [],[]
    trees = []
    classification_type = None
    for i in range(1,5):
        with open('test_result_blobs.txt','a') as fi:
            fi.write(f"Iteration {i},")

        #bootstrapping 
        data_groups.append(random_selector(data))
        accuracy.append(get_accuracy(data,classification_type,i,f))
        trees.append(i*3)
    print(accuracy_score)

    print("Accuracy of random forest classification is: \t",np.mean(accuracy))
    with open('test_result_blobs.txt','a') as x:
        x.write(f"The overall accuracy in ensemble of four trees is {np.mean(accuracy)}")
    plt.title("various trees ensemble for random forest and respective accuracy")
    plt.xlabel("Trees with varying number of depth and random features")
    plt.ylabel("accuracy")
    plt.plot(trees,accuracy)
    plt.show()
    print("done")