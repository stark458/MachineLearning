import numpy as np
import pandas as pd

# hello

#For the spirals.csv dataset we have variable X , Y => features and binary class output => label.
class RandomnessGenerator():
    def __init__(self, bootstraper, feature_subset=2):
        self.bootstrap = bootstraper
        self.feature_subset = feature_subset

    # def generate(self, dataset):
    #     """
    #     """
        
class Condition():

    def __init__(self, feature_index = None, threshold = None, left = None, right = None, information_gain = None, value = None):
        #used by decision node.
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.information_gain = information_gain
        
        #used by leaf node
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, minimum_samples = 2, max_depth = 2):
        """
        :params: minimum_samples: if the node contains l.q. Minimum samples then tree ends.
        """

        self.root = None
        self.minimum_samples = minimum_samples
        self.max_depth = max_depth




    def tree_builder(self, dataset, curr_depth=0):
        #X,Y = data[["x","y"]].to_numpy(),data[["class"]].to_numpy()
        X, Y = dataset[:,:-1], dataset[:,-1]
        bag_var = np.random.randint(len(X)-len(X)/2,len(X)+2)
        samplesize , featuresize = np.shape(X)

        if samplesize>= self.minimum_samples and curr_depth <= self.max_depth:

            #finding the best split:
            split = self.get_split(dataset,samplesize,featuresize)

            if bool(split.get('infogain')) and split['infogain']>0:
                left_subtree = self.tree_builder(split["dataset_left"],curr_depth+1)
                right_subtree = self.tree_builder(split["dataset_right"], curr_depth+1)
            
                return Condition(split["feature_index"],split["threshold"], left_subtree, right_subtree, split["infogain"])
        
        leaf_value = self.calculate(Y)

        return Condition(value=leaf_value)

    def get_split(self, dataset, samplesize,featuresize):
        """Method to get the best split


        """
        info_gain = - float("inf")
        best_split = {}
        for feature in range(featuresize):
            feature_values = dataset[:, feature]
            possible_thresholds = np.unique(feature_values)

            for threshold in possible_thresholds:
                dataset_left,dataset_right = self.split(dataset, feature,threshold)

                if len(dataset_left) !=0 and len(dataset_right) != 0:
                    
                    #finding parent class, and child classes from the dataset.

                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    informationgain = self.calcuate_info_gain(y,left_y,right_y,'gini')

                    if informationgain > info_gain:
                        best_split["feature_index"] = feature
                        best_split["infogain"] = informationgain
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
        
        return best_split
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        dataset_left = np.array([row for row in dataset if int(row[feature_index])<=threshold])
        dataset_right = np.array([row for row in dataset if int(row[feature_index])>threshold])

        # f1 = dataset.iloc[:,feature_index].values.tolist()
        # dataset_left = np.asarray([val for val in f1 if val<=threshold]) 
        # dataset_right = np.asarray([val for val in f1 if val>threshold]) 
        return dataset_left, dataset_right

    def calcuate_info_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    def entropy(self, y):
        ''' function to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
    def calculate(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.tree_builder(dataset)
    def predict(self, X):
        ''' function to predict new dataset '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    
    def print_tree(self, tree=None, indent=" "):

        ''' function to print the tree '''
    
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

if __name__ == '__main__':

    data = pd.read_csv('spirals.csv', index_col = 0)


    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1,1)
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.1, random_state=1)

    classifier = DecisionTreeClassifier(minimum_samples=10, max_depth=300)
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
