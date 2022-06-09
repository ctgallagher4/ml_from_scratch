import numpy as np

class knnclassifier:

    '''Charlie's implementation of the k-nearest neighbor algorithm using only numpy'''

    def __init__(self, X, y, n_neighbors=5,):
        self.n_neighbors = n_neighbors
        self.X_train = X
        self.y_train = y

    def compute_distance(self, x1, x2):
    '''A method for returning the euclidian distance between two points.'''
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict(self, X):
    '''A method for making predictions.'''
        predictions = []
        for X_item in X:
            distances = []
            for train_item in self.X_train:
                distances.append(self.compute_distance(X_item, train_item))
            
            indices = np.argsort(distances)[:self.n_neighbors]
            nearest_neighbors = self.y_train[indices]

            #return the majority vote
            count_dict = {}
            for i in nearest_neighbors:
                count_dict[i] = list(nearest_neighbors).count(i)
            values = list(count_dict.values())
            keys = list(count_dict.keys())
            majority_vote = keys[values.index(max(values))]
            
            predictions.append(majority_vote)

        return predictions


                
