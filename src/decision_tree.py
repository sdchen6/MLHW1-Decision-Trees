from math import log2
import numpy as np

class Node():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Node classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

def branch_features(attribute_col, features, targets):
    branch1 = np.empty((0,features.shape[1]))
    branch2 = np.empty((0,features.shape[1]))
    target1 = np.empty(0)
    target2 = np.empty(0)

    for i in range(0, len(attribute_col)):
        if attribute_col[i]:
            branch1 = np.vstack((branch1,features[i]))
            target1 = np.append(target1,targets[i])
        else:
            branch2 = np.vstack((branch2,features[i]))
            target2 = np.append(target2,targets[i])
    return branch1, branch2, target1, target2

def ID3(features, targets, attribute_names):
    node = Node()
    if np.all(targets == 1):
        node.attribute_name = "leaf"
        node.value = 1
        return node
    elif np.all(targets == 0):
        node.attribute_name = "leaf"
        node.value = 0
        return node
    elif np.size(features) == 0:
        labels, counts = np.unique(targets, return_counts=True)
        maxindex = counts.argmax()
        node.value = labels[maxindex]
        node.attribute_name = "leaf"
        return node
    else:
        Att = best_attribute(features, targets)
        node.attribute_index = Att
        node.attribute_name = attribute_names[Att]
        new_features = np.delete(features, Att, 1)
        new_attribute_names = np.delete(attribute_names, Att)
        branch1, branch2, target1, target2 = branch_features(features[:,Att],new_features,targets)
        node1 = Node()
        node2 = Node()
        labels, counts = np.unique(targets, return_counts=True)
        maxindex = counts.argmax()
        most_common = labels[maxindex]

        if np.size(target1) == 0:
            node1.attribute_name = "leaf"
            node1.value = most_common
        else:
            node1 = ID3(branch1, target1, new_attribute_names)
        
        if np.size(target2) == 0:
            node2.attribute_name = "leaf"
            node2.value = most_common
        else:
            node2 = ID3(branch2, target2, new_attribute_names)
        node.branches = [node1, node2]

    return node


class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Node classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)
        
        """
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )
        

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        Output:
            VOID: It should update self.tree with a built decision tree.
        """
        self._check_input(features)
        
        self.tree = ID3(features, targets, self.attribute_names)
        self.visualize()
    
    def predict_onerow(self, node, example):
        if len(node.branches) == 0:
            return node.value   
        else:
            attribute_index = self.attribute_names.index(node.attribute_name)
            if example[attribute_index] == 1:
                return self.predict_onerow(node.branches[0],example)
            else:
                return self.predict_onerow(node.branches[1],example)

    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        Outputs:
            predictions (np.array): numpy array of size N array which has the predicitons 
            for the input data.
        """
        self._check_input(features)
        # for each row of features,
        predictions = np.empty((0))
        for i in features:
            predictions = np.append(predictions,self.predict_onerow(self.tree, i))
        return predictions

    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

def best_attribute(features, targets):
    best_ig = -1
    best_att = -1
    for attribute_index in range(0,features.shape[1]):
        cur_ig = information_gain(features, attribute_index, targets)
        if cur_ig > best_ig:
            best_ig = cur_ig
            best_att = attribute_index
    return best_att

def binary_split(attributes,targets):
    pos_set = []
    neg_set = []
    for i in range(0, len(attributes)):
        if attributes[i]:
            pos_set.append(targets[i])
        else:
            neg_set.append(targets[i])
    return pos_set, neg_set

def entropy(fraction):
    if fraction == 0:
        return 0
    else:
        return -1*fraction*log2(fraction)


def information_gain(features, attribute_index, targets):

    pos_parent = np.sum(targets)/np.size(targets)
    neg_parent = 1- pos_parent
    parent_entropy =  entropy(pos_parent) + entropy(neg_parent)

    attribute = features[:,attribute_index]
    child_entropy = 0
    pos_set, neg_set = binary_split(attribute, targets)
    pos_set_pos = np.sum(pos_set)/np.size(pos_set)
    pos_set_neg = 1-pos_set_pos
    neg_set_pos = np.sum(neg_set)/np.size(neg_set)
    neg_set_neg = 1- neg_set_pos
    pos_set_weight = np.size(pos_set)/np.size(targets)
    neg_set_weight = np.size(neg_set)/np.size(targets)

    child_entropy_pos = entropy(pos_set_pos)*pos_set_weight + entropy(pos_set_neg)*pos_set_weight
    child_entropy_neg = entropy(neg_set_pos)*neg_set_weight + entropy(neg_set_neg)*neg_set_weight
    child_entropy = child_entropy_neg+ child_entropy_pos

    info_gain = parent_entropy-child_entropy

    return info_gain



if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Node(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Node(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
