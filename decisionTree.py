# DECISION TREE CLASSIFICATION MODEL
import csv
import math
import time

# Node class to store the attribute name, value to split on, gini impurity of each decision tree node as well as references to the left and right subtree
class Node:
    def __init__(self, attribute, split, value):
        self.attribute = attribute
        self.split = split
        self.value = value
        self.left = None
        self.right = None

# Decision tree class to build and traverse the decision tree
class DecisionTree:
    def __init__(self):
        self.root = None

    # Call a recursive insert function to build the decision tree using the training data
    # First find the top node of the decision tree
    def insert(self, dataset, classification):
        attrName, splitVal, maxGini = findGini(dataset, classification)
        self.root = self._insert_recursive(self.root, dataset, classification, attrName, splitVal, maxGini, 0)

    def _insert_recursive(self, root, dataset, classification, attribute, split, value, depth):
        if (len(dataset) != 0) and (depth < 8):   # Check for maximum depth
            # If only one classification value remains, assign this node as the terminal node
            if (len(set(classification[1:]))) <= 1 or (len(classification) <= 1) or (len(dataset[0]) <= 2):
                root = Node(classification[-1], None, None)
                return root

            # If the node for this subtree does not exist, assign the node with the provided values
            if not root:
                root = Node(attribute, split, value)
            
            # Split the dataset into further subsets for the left subtree and right subtree
            new_dataset_l, new_dataset_r, new_classification_l, new_classification_r = splitDataset(dataset, classification, attribute, split)
           
            # If both subsets are empty, assign the children node with the classification values that occurs most for the left and right subtrees
            if (len(new_dataset_l) == 0) and (len(new_dataset_r) == 0):
                outcome_l = max(set(new_classification_l), key = classification.count)
                outcome_r = max(set(new_classification_r), key = classification.count)
                if outcome_l == outcome_r:
                    root = Node(outcome_l, None, None)
                else:
                    root.left = Node(outcome_l, None, None)
                    root.right = Node(outcome_r, None, None)
                return root

            # Call the recursive insert function for the left subtree
            if (len(new_dataset_l) > 0) and (len(new_dataset_l[0]) > 1):
                attrName_l, splitVal_l, maxGini_l = findGini(new_dataset_l, new_classification_l)
                root.left = self._insert_recursive(root.left, new_dataset_l, new_classification_l, attrName_l, splitVal_l, maxGini_l, depth+1)
            # If the left subtree dataset is empty, assign the classification value that occurs most as the terminal node for the left subtree
            elif (len(new_classification_l) < 1) and (len(new_dataset_l[0]) <= 1):
                outcome = max(set(classification), key = classification.count)
                root.left = Node(outcome, None, None)
                return root

            # Call the recursive insert function for the left subtree
            if (len(new_dataset_r) > 0) and (len(new_dataset_r[0]) > 1):
                attrName_r, splitVal_r, maxGini_r = findGini(new_dataset_r, new_classification_r)
                root.right = self._insert_recursive(root.right, new_dataset_r, new_classification_r, attrName_r, splitVal_r, maxGini_r, depth+1)
            # If the right subtree dataset is empty, assign the classification value that occurs most as the terminal node for the right subtree
            elif (len(new_classification_r) < 1) and (len(new_dataset_r[0]) <= 1):
                outcome = max(set(classification), key = classification.count)
                root.right = Node(outcome, None, None)
                return root

            # If both child nodes have the same classification, remove them and assign the current node as the terminal node
            if (root.left) and (root.right):
                if (root.left.attribute == root.right.attribute) and (root.left.attribute in ["positive", "negative"]):
                    root = root.left
            # If branches extend unnecessarily, remove the unnecessary branches
            elif (root.left) and (not root.right):
                root = root.left
            elif (root.right) and (not root.left):
                root = root.right

            return root

        # If the maximum depth has reached or the data is empty, assign a terminal node based on the most occuring classification value
        else:
            outcome = max(set(classification), key = classification.count)
            root = Node(outcome, None, None)
            return root

    # Inorder traversal of the tree to return an array used for displaying the structure of the tree
    def inorderTraversal(self):
        result = self._recursiveInorderTraversal(self.root, 0, [])
        return result

    def _recursiveInorderTraversal(self, root, depth, result):
        if result is None:
            result = []
        if root:
            self._recursiveInorderTraversal(root.left, depth+1, result)
            result.append([root, depth])
            self._recursiveInorderTraversal(root.right, depth+1, result)
        return result

# Set the attributes array for each column in the data
def setAttributes():
    age = []
    gender = []
    impulse = []
    pressurehight = []
    pressurelow = []
    glucose = []
    kcm = []
    troponin = []
    attributes = [age, gender, impulse, pressurehight, pressurelow, glucose, kcm, troponin]
    return attributes

# Set the classification array
def setClassification():
    classification = []
    return classification

# Read in the data through the csv file
def readData(filename, attributes, classification):
    try:
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                if len(row) >= len(attributes):
                    i = 0
                    # Add the data to the list of arrays
                    while i < len(attributes):
                        try:
                            attributes[i].append(float(row[i]))
                        except ValueError:
                            attributes[i].append(row[i])
                        i += 1
                    classification.append(row[i])
        return True
    except FileNotFoundError:
        print("The file does not exist.")
        return False
    except Exception as e:
        print("An error occurred:", str(e))
        return False

# Pre process the dataset for any missing values before using it for training and testing
def preProcess(attributes, classification):
    for i, attribute in enumerate(attributes):
        attribute = attribute[1:]
        if isContinuous(attribute):
            mean_value = sum(attribute[1:]) / len(attribute)
        else:
            mode_value = max(set(attribute), key=attribute.count)
        for j in range(len(attribute)):
            if math.isnan(attribute[j]):
                attributes[i][j] = mean_value

# Split the data into training data and test data
def splitData(attributes, classification):
    total_rows = len(attributes[0])
    test_size = (total_rows // 4) + 1
    # Extract the test data
    test_attributes = [col[:test_size] for col in attributes]
    test_classification = classification[:test_size]
    # Extract the training data
    new_attributes = [[col[0]] + col[test_size:] for col in attributes]
    new_classification = [classification[0]] + classification[test_size:]

    return (new_attributes, new_classification, test_attributes, test_classification)

# Check if an array has continuous values or not
def isContinuous(arr):
    unique_values = set(arr)
    return len(unique_values) > 2

# Merge sort array 1 and array 2 based on the sorting of array 1
def mergeSort(arr1, arr2):
    if len(arr1) > 1:
        mid = len(arr1) // 2
        left_arr1, left_arr2 = arr1[:mid], arr2[:mid]
        right_arr1, right_arr2 = arr1[mid:], arr2[mid:]

        mergeSort(left_arr1, left_arr2)
        mergeSort(right_arr1, right_arr2) 

        i = j = k = 0

        while i < len(left_arr1) and j < len(right_arr1):
            if left_arr1[i] < right_arr1[j]:
                arr1[k] = left_arr1[i]
                arr2[k] = left_arr2[i]
                i += 1
            else:
                arr1[k] = right_arr1[j]
                arr2[k] = right_arr2[j]
                j += 1
            k += 1

        while i < len(left_arr1):
            arr1[k] = left_arr1[i]
            arr2[k] = left_arr2[i]
            i += 1
            k += 1

        while j < len(right_arr1):
            arr1[k] = right_arr1[j]
            arr2[k] = right_arr2[j]
            j += 1
            k += 1

# Call merge sort to sort the attribute array and classification array according to the sorted attribute array 
def sortAttribute(attributeColumn, classification):
    arr1, arr2 = attributeColumn[:], classification[:]
    mergeSort(arr1, arr2)
    return (arr1, arr2)

# Calculate the gini index for the given classification array
def gini(arr):
    total = len(arr)
    if total == 0:
        return 0
    class_counts = {}
    for item in arr:
        if item not in class_counts:
            class_counts[item] = 0
        class_counts[item] += 1
    gini_impurity = 1
    for class_count in class_counts.values():
        proportion = class_count / total
        gini_impurity -= proportion ** 2
    return gini_impurity

# Find the gini weighted sum computation for the given array (column) of data
def giniAttribute(arr, classification):
    sorted_data, sorted_class = sortAttribute(arr, classification)
    total = len(arr)
    if total == 0:
        return 0
    
    # If the array has continuous values, split the array into 2 parts for each iteration through the array and calculate the highest gini index
    # and its split point in that array
    if isContinuous(arr):
        gini_attribute = 9
        current_val = None
        for split in range(1, total):
            if current_val == sorted_data[split]:   #If the value is the same as the previous, skip the value
                continue
            current_val = sorted_data[split]
            gini_split = 0
            left = sorted_data[:split]
            right = sorted_data[split:]
            left_class = sorted_class[:split]
            right_class = sorted_class[split:]
            proportion_left = len(left) / total
            proportion_right = len(right) / total

            gini_value_left = gini(left_class)
            gini_value_right = gini(right_class)

            gini_split += (proportion_left * gini_value_left) + (proportion_right * gini_value_right)
            if gini_attribute > gini_split:
                gini_attribute = gini_split
                splitPoint = right[0]
    # Else the array has catgeorical values, calculate the categorical value with the highest gini index
    else:
        gini_attribute = 0
        for value in set(arr):
            value_count = arr.count(value)
            proportion = value_count / total
            gini_value = gini([sorted_class[i] for i in range(total) if arr[i] == value])
            gini_attribute += proportion * gini_value
            splitPoint = value
    return (gini_attribute, splitPoint)

# Find the attribute column with the highest gini index
def findGini(dataset, classification):
    if len(dataset[0]) <= 1:
        return None
    maxGini = -1
    attrName = ""
    classGini = gini(classification[1:])
    for i in range(len(dataset)):
        arr = dataset[i]
        attrGini, splitVal = giniAttribute(arr[1:], classification[1:])
        splitPoint = classGini - attrGini
        if maxGini < splitPoint:
            maxGini = splitPoint
            attrName = dataset[i][0]
    return (attrName, splitVal, maxGini)

# Split the dataset into further subsets by assigning attribute values < splitpoint to the left side and attribute >= splitpoint to the right side 
def splitDataset(dataset, classification, attributeName, splitPoint):
    new_dataset_left, new_classification_left = [[col[0]] for col in dataset], [classification[0]]
    new_dataset_right, new_classification_right = new_dataset_left[:], new_classification_left[:]
    i = 0
    while i < len(dataset):
        if dataset[i][0] == attributeName:
            break
        i += 1
    for j in range(len(dataset)):
        new_dataset_left[j] = [dataset[j][k] for k in range(len(dataset[j])) if (k == 0) or (dataset[i][k] < splitPoint)]
        new_dataset_right[j] = [dataset[j][k] for k in range(len(dataset[j])) if (k == 0) or (dataset[i][k] >= splitPoint)]
    new_classification_left = [classification[k] for k in range(len(classification)) if (k == 0) or (dataset[i][k] < splitPoint)]
    new_classification_right = [classification[k] for k in range(len(classification)) if (k == 0) or (dataset[i][k] >= splitPoint)]
    new_dataset_left.pop(i)   #Remove the splitpoint attribute column
    new_dataset_right.pop(i)  #Remove the splitpoint attribute column
    return (new_dataset_left, new_dataset_right, new_classification_left, new_classification_right)

# For testing purposes, print the structure of the decision tree
def printDecisionTree(tree):
    lst = tree.inorderTraversal()
    print("\n\n")
    for l in lst:
        indent = "           " * (6 - l[1])
        print(indent, l[0].attribute, l[0].split, "\n")

# Read in the data, pre-process then split the data to training and test sets
# Build the decision tree using the training data
def buildDecisionTree():
    filename = "Heart Attack v3.csv"
    attributes = setAttributes()
    classification = setClassification()
    if not (readData(filename, attributes, classification)):
        return None
    #preProcess(attributes, classification)
    training_attributes, training_classification, test_attributes, test_classification = splitData(attributes, classification)
    decisionTree = DecisionTree()
    decisionTree.insert(training_attributes, training_classification)
    #printDecisionTree(decisionTree)   #For testing purposes
    accuracyTest(decisionTree, test_attributes, test_classification)

# Compute the accuracy of the decision tree using the test data
def accuracyTest(tree, test_attributes, test_classification):
    correct_predictions = 0
    total_samples = len(test_classification)
    for i in range(1, total_samples):
        predicted_class = predictClass(tree.root, test_attributes, i)
        if predicted_class == test_classification[i]:
            correct_predictions += 1

    accuracy = (correct_predictions / total_samples) * 100
    print(f"\nACCURACY: [--- {accuracy:.2f}% ---]")

# Predict the classification of the data using the decision tree
def predictClass(node, attributes, sample_index):
    if (not node.left) and (not node.right):
        return node.attribute
    i = 0
    while i < len(attributes):
        if node.attribute == attributes[i][0]:
            if attributes[i][sample_index] < node.split:
                return predictClass(node.left, attributes, sample_index)
            else:
                return predictClass(node.right, attributes, sample_index)
            i = 0
        i += 1
    return None

# Main function to run the program
def main():
    start_time = time.time()
    buildDecisionTree()
    total_time = (time.time() - start_time)
    print(f"\nTIME: [--- {total_time:.2f} seconds ---]")

if __name__ == '__main__':
    main()