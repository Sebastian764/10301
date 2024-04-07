import argparse
import inspection
import numpy as np
import matplotlib.pyplot as plt

val = 0
class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self, data:list[list[str]]):
        self.left = None
        self.right = None
        self.attr = None            # holds str with attribute
        self.vote = None            # holds majority vote in leaves, None for everything else
        self.depth = 0              # current depth
        self.curr_data = data
        self.print_count_dict = {"0": 0, "1": 0} # helps print tree
        self.print_split = None                  # helps print tree
        self.test = "test"

def printNode(node):
    print(f"left: {node.left}")
    print(f"right: {node.right}")
    print(f"attr: {node.attr}")
    print(f"vote: {node.vote}")
    print(f"depth: {node.depth}")
    print(f"curr_data: {node.curr_data}")


# should contain full data without attributes cut out
def create_decision_tree(max_depth, root):
    # global val
    # val += 1
    # print(f"ITERATION = {val}")
    # class_new_data contains values for class label
    class_new_data = []
    itr_data = root.curr_data[1:] # remove first row because it contains attributes
    for row_num in range(len(itr_data)):
        elem = itr_data[row_num][-1]
        class_new_data.append(elem)
        root.print_count_dict[elem] += 1


    # if max depth was reached or max num of attributes is reached OR entropy of class label is 0 (all values are equal) OR only 1 attr left (only 2 cols left)
    if max_depth == root.depth or inspection.calculate_entropy(class_new_data, -1) == 0 or len(itr_data[0]) < 2:
        print(class_new_data)
        root.vote = inspection.majority_vote(class_new_data)        
        return 

    # find attr to split on
    mutual_info_index = highest_mutual_info_index(itr_data)
    # root.test = "test"
    attr = root.curr_data[0][mutual_info_index]
    atrr_new_list = root.curr_data[0].copy()
    atrr_new_list.remove(attr)


    if root.depth == 0:
        root.attr = attr

    new_right_data = []
    new_left_data = []
    
    for row in range(len(itr_data)):
        if itr_data[row][mutual_info_index] == "0":
            # right data, for 0 values
            # create temp_row with all vals in curr row, exlcuding split val
            temp_row = [itr_data[row][col] for col in range(len(itr_data[0])) if col != mutual_info_index]
            new_right_data.append(temp_row)
        elif itr_data[row][mutual_info_index] == "1":
            # left data, for 1 values
            # create temp_row with all vals in curr row, exlcuding split val
            temp_row = [itr_data[row][col] for col in range(len(itr_data[0])) if col != mutual_info_index]
            new_left_data.append(temp_row)
        
    root.test = f'split attr: {len(new_right_data)}, {len(new_left_data)}' 
    # create leaf node if it is empty
    if len(new_right_data) == 0 or len(new_left_data) == 0:
        root.vote = inspection.majority_vote(class_new_data)        
        return 

    # add back attributes
    new_right_data.insert(0, atrr_new_list)
    new_left_data.insert(0, atrr_new_list)

    # create new left and right nodes
    # make sure to pass updated data
    # if 1 < len(new_right_data):
    right_node = Node(new_right_data)
    right_node.depth = root.depth + 1
    right_node.attr = attr
    right_node.print_split = "0"
    root.right = right_node
    create_decision_tree(max_depth, right_node)
        

    # if 1 < len(new_left_data):
    left_node = Node(new_left_data)
    left_node.depth = root.depth + 1
    left_node.attr = attr
    left_node.print_split = "1"
    root.left = left_node
    create_decision_tree(max_depth, left_node)



def print_tree(Node):
    if Node == None:
        return

    # print "depth" of tree
    for i in range(Node.depth):
        print("| ", end='')
    
    if Node.attr != None and Node.depth != 0:
        print(f"{Node.attr} = {Node.print_split}: ", end='')
    
    print(f"[{Node.print_count_dict['0']} 0/{Node.print_count_dict['1']} 1]")
    print_tree(Node.right)
    print_tree(Node.left)

def print_tree_to_file(tree:Node, file_path:str):

    def write_tree(Node, file):
        if Node == None:
            return
        
        for i in range(Node.depth):
            print("| ", end='', file=file)
        
        if Node.attr is not None and Node.depth != 0:
            print(f"{Node.attr} = {Node.print_split}: ", end='', file=file)
        
        print(f"[{Node.print_count_dict['0']} 0/{Node.print_count_dict['1']} 1]", file=file)
        write_tree(Node.right, file)
        write_tree(Node.left, file)
    
    # Open the file at the beginning of the function and pass the file object to the helper function
    with open(file_path, 'w') as file:
        write_tree(tree, file)

# def predict(node, example:dict):
#     if node.vote != None:
#         return node.vote
#     if node.attr == None:
#         attr_val = example[node.left.attr]
#     else:
#         # printNode(node)
#         attr_val = example[node.attr]
#     if attr_val == "1":
#         if node.left != None:
#             return predict(node.left, example)
#     else:
#         if node.right != None:
#             return predict(node.right, example)
    
def predict(node, example):
    # Base case: if the current node is a leaf node, return its vote
    if node.vote is not None:
        # assert(node.left == None and node.right == None)
        # assert(node.attr == None)
        return node.vote

    
    # if node.attr in example:
    attr_value = example[node.left.attr]
    # print(attr_value, type(attr_value))
    if attr_value == "1":
        # If attribute value is 1 and a left child exists, traverse left
        # if node.left is not None:
        assert(node.vote == None)
        return predict(node.left, example)
    elif attr_value == "0":
        # If attribute value is 0 and a right child exists, traverse right
        # if node.right is not None:
        assert(node.vote == None)
        return predict(node.right, example)
    print(attr_value)
    print(node.test)
    printNode(node)
    # raise Exception("Warning: Reached an unexpected condition in tree traversal.")


def predict_wrapper(data, tree):
    # Extract attribute names and prepare data for prediction
    attributes = data[0][:-1]  # Exclude the label column
    # print(attributes)
    examples = data[1:]  # Exclude the header row
    # print(examples)
    predictions = []

    # Iterate through each example to predict its label
    for example in examples:
        # print(example)
        example_dict = {attr: value for attr, value in zip(attributes, example[:-1])}  # Exclude the label
        # print(example_dict)
        prediction = predict(tree, example_dict)
        # print(type(prediction))
        predictions.append(prediction)

    return predictions

# def predict_wrapper(data:list[list[str]], tree:Node):
#     predictions = []
#     atr_list = data[:1]
#     atr_list = atr_list[0]
#     atr_list.pop() # remove class label
#     data_clean = data[1:]
#     # handle case where tree is single node
#     if tree.right.attr is None or tree.left.attr is None:
#         class_label = [row[-1] for row in data_clean]
#         major_vote = inspection.majority_vote(class_label)
#         for i in range(len(predictions)):
#             predictions.append(major_vote)
#         return predictions
        
#     atr_dict = dict()
#     for atr in atr_list:
#         atr_dict[atr] = None

#     # create dictionary with training data and test data
#     atr_dict_list = []
#     for i in range(len(data_clean)):
#         # we need one dict per row 
#         temp_atr_dict = atr_dict.copy()
#         for j in range(len(data_clean[0])-1):
#             curr_atr = atr_list[j]
#             temp_atr_dict[curr_atr] = data_clean[i][j]
#         atr_dict_list.append(temp_atr_dict)
    

#     curr_atr = tree.right.attr
#     for i in range(len(atr_dict_list)):
#         curr_atr_dict = atr_dict_list[i]
#         # find first atr because root.atr is always None
#         if curr_atr_dict[curr_atr] == "0":
#             prediction = predict(tree.right, curr_atr_dict)
#             predictions.append(prediction)
#         if curr_atr_dict[curr_atr] == "1":
#             prediction = predict(tree.left, curr_atr_dict)
#             predictions.append(prediction)
    
#     return predictions


def mutual_information(data:list[list[str]], label_index:int) -> float:
    # find entropy for class label
    HY = inspection.calculate_entropy(data, -1)

    label_data = [row[label_index] for row in data]
    class_data = [row[-1] for row in data]
    
    num_row = len(data)

    # finds corresponding values on class label depending on label we are searching
    class_if_zero = {"0":0, "1":0}
    class_if_one = {"0":0, "1":0}

    for i in range(len(label_data)):
        if label_data[i] == "1":
            class_if_one[class_data[i]] += 1
        else:
            class_if_zero[class_data[i]] += 1
    res = 0
    zero_val = sum(class_if_zero.values())
    one_val = sum(class_if_one.values())

    if zero_val > 0:
        for counts in class_if_zero.values():
            if counts == 0:
                continue
            res -= zero_val/num_row * (counts/zero_val * np.log2(counts/zero_val))
        # res += zero_val/num_row *-sum(counts/zero_val * np.log2(counts/zero_val) for counts in class_if_zero.values() if counts > 0)
    # else: 
    #     return 0
    if one_val > 0:
        for counts in class_if_one.values():
            if counts == 0:
                continue
            res -= one_val/num_row * (counts/one_val * np.log2(counts/one_val))
        # res += one_val/num_row *-sum(counts/one_val * np.log2(counts/one_val) for counts in class_if_one.values() if counts > 0)
    # else:
    #     return 0
    return HY - res

def highest_mutual_info_index(data:list[list[str]]) -> int:
    
    res = []
    for i in range(len(data[0])-1): 
        mutal_info_val = mutual_information(data, i)
        res.append(mutal_info_val)
    mutual_info_index = np.argmax(res)
    # print(max(res))
    # print(mutual_info_index)
    # print(res[mutual_info_index])
    
    return mutual_info_index

def percent_difference(array1, array2):
    # Ensure both arrays have the same length

    assert(len(array1) == len(array2))
    
    
    num_diff = 0
    
    for i in range(len(array1)):
        if int(array1[i]) != int(array2[i]):
            num_diff += 1


    return num_diff/ len(array1)


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, 
                        help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the training data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the test data should be written')
    parser.add_argument("metrics_out", type=str, 
                        help='path of the output .txt file to which metrics such as train and test error should be written')
    parser.add_argument("print_out", type=str,
                        help='path of the output .txt file to which the printed tree should be written')
    args = parser.parse_args()
    
    #Here's an example of how to use argparse
    print_out = args.print_out

    #Here is a recommended way to print the tree to a file
    # with open(print_out, "w") as file:
    #     print_tree(dTree, file)

    train_data = inspection.read_data(args.train_input)
    test_data = inspection.read_data(args.test_input)
    max_depth = args.max_depth
    train_out = args.train_out
    test_out = args.test_out
    metrics_out = args.metrics_out
    print_out = args.print_out

    # create tree
    decision_tree = Node(train_data)
    create_decision_tree(max_depth, decision_tree)

    print_tree(decision_tree)

    # predict train and test data
    train_predictions = predict_wrapper(train_data, decision_tree)
    test_predictions = predict_wrapper(test_data, decision_tree)


    # write predictions of training
    inspection.write_predictions(train_out, train_predictions)
    # write predictions of test
    inspection.write_predictions(test_out, test_predictions)

    clean_train_data = train_data[1:]
    clean_test_data = test_data[1:]


    # write error file
    train_error = inspection.calculate_error(clean_train_data, train_predictions)
    test_error = inspection.calculate_error(clean_test_data, test_predictions)
    inspection.write_metrics(metrics_out, train_error, test_error)

    # "print" tree
    print_tree_to_file(decision_tree, print_out)
    
    training = []
    testing = []
    for i in range(0,12):
        decision_tree = Node(train_data)
        create_decision_tree(i, decision_tree)

        train_predictions = predict_wrapper(train_data, decision_tree)
        test_predictions = predict_wrapper(test_data, decision_tree)

        clean_train_data = train_data[1:]
        clean_test_data = test_data[1:]
        train_error = inspection.calculate_error(clean_train_data, train_predictions)
        test_error = inspection.calculate_error(clean_test_data, test_predictions)
        training.append(train_error)
        testing.append(test_error)
    
    max_depth = [i for i in range(0,12)]
    plt.figure(figsize=(10, 6))
    plt.plot(max_depth, training, label='Training Error', marker='o')
    plt.plot(max_depth, testing, label='Testing Error', marker='s')
    plt.xlabel('Depth of the tree')
    plt.ylabel('Error')
    plt.title('Error vs. Tree Depth for Heart Disease Dataset')
    plt.xticks(max_depth)
    plt.legend()
    plt.grid(True)
    plt.show()

    # example run command:
    # /usr/local/bin/python3 /Users/pc/Documents/301/hw2/handout/decision_tree.py /Users/pc/Documents/301/hw2/handout/education_train.tsv /Users/pc/Documents/301/hw2/handout/education_test.tsv 2 /Users/pc/Documents/301/hw2/handout/test_output/education_2_train.txt /Users/pc/Documents/301/hw2/handout/test_output/education_2_test.txt /Users/pc/Documents/301/hw2/handout/test_output/education_2_metrics.txt /Users/pc/Documents/301/hw2/handout/test_output/education_2_print.txt
    # /usr/local/bin/python3 /Users/pc/Documents/301/hw2/handout/decision_tree.py /Users/pc/Documents/301/hw2/handout/heart_train.tsv /Users/pc/Documents/301/hw2/handout/heart_test.tsv 2 /Users/pc/Documents/301/hw2/handout/test_output/heart_2_train.txt /Users/pc/Documents/301/hw2/handout/test_output/heart_2_test.txt /Users/pc/Documents/301/hw2/handout/test_output/heart_2_metrics.txt /Users/pc/Documents/301/hw2/handout/test_output/heart_2_print.txt
    # /usr/local/bin/python3 /Users/pc/Documents/301/hw2/handout/decision_tree.py /Users/pc/Documents/301/hw2/handout/small_train.tsv /Users/pc/Documents/301/hw2/handout/small_test.tsv 2 /Users/pc/Documents/301/hw2/handout/test_output/small_2_train.txt /Users/pc/Documents/301/hw2/handout/test_output/small_2_test.txt /Users/pc/Documents/301/hw2/handout/test_output/small_2_metrics.txt /Users/pc/Documents/301/hw2/handout/test_output/small_2_print.txt