import sys
import csv 
import numpy as np 


def read_data(file_path: str) -> list[list[str]]:
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        data = list(reader)
    return data

def calculate_entropy(data:list[list[str]], atr:int)->float:

    num_row = len(data)
    elems_tracker = dict()

    for row in data:
        if row[atr] not in elems_tracker:
            elems_tracker[row[atr]] = 1
        else:
            elems_tracker[row[atr]] += 1
    
    return -sum(counts/num_row * np.log2(counts/num_row) for counts in elems_tracker.values())

def calculate_error(data:list[list[str]], predictions:list[int])->float:
    # Calculate the error rate of the predictions)
    class_label_index = len(data[0])-1
    num_correct = 0
    total = 0

    for i in range(len(data)):
        if int(data[i][class_label_index]) == predictions[i]:
            num_correct += 1
        total += 1

    return (total - num_correct) / total

def majority_vote(training_data:list[list[str]]) -> int:
    class_label_index = len(training_data[0])-1

    label_count = {"0":0, "1":0}
    for entry in training_data:
        class_label = entry[class_label_index]
        label_count[class_label] += 1
    
    zero_count = label_count["0"]
    one_count = label_count["1"]
    # print('TESTTTTTTTTTTT')
    if zero_count <= one_count:
        print("test", 1, one_count)
        # return "1"
        return 1
    else:
        print("test", 0, zero_count)
        # return "0"
        return 0

def predict(data:list[list[str]], majority_label:int) ->list[int]:
    # Predict labels for the test data based on the majority label
    prediction = []
    for i in range(len(data)):
        prediction.append(majority_label)

    return prediction

def write_predictions(file_path:str, predictions:list[int]):
    # Write predictions to a file
    with open(file_path, 'w') as file:
        for prediction in predictions:
            file.write(f"{prediction}\n")

def write_metrics_entropy(file_path:str, train_error:float, entropy:float):
    # Write metrics to a file
    with open(file_path, 'w') as file:
        file.write(f"entropy: {entropy}\n")
        file.write(f"error: {train_error}\n")

def write_metrics(file_path:str, train_error:float, test_error:float):
    # Write metrics to a file
    with open(file_path, 'w') as file:
        file.write(f"error(train): {train_error}\n")
        file.write(f"error(test): {test_error}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Usage: python3 inspection.py <input> <output>") 
    infile = sys.argv[1]
    outfile = sys.argv[2]

    print(f"The input file is: {infile}")
    print(f"The output file is: {outfile}")	

    data = read_data(infile)
    training_data = data[1:]
    # add up and determine most common
    majority_label = majority_vote(training_data)

    # array with predicted values
    predict_train = predict(training_data, majority_label)

    # calculate error
    error_train = calculate_error(training_data, predict_train)

    # calculate entropy
    entropy = calculate_entropy(training_data, -1)

    # write error file
    write_metrics_entropy(outfile, error_train, entropy)

    # # write out files
    # write_predictions(train_out, predict_train)
    # write_predictions(test_out, predict_test)
