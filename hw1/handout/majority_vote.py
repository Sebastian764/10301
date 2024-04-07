import sys
import csv

def read_data(file_path: str) -> list[list[str]]:
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        data = list(reader)
        
    return data

def majority_vote(training_data:list[list[str]]) -> int:
    class_label_index = len(training_data[0])-1

    label_count = {"0":0, "1":0}
    for entry in training_data:
        class_label = entry[class_label_index]
        label_count[class_label] += 1
    
    zero_count = label_count["0"]
    one_count = label_count["1"]

    if zero_count <= one_count:
        return 1
    else:
        return 0

def predict(data:list[list[str]], majority_label:int) ->list[int]:
    # Predict labels for the test data based on the majority label
    prediction = []
    for i in range(len(data)):
        prediction.append(majority_label)

    return prediction


    
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

def write_predictions(file_path:str, predictions:list[int]):
    # Write predictions to a file
    with open(file_path, 'w') as file:
        for prediction in predictions:
            file.write(f"{prediction}\n")

def write_metrics(file_path:str, train_error:float, test_error:float):
    # Write metrics to a file
    with open(file_path, 'w') as file:
        file.write(f"error(train): {train_error}\n")
        file.write(f"error(test): {test_error}\n")



if __name__ == '__main__':
    if len(sys.argv) < 5:
        raise Exception("Usage: python3 majority_vote.py <train input>, <test input>, <train out>, <test out>, <metrics out>") 
    infile = sys.argv[1]
    outfile = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    print(f"The input file is: {infile}")
    print(f"The output file is: {outfile}")	

    # extract data
    training_data = read_data(infile)
    test_data = read_data(outfile)
        
    # add up and determine most common
    majority_label = majority_vote(training_data[1:])
    # majority_label_test = majority_vote(test_data[1:])

    # array with predicted values
    predict_train = predict(training_data[1:], majority_label)
    predict_test = predict(test_data[1:], majority_label)

    # calculate error
    error_train = calculate_error(training_data[1:], predict_train)
    error_test = calculate_error(test_data[1:], predict_test)

    # write error file
    write_metrics(metrics_out, error_train, error_test)

    # write out files
    write_predictions(train_out, predict_train)
    write_predictions(test_out, predict_test)


