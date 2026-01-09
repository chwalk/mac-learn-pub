import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.model_selection as ms

# Function to print summaries of the data
def print_data_summary(data):
    numpified_data = np.array(data)
    # Flatten if the array is 2D
    if numpified_data.ndim == 2 and numpified_data.shape[1] == 1:
        flattened_numpified_data = numpified_data.flatten()
    else:
        flattened_numpified_data = numpified_data
    # Format first 5 and last 5 values
    first_five = ", ".join(f"{x:7.3f}" for x in flattened_numpified_data[:5])
    last_five = ", ".join(f"{x:7.3f}" for x in flattened_numpified_data[-5:])
    print(f"[{first_five}, ..., {last_five}]")

def create_linear_regression_model(predictors, response):
    model = lm.LinearRegression()
    model.fit(predictors, response)

    return model

def perform_linear_regression_prediction(model, predictors):
    prediction = model.predict(predictors)

    return prediction

def linear_regression(simple_or_multiple, cars_df, create_testing_set, one_hot_encode):
    if simple_or_multiple == 'multiple' and one_hot_encode:
        carname_dummies = pd.get_dummies(cars_df['CarName'], prefix='CarName')
        predictors = pd.concat([cars_df[['carlength', 'carwidth', 'carheight']], carname_dummies], axis=1).values
        response = cars_df['horsepower'].values
    elif simple_or_multiple == 'multiple':
        predictors = cars_df[['carlength', 'carwidth', 'carheight']].values
        response = cars_df['horsepower'].values
    else:
        predictors = cars_df[['enginesize']].values
        response = cars_df['price'].values

    # Set training and testing variable values
    if create_testing_set:
        training_predictors, testing_predictors, training_response, testing_response \
            = ms.train_test_split(
                predictors, response, test_size=0.25, random_state=42)
    else:
        training_predictors = predictors
        training_response = response
        testing_predictors = None
        testing_response = None
        testing_prediction = None

    model = create_linear_regression_model(training_predictors, training_response)
    training_prediction = perform_linear_regression_prediction(model, training_predictors)

    if create_testing_set:
        testing_prediction = perform_linear_regression_prediction(model, testing_predictors)

    return training_prediction, training_response, training_predictors, testing_prediction, testing_response, testing_predictors

def printing_values(simple_or_multiple, test_set_created, prediction, response, predictors):
    testing_or_training = 'testing' if test_set_created else 'training'
    if simple_or_multiple == 'simple':
        print(f'"The {testing_or_training} data predictors, prediction and response values:"')
        print_data_summary(predictors)
    else:
        print(f'"The {testing_or_training} data prediction and response values:"')
    print_data_summary(prediction)
    print_data_summary(response)

def plotting_values(simple_or_multiple, test_set_created, prediction, response, predictors):
    color = 'green' if test_set_created else 'blue'
    label = 'Testing Data' if test_set_created else 'Training Data'
    if simple_or_multiple == 'multiple':
        predictors = predictors[:, 0]
    else:
        predictors = predictors.flatten()
    plt.scatter(predictors, response, color=color, label=label)
    plt.plot(predictors, prediction, color = 'red', label = 'Best Fit Line')
    xlabel = 'Engine Size' if simple_or_multiple == 'simple' else 'Car Dimensions'
    plt.xlabel(xlabel)
    ylabel = 'Price' if simple_or_multiple == 'simple' else 'Horsepower'
    plt.ylabel(ylabel)
    title = f'Linear Regression: {xlabel} vs {ylabel} ({label})'
    plt.title(title)
    plt.legend()
    plt.show()

def input_prompts():
    print('Welcome to the most restricted linear regression program ever!')
    print('Please input the type of linear regression you would like to perform: simple or multiple?')

    simple_or_multiple = input().strip().lower()
    if simple_or_multiple not in ['simple', 'multiple']:
        print('Invalid input. Defaulting to simple linear regression.')
        simple_or_multiple = 'simple'

    print(f'You have selected {simple_or_multiple} linear regression.')
    print('Next, would you like to create a testing set? (yes or no)')

    testing_input = input().strip().lower()
    if testing_input not in ['yes', 'no']:
        print('Invalid input. Defaulting to no testing set.')
        use_testing_set = False
    else:
        use_testing_set = testing_input == 'yes'

    print(f'You have selected to {"create" if use_testing_set else "not create"} a testing set.')
    print('Finally, would you like to one-hot encode the car names? (yes or no)')

    if simple_or_multiple == 'simple':
        print('Note: One-hot encoding is only applicable for multiple linear regression. Defaulting to no one-hot encoding.')
        one_hot_encode = False
    else:
        encoding_input = input().strip().lower()
        if encoding_input not in ['yes', 'no']:
            print('Invalid input. Defaulting to no one-hot encoding.')
            one_hot_encode = False
        else:
            one_hot_encode = encoding_input == 'yes'

    print(f'You have selected to {"one-hot encode" if one_hot_encode else "not one-hot encode"} the car names.')
    print('Performing linear regression...')

    return simple_or_multiple, use_testing_set, one_hot_encode

def main():

    cars_df = pd.read_csv('cars.csv')

    simple_or_multiple, use_testing_set, one_hot_encode = input_prompts()

    training_prediction, training_response, training_predictors, testing_prediction, testing_response, testing_predictors \
        = linear_regression(simple_or_multiple, cars_df, use_testing_set, one_hot_encode)

    if use_testing_set:
        printing_values(simple_or_multiple, False, training_prediction, training_response, training_predictors)
        printing_values(simple_or_multiple, use_testing_set, testing_prediction, testing_response, testing_predictors)
        plotting_values(simple_or_multiple, False, training_prediction, training_response, training_predictors)
        plotting_values(simple_or_multiple, use_testing_set, testing_prediction, testing_response, testing_predictors)
    else:
        printing_values(simple_or_multiple, use_testing_set, training_prediction, training_response, training_predictors)
        plotting_values(simple_or_multiple, use_testing_set, training_prediction, training_response, training_predictors)

if __name__ == "__main__":
    main()