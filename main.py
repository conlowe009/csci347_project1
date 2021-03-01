# Project 1, Part 2 Code
# CSCI 347: Data Mining
# Connor Lowe
# 3 February 2021
import numpy as np
import math

# 2-Dimensional Numerical Data Set
arr_big = np.array([[1, 2, 3, 4, 5, 6],
                    [7, 8, 9, 10, 11, 12],
                    [13, 14, 15, 16, 17, 18],
                    [19, 20, 21, 22, 23, 24],
                    [25, 26, 27, 28, 29, 30],
                    [31, 32, 33, 34, 35, 36],
                    [37, 38, 39, 40, 41, 42],
                    [43, 44, 45, 46, 47, 48],
                    [49, 50, 51, 52, 53, 54]])

arr_small = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [10, 11, 12]])

# 2-Dimensional Categorical Data Set of Various Types
cat_arr = np.array([['Up', 'N', 'Helena', 'A'],
                    ['Left', 'S', 'Bozeman', 'A'],
                    ['Left', 'S', 'Missoula', 'B'],
                    ['Down', 'W', 'Helena', 'C']])

# 1 Dimensional Vectors
vec1 = np.array([1, 2, 3, 4, 5])
vec2 = np.array([1, 2, 3, 4, 5])
# vec2 = np.array([6, 7, 8, 9, 10])


# Function for the multivariate mean of a numerical data set
def __2dim_mean__(input_arr):
    total = np.zeros([len(input_arr[0])])
    num_elements = len(input_arr)
    # Sum for each column of 2 dimensional array
    for row in range(len(input_arr)):
        for col in range(len(input_arr[0])):
            x_i = input_arr[:, col]
            total[col] = total[col] + x_i[row]
    # Mean = sum total / number of elements
    mean = total
    return mean


# Function for sample covariance between two attributes
def __covariance__(attr_1, attr_2):
    x, y, total = 0, 0, 0
    n = len(attr_1)
    # Sum of each attribute
    for elem in range(n):
        x = x + attr_1[elem]
        y = y + attr_2[elem]
    # Mean of each attribute
    x_mean = x / n
    y_mean = y / n
    # Covariance sum
    for elem in range(n):
        total = total + ((attr_1[elem]) - x_mean) * ((attr_2[elem]) - y_mean)
    covariance = (1 / (n - 1)) * total
    return covariance


# Function for correlation between two attributes
def __correlation__(attr_1, attr_2):
    x, y, axb, a2, b2 = 0, 0, 0, 0, 0
    n = len(attr_1)
    # Sum of each attribute
    for elem in range(n):
        x = x + attr_1[elem]
        y = y + attr_2[elem]
    # Mean of each attribute
    x_mean = x / n
    y_mean = y / n
    # Obtains all the necessary sums and values for correlation
    for elem in range(n):
        # a = x - x_mean
        a = (attr_1[elem] - x_mean)
        # b = y - y_mean
        b = (attr_2[elem] - y_mean)
        # Sum of each product of a*b
        axb = axb + (a * b)
        # Sum of a^2
        a2 = a2 + (a * a)
        # Sum of b^2
        b2 = b2 + (b * b)
    # Correlation formula = a*b/sqrt(a^2 * b^2)
    correlation = axb / (math.sqrt(a2 * b2))
    return correlation


# Function for min by column
# ref: https://www.geeksforgeeks.org/program-find-minimum-maximum-element-array/
def getMin(input_arr):
    value = input_arr[0]
    n = len(input_arr)
    for i in range(1, n):
        value = min(value, input_arr[i])
    return value


# Function for max by column
# ref: https://www.geeksforgeeks.org/program-find-minimum-maximum-element-array/
def getMax(input_arr):
    value = input_arr[0]
    n = len(input_arr)
    for i in range(1, n):
        value = max(value, input_arr[i])
    return value


# Helper function for 1 dimensional means
def __1dim_mean__(input_arr):
    total = 0
    num_elements = 0
    # Sums each element and divides by number of elements
    for row in range(len(input_arr)):
        total = total + input_arr[row]
        num_elements = num_elements + 1
    mean = total / num_elements
    return mean


# Helper function for standard deviation
def __std_dev__(input_arr):
    total = 0
    num_elements = 0
    # Uses helper function fo find mean of 1D vector
    mean = __1dim_mean__(input_arr)
    # Calculates sum of (x - mean)^2, and keeps track of the number of elements
    for row in range(len(input_arr)):
        total = total + (input_arr[row] - mean) ** 2
        num_elements = num_elements + 1
    # Calculates standard deviation as sqrt((x-mean)^2/(n-1)
    std_dev = math.sqrt(total / (num_elements - 1))
    return std_dev


# Helper function that gets a list of unique instances from a data set
def __list_unique__(input_arr):
    unique = []
    for x in input_arr:
        if x not in unique:
            unique.append(x)
    return unique


# Function for range normalization
def __range_normal__(input_arr):
    row_len = len(input_arr)
    col_len = len(input_arr[0])
    # Creates an empty numpy array that is the same size as the input array
    rng_normal = np.empty([row_len, col_len])
    for row in range(row_len):
        for col in range(col_len):
            # Get max and min, subtract max - min
            max_min = getMax(input_arr[:, col]) - getMin(input_arr[:, col])
            # Get x_i and min, subtract x_i - min
            x_sub_min = (input_arr[row][col] - getMin(input_arr[:, col]))
            # (x_i - min)/(max - min)
            rng_normal[row][col] = x_sub_min / max_min
    return rng_normal


# Function for standard normalization
def __std_normal__(input_arr):
    row_len = len(input_arr)
    col_len = len(input_arr[0])
    # Creates an empty numpy array that is the same size as the input array
    std_normal = np.empty([row_len, col_len])
    for row in range(row_len):
        for col in range(col_len):
            x_i = input_arr[row][col]
            # Uses helper function to find the mean of a 1D array
            col_mean = __1dim_mean__(input_arr[:, col])
            # Uses helper function to find the standard deviation of each column
            std_dev = __std_dev__(input_arr[:, col])
            # Computes z-score normalization and writes to new array
            std_normal[row][col] = (x_i - col_mean) / std_dev
    return std_normal


# Function to find the covariance matrix for a set of data
def __covar_matrix__(input_arr):
    dim = np.array = input_arr.shape
    row_len = dim[1]
    # Creates an empty numpy array that is the number of attributes squared
    covar_matrix = np.empty([row_len, row_len])
    for row in range(row_len):
        for col in range(row_len):
            # Takes column slices of input array as we iterate through array
            x_i = input_arr[:, row]
            x_j = input_arr[:, col]
            # Uses covariance function with column slices and stores the values in the new array
            covar_matrix[row][col] = __covariance__(x_i, x_j)
    return covar_matrix


# Function to label encode a categorical set of data
def __label_encode__(input_arr):
    row_len = len(input_arr)
    col_len = len(input_arr[0])
    # Creates an empty numpy array that is the same size as the input array
    lbl_encoded = np.empty([row_len, col_len])
    for row in range(row_len):
        for col in range(col_len):
            # Takes column slice of categorical array
            x_i = input_arr[:, col]
            # Uses a helper function to get a list of unique instances from column slice
            x_u = __list_unique__(x_i)
            lbl_encoded[row][col] = x_u.index(x_i[row]) + 1
            ''' 
            Takes the indexed value to obtain the category from the column slice, then 
            compares the category from the column slice to the list of unique instances, 
            then uses the associated index from the list of unique instances to assign a 
            value to the label encoded matrix and the values are adjusted 
            to be greater than 0. 
            '''
    return lbl_encoded


# Main
print("\n2-Dimensional Numerical Data Set:\n", arr_big)
print("\n2-Dimensional Categorical Data Set:\n", cat_arr)
print("\n2-Dimensional Multivariate Mean:", __2dim_mean__(arr_big))
print("\nCovariance: %f" % __covariance__(vec1, vec2))
print("\nCorrelation: %f" % __correlation__(vec1, vec2))
print("\nRange Normalized Data:\n", __range_normal__(arr_big))
print("\nStandard Normalized Data:\n", __std_normal__(arr_big))
print("\nCovariance Matrix:\n", __covar_matrix__(arr_big))
print("\nLabel Encoded Matrix:\n", __label_encode__(cat_arr))
