import numpy as np 
from scipy import integrate 
import random as rd



def get_points(lower, upper, n):
    """General function to get n points between lower and upper

    Args:
        lower (number): lower limit
        upper (number): upper limit
        n (int): number of points to generate

    Returns:
        list: list of generated values
    """
    delta = (upper - lower) / n
    p = []
    for i in range(n):
        p.append(lower + i * delta)

    return np.array(p)

def get_points_with_final(lower, upper, n):
    """General function to get n points between lower and upper. 
    This method will return a list of points that will contain 

    Args:
        lower (number): lower limit
        upper (number): upper limit
        n (int): number of points to generate

    Returns:
        list: list of generated values
    """
    delta = (upper - lower) / (n - 1)
    p = []
    for i in range(n-1):
        p.append(lower + i * delta)
    p.append(upper)
    return np.array(p)

def get_comb(n, k):
    """Get combinations n by k

    Args:
        n (int): value
        k (int): k 

    Returns:
        int: combiations 
    """
    return np.math.factorial(n) / (np.math.factorial(n - k) * np.math.factorial(k))

def remove_elements(origin, elements_to_remove):
    """Remove elements from a given list

    Args:
        origin (list): original list
        elements_to_remove (list): elements to remove from origin

    Returns:
        list: new list without elements_to_remove
    """
    #print(elements_to_remove)
    #print(origin)
    lst=[]
    for k in origin:
        if k not in elements_to_remove:
            lst.append(k)
        #else:
            #print("cant remove %s" % k)
    #print(lst)
    return lst

def get_indexes(origin, elements):
    """Get indexes from elements in origin

    Args:
        origin (list): original list
        elements (list): List of elements to get indexes

    Returns:
        list: list of indexes of elements in origin
    """
    indexes=[]
    for val in elements:
        indexes.append(origin.index(val))

    return indexes

"""
def quantile(val, cdf, points):
    
    position = bisect(cdf, val)
    if position >= len(points):
        position -= 1
    return points[position]
"""

def get_k_folds(variables, k_value=5):
    """Get k-folds

    Args:
        variables (list): list of original variables
        k_value (int, optional): Number of k-folds. Defaults to 5.

    Returns:
        list: list of k-folds variables
    """
    list_loaded = []
    p = rd.sample(variables, k=k_value)
    list_loaded.extend(p)
    current = variables
    k_folds = [p]

    i = 1
    max_i = 80
    while len(current) > 0 and i < max_i:
        current = [elem for elem in current if elem not in list_loaded]
        if len(current) > 0:
            if len(current) < k_value:
                k_value = len(current)
            p = rd.sample(current, k=k_value)
            list_loaded.extend(p)
            k_folds.append(p)
            i = i + 1

    return k_folds

def get_integral_square_function(f, used_points):
    """Get the integral of a function. This will calculate numerical integration
        \\int_{I} f(x)^2 dx
        
    Args:
        f (list): function values
        used_points (list): discrete domain where f belongs

    Returns:
        int: integral_value
    """
    return integrate.simpson(f ** 2, x=used_points)

def get_function_error(f1, f2, used_points):
    """Get the error value between 2 functions. This will calculate numerical integration
        \\int_{I} (f1(x)-f2(x))^2 dx
        
    Args:
        f1 (list): function 1 values
        f2 (list):  function 2 values
        used_points (list): discrete domain where f1 and f2 belongs

    Returns:
        int: error_value
    """
    return integrate.simpson((f1 - f2) ** 2, x=used_points)

def get_mean_from_function(f, used_points):
    """Get the mean value of a function. This will calculate numerical integration
        \\int_{I} f(x) dx
        
    Args:
        f (list): function values
        used_points (list): discrete domain where mean_function belongs

    Returns:
        int: mean_value
    """
    return integrate.simpson(f, x=used_points)

def get_error_function(f1, f2):
    return (f1-f2)**2

def get_derivative(grid_size, order, elements):
    h = 1 / grid_size
    d = 1 / (h ** order)
    f_prime = []
    n_elements = len(elements)
    elm = np.array([l for l in elements])
    for i in range(n_elements):
        val_2_add = 0
        for k in range(order):
            val = (-1) ** (k + order)
            val = val * get_comb(order, k)
            if k + i < n_elements:
                val = val * elm[k + i]
            else:
                val = val * elm[n_elements - (k + i)]
            val_2_add = val_2_add + val
        f_prime.append(d * val_2_add)
    return np.array([l for l in f_prime])

def get_linear_operator(grid_size, operator_type, function):
    """Get Linear operator

    Args:
        grid_size (int): size of observed values
        operator_type (string): Linear operator. Options ('D1','D2','D3','D4','D2D1')
        function (list): function values

    Returns:
        list: values of linear operator
    """
    if operator_type == "D1":
        return get_derivative(grid_size, 1, function)
    elif operator_type == "D2":
        return get_derivative(grid_size, 2, function)
    elif operator_type == "D3":
        return get_derivative(grid_size, 3, function)
    elif operator_type == "D4":
        return get_derivative(grid_size, 4, function)
    elif operator_type == "D2D1":
        return get_derivative(grid_size, 2, function) + get_derivative(grid_size, 1, function)

def get_moment_k_value(f,used_points,k=1):
    return integrate.simpson(f**k, x=used_points)