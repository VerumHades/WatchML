def expect_type(value, expected_type):
    """
    Ensures a value matches a specific type.
    """
    if not isinstance(value, expected_type):
        actual = type(value).__name__
        expected = expected_type.__name__
        raise TypeError(f"Expected type {expected}, but received {actual} for value: {value}")
    return value

def expect_list_of_type(items, element_type):
    """
    Ensures the input is a list and every element matches the element_type.
    """
    expect_type(items, list)
    
    for item in items:
        expect_type(item, element_type)

    return items

def expect_not_none(value):
    """
    Ensures the value is not None.
    """
    if value is None:
        raise ValueError("Value expected to be defined, but received None.")
    return value

def expect_dict_structure(data, schema):
    """
    Validates that a dictionary contains specific keys with specific types.
    """
    expect_type(data, dict)
    
    for key, expected_type in schema.items():
        if key not in data:
            raise KeyError(f"Expected dictionary to contain key: '{key}'")
        expect_type(data[key], expected_type)

    return data

def expect_file_exists(file_path):
    """
    Ensures a file exists on the disk before processing.
    """
    import os
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Expected file at path: {file_path}")
    return file_path

def expect_non_empty(container):
    """
    Ensures a list, dict, or dataframe is not empty.
    """
    if len(container) == 0:
        raise ValueError(f"Expected non-empty container, but received: {container}")
    return container

def expect_numeric_range(value, min_val, max_val):
    """
    Ensures a number falls within a specific range (e.g., watch diameter).
    """
    expect_type(value, (int, float))
    
    if not (min_val <= value <= max_val):
        raise ValueError(f"Value {value} falls outside expected range [{min_val}, {max_val}]")
    return value

def expect_callable(func):
    """
    Ensures the value is a function or a lambda.
    """
    if not callable(func):
        actual = type(func).__name__
        raise TypeError(f"Expected a callable (function/lambda), but received: {actual}")
    return func

def expect_callable_return_type(func, expected_return_type, *args, **kwargs):
    """
    Ensures the callable returns a specific type when executed with given arguments.
    """
    expect_callable(func)
    
    result = func(*args, **kwargs)
    expect_type(result, expected_return_type)
    
    return func

def expect_lambda_with_type(func, expected_output_type, test_input):
    """
    Validates a lambda by running it against a test input and checking the result type.
    Example: expect_lambda_with_type(lambda x: x.upper(), str, "test")
    """
    expect_callable(func)
    
    try:
        output = func(test_input)
    except Exception as error:
        raise ValueError(f"Lambda failed execution with test input: {test_input}. Error: {error}")
        
    expect_type(output, expected_output_type)
    return func