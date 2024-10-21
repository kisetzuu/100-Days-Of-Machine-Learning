# Python Review

# 1. Variables and Data Types
int_var = 10  # Integer
float_var = 10.5  # Float
str_var = "Hello, World!"  # String
bool_var = True  # Boolean

# Lists (mutable)
list_var = [1, 2, 3, 4, 5]
list_var.append(6)  # Add an element
list_var[0] = 0  # Update an element
print(list_var)

# Tuples (immutable)
tuple_var = (1, 2, 3)
# tuple_var[0] = 0  # This would raise an error since tuples are immutable

# Dictionaries (key-value pairs)
dict_var = {"name": "Alice", "age": 25, "city": "New York"}
print(dict_var['name'])  # Access by key
dict_var['age'] = 26  # Update value

# 2. Control Flow (if-else, loops)
for num in range(1, 6):  # for loop from 1 to 5
    if num % 2 == 0:
        print(f"{num} is even")
    else:
        print(f"{num} is odd")

# 3. Functions and Lambdas
def add_numbers(a, b):
    """Returns the sum of two numbers."""
    return a + b

# Lambda function (anonymous function)
multiply = lambda x, y: x * y
print(multiply(5, 2))  # Output: 10

# 4. List Comprehensions
squared_numbers = [x**2 for x in range(1, 6)]  # Square of numbers from 1 to 5
print(squared_numbers)

# 5. Exception Handling
try:
    result = 10 / 0  # Division by zero error
except ZeroDivisionError as e:
    print(f"Error: {e}")

# 6. Classes and OOP
class Animal:
    """Basic class representing an animal."""
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} makes a sound."

# Inheritance
class Dog(Animal):
    def speak(self):
        return f"{self.name} barks."

dog = Dog("Rover")
print(dog.speak())  # Output: Rover barks

# 7. NumPy: Array Creation and Operations
import numpy as np

# Creating a 2D array
array = np.array([[1, 2], [3, 4]])
print(f"Array:\n{array}")

# Element-wise operations
array_squared = array**2
print(f"Squared Array:\n{array_squared}")

# Broadcasting: Adding a scalar to the array
array_plus_1 = array + 1
print(f"Array + 1:\n{array_plus_1}")

# 8. Pandas: DataFrame Basics
import pandas as pd

# Creating a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
df = pd.DataFrame(data)
print(f"DataFrame:\n{df}")

# Selecting a column
print(f"Names:\n{df['Name']}")

# Adding a new column
df['City'] = ['New York', 'San Francisco', 'Los Angeles']
print(f"Updated DataFrame:\n{df}")

# 9. Matplotlib: Basic Plot
import matplotlib.pyplot as plt

# Line plot
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

plt.plot(x, y, marker='o')
plt.title('Simple Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# 10. Seaborn: Visualizing Distributions
import seaborn as sns

# Example using the Iris dataset
iris = sns.load_dataset('iris')
sns.scatterplot(data=iris, x='sepal_length', y='sepal_width', hue='species')
plt.title('Iris Sepal Length vs Width')
plt.show()
