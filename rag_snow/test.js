var text = `In Python, a **function** is a reusable block of organized, executable code that performs a single, related action. Functions help break our program into smaller, modular chunks. As our program grows larger and larger, functions make it more manageable, readable, and reusable.

Here is a comprehensive guide to understanding and using functions in Python.

---

## 1. Anatomy of a Function

To define a function in Python, you use the def keyword, followed by the function name, parentheses (), and a colon :. The block of code inside the function is indented.

def greet_user(username):
    """Display a simple greeting."""  # Docstring
    print(f"Hello, {username}!")      # Function body

### Key Components:

* **def Keyword:** Signals the start of a function definition.
* **Function Name:** Follows standard Python naming conventions (snake_case).
* **Parameters (Optional):** Inputs passed into the function inside the parentheses.
* **Docstring (Optional):** A triple-quoted string that describes what the function does.
* **Function Body:** The indented block of code that executes when the function is called.
* **return Statement (Optional):** Sends a result back to the caller. If omitted, the function returns None by default.

---

## 2. Arguments vs. Parameters

While often used interchangeably, there is a technical difference:

* **Parameters:** The variables listed in the function's definition (e.g., username above).
* **Arguments:** The actual values sent to the function when it is called (e.g., greet_user("Alice") $\rightarrow$ "Alice" is the argument).

### Types of Arguments

Python offers highly flexible ways to pass arguments into functions:

#### A. Positional Arguments

Arguments must be passed in the exact order they are defined.

def describe_pet(animal_type, pet_name):
    print(f"I have a {animal_type} named {pet_name}.")

describe_pet("hamster", "Harry")  # Order matters!

#### B. Keyword Arguments

You explicitly state the parameter name and value when calling the function. Order no longer matters.

describe_pet(pet_name="Harry", animal_type="hamster")

#### C. Default Values

You can provide default values for parameters. If an argument isn't provided during the function call, Python uses the default.

def describe_pet(pet_name, animal_type="dog"):
    print(f"I have a {animal_type} named {pet_name}.")

describe_pet("Willie")  # Uses default "dog"
describe_pet("Harry", "hamster")  # Overrides default

> **Note:** Parameters with default values *must* be placed after parameters without default values in the function definition.

---

## 3. Arbitrary Arguments (*args and kwargs)

Sometimes you won't know ahead of time how many arguments a function needs to accept. Python handles this with two special notations:

### *args (Arbitrary Positional Arguments)

Collects extra positional arguments into a **tuple**.

def make_pizza(size, *toppings):
    print(f"Making a {size}-inch pizza with the following toppings:")
    for topping in toppings:
        print(f"- {topping}")

make_pizza(12, "pepperoni", "mushrooms", "green peppers")

### kwargs (Arbitrary Keyword Arguments)

Collects extra keyword arguments into a **dictionary**.

def build_profile(first, last, **user_info):
    user_info['first_name'] = first
    user_info['last_name'] = last
    return user_info

user_profile = build_profile('albert', 'einstein', location='princeton', field='physics')
print(user_profile)

---

## 4. Return Values

Functions can process data and return a value (or values) back to the main program using the return statement.

### Returning Single vs. Multiple Values

Python allows a function to return multiple values separated by commas. Under the hood, Python packages them into a **tuple**.

def get_coordinates():
    x = 10
    y = 20
    return x, y  # Returns a tuple (10, 20)

x_coords, y_coords = get_coordinates()  # Unpacking the tuple

---

## 5. Variable Scope

Scope determines where a variable can be accessed within your code.

* **Local Scope:** Variables created inside a function belong to that function's local scope and cannot be accessed outside of it.
* **Global Scope:** Variables created in the main body of the Python file are global and can be read inside functions.

x = "global"

def my_func():
    y = "local"
    print(x)  # Accessible (reads global)
    print(y)  # Accessible

my_func()
# print(y)  # NameError: name 'y' is not defined


If you need to *modify* a global variable inside a function, you must use the global keyword (though this is generally discouraged in clean coding practices).

---

## 6. Lambda Functions (Anonymous Functions)

In Python, a lambda function is a small, anonymous function defined without the def keyword. It can take any number of arguments but can only have **one expression**.

Syntax: "lambda arguments: expression"

python
# Standard function
def add(a, b):
    return a + b

# Equivalent Lambda function
add_lambda = lambda a, b: a + b

print(add_lambda(5, 3))  # Outputs: 8


Lambda functions are incredibly useful when passed as arguments to higher-order functions like map(), filter(), or sorted().

---

## 7. Best Practices for Writing Functions

To keep your Python code clean, readable, and maintainable, aim for the following habits:

* **Single Responsibility Principle:** A function should do *one* thing, and do it well. If a function is doing multiple unrelated tasks, split it up.
* **Use Clear Names:** Name your functions using verbs that describe what they do (e.g., calculate_total(), not total()).
* **Type Hinting:** Modern Python allows you to specify expected data types for parameters and return values to improve code clarity and IDE autocomplete.

def add_numbers(a: int, b: int) -> int:
    return a + b




* **Keep them Short:** If a function spans more than a screen or two of code, it’s usually a sign that it should be broken down into smaller helper functions.";`


var rag = new RAGUtil();
// var chunkList = rag.createChunks(text, 100, 10);
// var rawEmbeddings = rag.createEmbeddings(chunkList);
// rag.storeEmbeddings(chunkList, rawEmbeddings);

var query = "What are the best practices for writing functions?";
// var queryEmbeddings = rag.createQueryEmbedding(query)



var topChunks = rag.search(query, 5);

var context = topChunks.join("\n---\n");

var systemPrompt = `You are a helpful assistant. Answer the user's question using ONLY the provided text context.
If the answer cannot be found in the context, say 'I cannot find the answer in the document.'
Do not make up information or use outside knowledge.`

var userPrompt = `Context:
${context}
Question: ${query}
Answer:`

gs.print(rag.callLLM(systemPrompt, userPrompt));

/*
OUTPUT:

*** Script: The document lists the following best‑practice habits for writing functions:

- **Single Responsibility Principle** – a function should do one thing and do it well; if it is handling multiple unrelated tasks, split it into separate functions.  
- **Use Clear Names** – name functions with verbs that describe what they do (e.g., `calculate_total` rather than just `total`).  
- **Type Hinting** – specify expected data types for parameters and return values to improve code clarity and IDE autocomplete.  
- **Keep Them Short** – if a function grows beyond a screen or two of code, break it into smaller helper functions.

*/

