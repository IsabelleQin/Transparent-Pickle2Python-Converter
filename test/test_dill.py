import dill

# Define a lambda function
square = lambda x: x ** 2

# Serialize (dump) the function to a file
with open("dill_square_func.pkl", "wb") as file:
    dill.dump(square, file)

print("Function successfully serialized!")

# Deserialize (load) the function back from the file
with open("dill_square_func.pkl", "rb") as file:
    loaded_square = dill.load(file)

# Test the restored function
print(loaded_square(5))  # Outputs: 25
