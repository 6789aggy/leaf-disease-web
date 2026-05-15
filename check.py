import os

train_count = len(os.listdir("train/images"))
valid_count = len(os.listdir("valid/images"))
test_count = len(os.listdir("test/images"))

total = train_count + valid_count + test_count

print("Train:", train_count)
print("Valid:", valid_count)
print("Test:", test_count)
print("Total:", total)