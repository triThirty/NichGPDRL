import time

# Set the current time as time 0
start_time = time.time()


# Function to compute the passing time since time 0
def get_passing_time():
    return time.time() - start_time


# Example usage
time.sleep(2)  # Simulate some delay
print(f"Time passed since start: {get_passing_time()} seconds")

time.sleep(3)  # Simulate additional delay
print(f"Time passed since start: {get_passing_time()} seconds")
