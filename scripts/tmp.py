import numpy as np
from dante.obj_functions import Ackley
from dante.neural_surrogate import AckleySurrogateModel
from dante.tree_exploration import TreeExploration
from dante.utils import generate_initial_samples

# Define parameters
NUM_DIMENSIONS = 20
NUM_INITIAL_SAMPLES = 200
NUM_ACQUISITIONS = 1000
SAMPLES_PER_ACQUISITION = 20

# Initialise the objective function and surrogate model
obj_function = Ackley(dims=NUM_DIMENSIONS)
surrogate = AckleySurrogateModel(input_dims=NUM_DIMENSIONS, epochs=500)

# Generate initial samples
input_x, input_y = generate_initial_samples(
    obj_function, num_init_samples=NUM_INITIAL_SAMPLES, apply_scaling=True
)

# Main optimisation loop
for i in range(NUM_ACQUISITIONS):
    # Train surrogate model and create tree explorer
    trained_surrogate = surrogate(input_x, input_y)
    tree_explorer = TreeExploration(func=obj_function, model=trained_surrogate, num_samples_per_acquisition=SAMPLES_PER_ACQUISITION)
    
    # Perform tree exploration to find promising samples
    new_x = tree_explorer.rollout(input_x, input_y, iteration=i)
    new_y = np.array([obj_function(x, apply_scaling=True) for x in new_x])
    
    # Update dataset with new samples
    input_x = np.concatenate((input_x, new_x), axis=0)
    input_y = np.concatenate((input_y, new_y))
    print(f'number of data is {len(input_y)}')
    print(f'current best y value is {input_y.max()}')
    
    # Check for convergence
    if np.isclose(np.array([obj_function(x, apply_scaling=False, track=False) for x in input_x]).min(), 0.0):
        print(f"Optimal solution found after {i+1} iterations.")
        break

# Print results
best_index = np.argmax(input_y)
print(f"Best solution: {input_x[best_index]}")
print(f"Best objective value: {obj_function(input_x[best_index], apply_scaling=False, track=False)}")