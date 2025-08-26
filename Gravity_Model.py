
## Developed by Sadra Daneshvar
### Feb 2, 2023

import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for data manipulation


class GravityModel:
    """
    A class to implement the Gravity Model for transportation planning.
    """
    
    def __init__(self):
        # Initialize attributes to store results
        self.final_matrix = None
        self.iteration_count = 0
        self.error = 0.0
        self.stop_reason = ""
        self.Ai = None
        self.Bj = None
        self.Tij = None
        self.O_adjusted = None
        self.D_adjusted = None
        
    def format_matrix(self, matrix, matrix_name):
        """Format and print matrices for visualization."""
        matrix_size = matrix.shape[0]  # Get the number of rows in the matrix
        # Create column names for the matrix
        column_names = [f"Zone {i}" for i in range(1, matrix_size + 1)]
        # Convert the matrix into a pandas DataFrame for pretty printing
        formatted_matrix = pd.DataFrame(
            matrix, columns=column_names, index=column_names
        )
        # Print the formatted matrix
        print(f"{matrix_name}:\n", formatted_matrix, "\n")

    def normalize_od_matrices(self, O, D):
        """Normalize O and D so their sums are equal."""
        sum_O = np.sum(O)  # Sum of all elements in O
        sum_D = np.sum(D)  # Sum of all elements in D
        
        # Adjust O or D if their sums are not equal
        if sum_O != sum_D:
            if sum_O < sum_D:
                correction_ratio = sum_D / sum_O  # Calculate correction ratio
                O = O * correction_ratio  # Adjust O by the correction ratio
            else:
                correction_ratio = sum_O / sum_D  # Calculate correction ratio
                D = D * correction_ratio  # Adjust D by the correction ratio
        
        return O, D

    def calculate_balancing_factors(self, O, D, deterrence_matrix):
        """Calculate Ai and Bj balancing factors."""
        n = len(O)
        Ai = np.ones(n)  # Ai balancing factor, initially set to 1 for each zone
        Bj = np.ones(n)  # Bj balancing factor, initially set to 1 for each zone
        
        return Ai, Bj

    def update_balancing_factors(self, O, D, deterrence_matrix, Ai, Bj):
        """Update Ai and Bj balancing factors for one iteration."""
        n = len(O)
        
        # Update Ai balancing factors
        for i in range(n):
            Ai[i] = 1 / (np.sum(Bj * D * deterrence_matrix[i, :]) + 1e-9)

        # Update Bj balancing factors
        Bj_new = np.ones(n)  # Temporary array for new Bj values
        for j in range(n):
            Bj_new[j] = 1 / (np.sum(Ai * O * deterrence_matrix[:, j]) + 1e-9)
        
        return Ai, Bj_new

    def calculate_trip_matrix(self, O, D, deterrence_matrix, Ai, Bj):
        """Calculate the trip matrix Tij."""
        return np.outer(Ai * O, Bj * D) * deterrence_matrix

    def calculate_error(self, O, D, Tij):
        """Calculate the model error."""
        T = np.sum(O)  # Total number of trips
        error = (
            np.sum(np.abs(O - np.sum(Tij, axis=1)))
            + np.sum(np.abs(D - np.sum(Tij, axis=0)))
        ) / T
        return error

    def create_final_matrix_dataframe(self, Tij):
        """Create the final formatted matrix with origin and destination sums."""
        n = Tij.shape[0]
        final_matrix = pd.DataFrame(
            Tij,
            columns=[f"Zone {i}" for i in range(1, n + 1)],
            index=[f"Zone {i}" for i in range(1, n + 1)],
        )
        final_matrix["Origin"] = final_matrix.sum(axis=1)  # Add sum of rows as Origin
        final_matrix.loc["Destination"] = final_matrix.sum()  # Add sum of columns as Destination
        return final_matrix

    def run(self, O, D, cost_matrix, deterrence_matrix, error_threshold=0.01, improvement_threshold=1e-4, max_iterations=1e6):
        """
        Execute the gravity model algorithm.
        
        Parameters:
        -----------
        O : array-like
            Origin matrix
        D : array-like  
            Destination matrix
        cost_matrix : array-like
            Cost matrix
        deterrence_matrix : array-like
            Deterrence matrix
        error_threshold : float, optional
            Error threshold for stopping condition (default: 0.01)
        improvement_threshold : float, optional
            Improvement threshold for stopping condition (default: 1e-4)
        
        Returns:
        --------
        self : GravityModel
            Returns self for method chaining
        """
        # Print the initial cost matrix and deterrence matrix
        self.format_matrix(cost_matrix, "Initial Cost Matrix")
        self.format_matrix(deterrence_matrix, "Deterrence Matrix")

        # Normalize O and D so their sums are equal
        O_normalized, D_normalized = self.normalize_od_matrices(O.copy(), D.copy())
        self.O_adjusted = O_normalized
        self.D_adjusted = D_normalized

        n = len(O_normalized)  # Number of zones
        T = np.sum(O_normalized)  # Total number of trips

        # Initialize balancing factors Ai and Bj
        Ai, Bj = self.calculate_balancing_factors(O_normalized, D_normalized, deterrence_matrix)

        previous_error = np.inf  # Initialize previous error to infinity
        iteration_count = 0  # Initialize iteration count
        stop_reason = ""  # Initialize stop reason string

        # Iterative process
        while True:
            iteration_count += 1  # Increment iteration count

            # Update balancing factors
            Ai, Bj_new = self.update_balancing_factors(O_normalized, D_normalized, deterrence_matrix, Ai, Bj)

            # Calculate Tij matrix for the model
            Tij = self.calculate_trip_matrix(O_normalized, D_normalized, deterrence_matrix, Ai, Bj_new)

            # Calculate the error of the model
            error = self.calculate_error(O_normalized, D_normalized, Tij)

            # Calculate the change in error from the previous iteration
            error_change = abs(previous_error - error)

            # Check stopping conditions
            if error < error_threshold:
                stop_reason = "Error threshold met"  # Set stop reason
                break  # Break the loop if error threshold is met
            elif error_change < improvement_threshold:
                stop_reason = "Slow improvement"  # Set stop reason
                break  # Break the loop if improvement is slow
            elif iteration_count >= max_iterations:
                stop_reason = "Max Iterations met" # Set stop reason
                break # Break the loop if too many iterations are performed

            previous_error = error  # Update the previous error
            Bj = Bj_new  # Update Bj with new values

        # Store results as instance attributes
        self.Ai = Ai
        self.Bj = Bj
        self.Tij = Tij
        self.iteration_count = iteration_count
        self.stop_reason = stop_reason
        self.error = error
        self.final_matrix = self.create_final_matrix_dataframe(Tij)

        # Print the final results
        print("Final OD Matrix:")
        print(self.final_matrix.round(3), "\n")
        print(f"Number of Iterations: {self.iteration_count}")
        print(f"Stopping Condition: {self.stop_reason}")
        print(f"Error: {self.error*100:.3f}%")

        return self

    def __call__(self, O, D, cost_matrix, deterrence_matrix, error_threshold=0.01, improvement_threshold=1e-4):
        """
        Make the instance callable to preserve original functionality.
        """
        return self.run(O, D, cost_matrix, deterrence_matrix, error_threshold, improvement_threshold)


# Create a function that maintains backward compatibility
def gravity_model(O, D, cost_matrix, deterrence_matrix, error_threshold=0.01, improvement_threshold=1e-4):
    """
    Backward compatibility function for the original gravity_model function.
    """
    model = GravityModel()
    return model.run(O, D, cost_matrix, deterrence_matrix, error_threshold, improvement_threshold)
