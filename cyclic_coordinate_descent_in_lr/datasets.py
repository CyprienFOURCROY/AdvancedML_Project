import numpy as np
import pandas as pd
import os

def generate_synthetic_data(p: float, n: int, d: int, g: float, save_path: str = None, random_state: int = None) -> pd.DataFrame:
    """
    Generate a synthetic dataset with a binary target variable Y sampled from Bernoulli(p).
    The feature vectors X are drawn from a d-dimensional multivariate normal distribution:
    - When Y = 0, X follows N(0, S) where S[i, j] = g^|i-j|.
    - When Y = 1, X follows N((1, 1/2, 1/3, ..., 1/d), S) where S[i, j] = g^|i-j|.

    Parameters:
        p (float): Probability of Y=1 in Bernoulli distribution.
        n (int): Number of observations to generate.
        d (int): Dimensionality of the feature space.
        g (float): Parameter controlling the covariance structure.
        save_path (str, optional): Path to save the dataset as a CSV file.
        random_state (int, optional): Random seed for reproducibility.


    Returns:
        pd.DataFrame: A DataFrame with d feature columns (X1, X2, ..., Xd) and one target column (Y).
    """
     
    if not (0 <= p <= 1):
        raise ValueError("p must be in the range [0,1]")
    if not (isinstance(n, int) and n > 0):
        raise ValueError("n must be a positive integer")
    if not (isinstance(d, int) and d > 0):
        raise ValueError("d must be a positive integer")
    if not (isinstance(g, float) and g >= 0):
        raise ValueError("g must be a non-negative float") 
    
    if random_state is not None:
        np.random.seed(random_state)

    # Generate class labels Y from Bernoulli distribution with probability p
    Y = np.random.binomial(1, p, size=n)
    
    # Construct covariance matrix S with S[i, j] = g^|i-j|
    indices = np.arange(d)
    S = np.array([[g ** abs(i - j) for j in indices] for i in indices])
    
    # Define means for X|Y=0 and X|Y=1
    mean_0 = np.zeros(d)
    mean_1 = np.array([1 / (i + 1) for i in range(d)])
    
    # Generate feature space X
    X = np.array([
        np.random.multivariate_normal(mean_0, S) if y == 0 else np.random.multivariate_normal(mean_1, S)
        for y in Y
    ])
    
    # Create DataFrame
    columns = [f'X{i}' for i in range(d)]
    df = pd.DataFrame(X, columns=columns)
    df['Y'] = Y

    # Save to CSV if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Dataset saved to {save_path}")
    
    return df


random_state = 123
path = 'your_path'

# Code to generate synthrtic datasets
small_dataset = 150
low_dimensionality = 5


medium_dataset = 1500
medium_dimensionality = 25

large_dataset = 15000
large_dimensionality = 125

### Small Dataset
#production of small dataset with small number of features:
for i in range(0,5):
    p = 0.1 + i*0.2
    print(p)
    for j in [0.05, 0.1, 0.4, 0.5, 0.6, 0.9, 0.95]:
        g = j
        print(g)
        path = os.path.join(path, f'/s&s/p_{p}_and_g_{g}"')
        generate_synthetic_data(p, small_dataset, medium_dimensionality, g, path, random_state)

#production of small dataset with medium number of features:
for i in range(0,5):
    p = 0.1 + i*0.2
    print(p)
    for j in [0.05, 0.1, 0.4, 0.5, 0.6, 0.9, 0.95]:
        g = j
        print(g)
        path = os.path.join(path, f'/s&m/p_{p}_and_g_{g}"')
        generate_synthetic_data(p, small_dataset, medium_dimensionality, g, path, random_state)

#production of small dataset with large number of features:
for i in range(0,5):
    p = 0.1 + i*0.2
    print(p)
    for j in [0.05, 0.1, 0.4, 0.5, 0.6, 0.9, 0.95]:
        g = j
        print(g)
        path = os.path.join(path, f'/s&l/p_{p}_and_g_{g}"')
        generate_synthetic_data(p, small_dataset, large_dimensionality, g, path, random_state)

### Medium Datasets
#production of medium dataset with small number of features:
for i in range(0,5):
    p = 0.1 + i*0.2
    print(p)
    for j in [0.05, 0.1, 0.4, 0.5, 0.6, 0.9, 0.95]:
        g = j
        print(g)
        path = os.path.join(path, f'/m&s/p_{p}_and_g_{g}"')
        generate_synthetic_data(p, medium_dataset, low_dimensionality, g, path, random_state)

#production of medium dataset with medium number of features:
for i in range(0,5):
    p = 0.1 + i*0.2
    print(p)
    for j in [0.05, 0.1, 0.4, 0.5, 0.6, 0.9, 0.95]:
        g = j
        print(g)
        path = os.path.join(path, f'/m&m/p_{p}_and_g_{g}"')
        generate_synthetic_data(p, medium_dataset, medium_dimensionality, g, path, random_state)

#production of medium dataset with large number of features:
for i in range(0,5):
    p = 0.1 + i*0.2
    print(p)
    for j in [0.05, 0.1, 0.4, 0.5, 0.6, 0.9, 0.95]:
        g = j
        print(g)
        path = os.path.join(path, f'/m&l/p_{p}_and_g_{g}"')
        generate_synthetic_data(p, medium_dataset, large_dimensionality, g, path, random_state)


### Large Datasets
#production of large dataset with small number of features:
for i in range(0,5):
    p = 0.1 + i*0.2
    print(p)
    for j in [0.05, 0.1, 0.4, 0.5, 0.6, 0.9, 0.95]:
        g = j
        print(g)
        path = os.path.join(path, f'/l&s/p_{p}_and_g_{g}"')
        generate_synthetic_data(p, large_dataset, low_dimensionality, g, path, random_state)

#production of large dataset with medium number of features:
for i in range(0,5):
    p = 0.1 + i*0.2
    print(p)
    for j in [0.05, 0.1, 0.4, 0.5, 0.6, 0.9, 0.95]:
        g = j
        print(g)
        path = os.path.join(path, f'/l&m/p_{p}_and_g_{g}"')
        generate_synthetic_data(p, large_dataset, medium_dimensionality, g, path, random_state)

#production of large dataset with large number of features:
for i in range(0,5):
    p = 0.1 + i*0.2
    print(p)
    for j in [0.05, 0.1, 0.4, 0.5, 0.6, 0.9, 0.95]:
        g = j
        print(g)
        path = os.path.join(path, f'/l&l/p_{p}_and_g_{g}"')
        generate_synthetic_data(p, large_dataset, large_dimensionality, g, path, random_state)