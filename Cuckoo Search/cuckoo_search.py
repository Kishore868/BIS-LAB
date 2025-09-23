import numpy as np

# ------------------------------
# Neural Network Functions
# ------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_pass(weights, X):
    """
    Forward pass for 2-2-1 NN:
    weights = [w1,w2,w3,w4,b1,b2,w5,w6,b3]
    """
    # Unpack weights
    w_input_hidden = weights[:4].reshape(2,2)   # w1-w4
    b_hidden = weights[4:6]                     # b1, b2
    w_hidden_output = weights[6:8].reshape(2,1) # w5,w6
    b_output = weights[8]                       # b3
    
    # Hidden layer
    hidden = sigmoid(np.dot(X, w_input_hidden) + b_hidden)
    
    # Output layer
    output = sigmoid(np.dot(hidden, w_hidden_output) + b_output)
    return output

def fitness(weights, X, y):
    """Mean Squared Error as fitness"""
    y_pred = forward_pass(weights, X)
    return np.mean((y - y_pred)**2)

# ------------------------------
# Cuckoo Search Algorithm
# ------------------------------
def levy_flight(Lambda):
    u = np.random.normal(0,1)
    v = np.random.normal(0,1)
    return u / (abs(v)**(1/Lambda))

def cuckoo_search(n=20, d=9, pa=0.25, alpha=0.01, beta=1.5, max_iter=500, X=None, y=None):
    # Initialize nests
    nests = np.random.uniform(-1,1,(n,d))
    fitness_values = np.array([fitness(w, X, y) for w in nests])
    
    best_idx = np.argmin(fitness_values)
    best = nests[best_idx].copy()
    best_score = fitness_values[best_idx]
    
    for t in range(max_iter):
        for i in range(n):
            step = alpha * levy_flight(beta)
            new_sol = nests[i] + step * (nests[i] - best)
            f_new = fitness(new_sol, X, y)
            if f_new < fitness_values[i]:
                nests[i] = new_sol
                fitness_values[i] = f_new
                if f_new < best_score:
                    best = new_sol.copy()
                    best_score = f_new
        # Abandon worst nests
        abandon = np.random.rand(n,d) < pa
        nests = nests + abandon * np.random.uniform(-1,1,(n,d))
        fitness_values = np.array([fitness(w, X, y) for w in nests])
    return best, best_score

# ------------------------------
# Example: XOR Problem
# ------------------------------
if __name__ == "__main__":
    # XOR dataset
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])
    
    best_weights, best_error = cuckoo_search(X=X, y=y)
    
    print("Best Weights:", best_weights)
    print("Best Error:", best_error)
    
    preds = forward_pass(best_weights, X)
    print("Predictions:", preds.round(3))
