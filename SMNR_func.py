import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats
import sklearn.linear_model
from scipy.special import binom
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from scipy.special import binom
from scipy.stats import hypergeom
from scipy.special import gammaln
from sklearn.utils.extmath import randomized_svd
import warnings
import time
import networkx as nx
from collections import defaultdict
from scipy.integrate import quad
from itertools import product
import networkx as nx

def mat2vec_ix(mat, directed, selfloops):
    """
    Returns indices to vectorize adjacency matrices, removing unused entries.
    """
    n = mat.shape[0]
    if mat.shape[0] == mat.shape[1]:
        if not directed:
            # For undirected graphs, take upper triangle
            mask = np.zeros_like(mat, dtype=bool)
            triu_ix = np.triu_indices(n, k=1 if not selfloops else 0)
            mask[triu_ix] = True
            return mask
        else:
            # For directed graphs, remove diagonal if no selfloops
            mask = np.ones_like(mat, dtype=bool)
            if not selfloops:
                np.fill_diagonal(mask, False)
            return mask
    else:
        # For bipartite graphs
        return np.ones_like(mat, dtype=bool)

def vec2mat(vec, directed, selfloops, n):
    """
    Generates adjacency matrix from vector.
    """
    if isinstance(n, tuple) and len(n) == 2:
        # Bipartite graph
        mat = np.zeros(n)
    else:
        # Unipartite graph
        mat = np.zeros((n, n))
    
    idx = mat2vec_ix(mat, directed, selfloops)
    mat[idx] = vec
    return mat

def compute_xi(A, directed, selfloops, regular=False):
    """
    Computes combinatorial matrix according to soft configuration model.
    """
    n = A.shape[0]  # define the node size
    
    if not directed and not np.allclose(A, A.T):
        warnings.warn("Asymmetric adjacency matrix for undirected graph. Symmetrizing...")
        A = A + A.T
    
    if regular:
        Kin = A.sum(axis=0)
        ix = mat2vec_ix(A, directed, selfloops)
        m = A[ix].sum()
        M = m**2
        if not directed:
            M *= 4
        n_edges = ix.sum()
        xiregular = np.full_like(A, M / n_edges, dtype=float)
        if not selfloops and directed:
            np.fill_diagonal(xiregular, 0)
        return np.ceil(xiregular)
    else:
        if not selfloops:
            np.fill_diagonal(A, 0)
        Kin = A.sum(axis=0)  # In-degrees
        Kout = A.sum(axis=1)  # Out-degrees
        xi = np.outer(Kout, Kin)
        
        if A.shape[0] == A.shape[1]:
            if not selfloops and directed:
                diagxi = np.diag(xi)
                vbas = np.floor(diagxi / (n - 1))
                diagxir = diagxi - vbas * (n - 1)
                np.fill_diagonal(xi, 0)
                # Redistribute diagonal values
                for i in range(n):
                    # Create a vector with zeros except for the current node
                    v = np.zeros(n)
                    # Indices excluding i
                    idx = np.concatenate([np.arange(i), np.arange(i+1, n)])
                    # Randomly distribute diagxir[i]
                    if diagxir[i] > 0:
                        chosen = np.random.choice(idx, size=int(diagxir[i]), replace=False)
                        v[chosen] = 1
                    v += vbas[i]
                    xi[i, :] += v
                    xi[:, i] += v
            else:
                if not directed:
                    xi = xi + xi.T - np.diag(np.diag(xi))
                    if not selfloops:
                        sdiag = np.diag(xi).sum()
                        Kin_total = Kin.sum()
                        toadd = np.ceil((sdiag / Kin_total) * Kin / (n - 1))
                        for i in range(n):
                            xi[i, :] += toadd[i]
                            xi[:, i] += toadd[i]
                        np.fill_diagonal(xi, 0)
        return xi

def soft_threshold(x, threshold):
    """
    Apply soft thresholding function to a matrix.
    
    Parameters:
    x : numpy.ndarray
        Input matrix to apply thresholding
    threshold : float
        Threshold parameter
        
    Returns:
    numpy.ndarray
        Thresholded matrix with the same dimensions as A
    """
    # Create a copy of the input matrix to avoid modifying original data
    s = np.copy(x)
    
    # Identify elements with absolute value less than or equal to threshold
    # and set them to zero
    abs_mask = np.abs(s) <= threshold
    s[abs_mask] = 0.0
    
    # For elements greater than threshold, subtract the threshold value
    pos_mask = s > threshold
    s[pos_mask] -= threshold
    
    # For elements less than negative threshold, add the threshold value
    neg_mask = s < -threshold
    s[neg_mask] += threshold
    
    return s

def compute_odds_matrices(R_list, theta):
    """
    Computes the combined odds matrix from multiple relation layers.
    """
    odds = np.ones_like(R_list[0])
    for i, R in enumerate(R_list):
        # Avoid negative exponents causing NaNs
        exp_val = np.clip(theta[i], -10, 10)
        # Avoid negative values
        R_safe = np.maximum(R, 1e-5)
        odds *= R_safe ** exp_val
    return odds

def logl_multinomial(adj, xi, omega, directed, selfloops):
    """
    Computes approximated log-likelihood using multinomial distribution.
    """
    ix = mat2vec_ix(adj, directed, selfloops)
    pp = np.sum(xi[ix] * omega[ix])
    p = (xi[ix] * omega[ix]) / pp
    counts = adj[ix].astype(int)
    
    # Multinomial log-likelihood
    total = counts.sum()
    log_fact_total = gammaln(total + 1)
    log_fact_counts = gammaln(counts + 1).sum()
    log_probs = counts * np.log(p)
    log_lik = log_fact_total - log_fact_counts + np.sum(log_probs)
    return log_lik

def compute_hessian(theta, R_list, xi, A, rho, directed, selfloops):
    """
    Computes Hessian matrix of the augmented Lagrangian at given theta.
    """
    n = len(theta)  # Number of parameters
    ix = mat2vec_ix(A, directed, selfloops)
    A_flat = A[ix]
    xi_flat = xi[ix]
    m = np.sum(A_flat)  # Total number of edges
    
    # Compute product of all relation layers raised to theta
    product = np.ones(len(A_flat))
    for i, R in enumerate(R_list):
        R_safe = np.maximum(R[ix], 1e-5)
        product *= R_safe ** theta[i]
    
    # Compute denominator S
    S = np.sum(xi_flat * product)
    
    # Precompute A_arr and B_mat
    A_arr = np.zeros(n)
    B_mat = np.zeros((n, n))
    
    # Compute A_arr and B_mat
    for l in range(n):
        logR_l = np.log(np.maximum(R_list[l][ix], 1e-5))
        A_arr[l] = np.sum(logR_l * xi_flat * product)
        
        for k in range(n):
            logR_k = np.log(np.maximum(R_list[k][ix], 1e-5))
            if k >= l:  # Only compute upper triangle
                B_mat[l, k] = np.sum(logR_l * logR_k * xi_flat * product)
            else:
                B_mat[l, k] = B_mat[k, l]  # Symmetric
    
    # Compute Hessian matrix
    H = np.zeros((n, n))
    for l in range(n):
        for k in range(n):
            if l == k:
                H[l, k] = (m * B_mat[l, k] / S) - (m * A_arr[l]**2 / S**2) + rho
            else:
                H[l, k] = (m * B_mat[l, k] / S) - (m * A_arr[l] * A_arr[k] / S**2)
    
    return H

def fnM(theta, R_list, xi, A, sigma, rho, z, directed, selfloops):
    """
    Computes partial derivatives for ADMM optimization.
    """
    ix = mat2vec_ix(A, directed, selfloops)
    A_flat = A[ix]
    xi_flat = xi[ix]
    
    # Compute product of all relation layers raised to theta
    product = np.ones(len(A_flat))
    for i, R in enumerate(R_list):
        R_safe = np.maximum(R[ix], 1e-5)  # Ensure positive values
        product *= R_safe ** theta[i]
    
    # Compute the gradient components
    grad = np.zeros(len(theta))
    for i in range(len(theta)):
        logR = np.log(np.maximum(R_list[i][ix], 1e-5))
        
        # Compute numerator and denominator for term1
        numerator = np.sum(logR * xi_flat * product)
        denominator = np.sum(xi_flat * product)
        term1 = numerator / denominator if denominator != 0 else 0
        
        term2 = np.sum(A_flat * logR) / np.sum(A_flat) if np.sum(A_flat) != 0 else 0
        
        term3 = (sigma[i] + rho * (theta[i] - z[i])) / np.sum(A_flat)
        
        grad[i] = term1 - term2 + term3
    
    return grad

def compute_ApproxObj(theta, R_list, xi, A, lambd, directed, selfloops):
    """
    Computes the approximate objective function value.
    """
    ix = mat2vec_ix(A, directed, selfloops)
    A_flat = A[ix]
    xi_flat = xi[ix]
    
    # Compute product of all relation layers raised to theta
    product = np.ones(len(A_flat))
    for i, R in enumerate(R_list):
        R_safe = np.maximum(R[ix], 1e-5)
        product *= R_safe ** theta[i]
    
    # Compute the probabilities
    p = xi_flat * product
    p_norm = p / p.sum()
    
    # Compute the log-likelihood part
    log_lik = np.sum(A_flat * np.log(p_norm))
    
    # Add L1 penalty
    obj = -log_lik + lambd * np.sum(np.abs(theta))
    return obj

def get_zero_dummy(dat, name=None, zero_values=None):
    """
    Converts adjacency matrix to handle zero values in relation layers.
    Returns two matrices:
        1. Original matrix with zeros replaced by 1
        2. Indicator matrix with 1 for non-zeros and natural constant for zeros
    
    Parameters:
    dat : ndarray
        Original adjacency matrix
    name : str, optional
        Base name for output matrices (not used in Python version)
    zero_values : float, optional
        Value to replace zeros (default: natural constant e)
    
    Returns:
    list: [adjusted_matrix, zero_dummy_matrix]
    """
    if zero_values is None:
        zero_values = np.exp(1)  # Natural constant e
    
    # Create adjusted matrix: replace 0 with 1
    adjusted_dat = np.where(dat == 0, 1.0, dat)
    
    # Create zero dummy matrix: 1 for non-zeros, zero_values for zeros
    zero_dummy = np.where(dat == 0, zero_values, 1.0)
    
    return [adjusted_dat, zero_dummy]

def SMNR_predict(train_adj, test_R_list, est_coef, directed=False, selfloops=False, method="exact"):
    """
    Predicts test set interaction network adjacency matrix using trained SMNR model.
    
    Parameters:
    train_adj : ndarray
        Training set interaction adjacency matrix
    test_R_list : list of ndarray
        List of test set relation matrices
    est_coef : ndarray
        Estimated coefficients (theta) from SMNR model
    directed : bool
        Whether the graph is directed
    selfloops : bool
        Whether self-loops are allowed
    method : str
        Prediction method ("exact" or "approximate")
    
    Returns:
    ndarray: Predicted test set adjacency matrix
    """
    # Calculate training set properties
    n_train = train_adj.shape[0]
    ix_train = mat2vec_ix(train_adj, directed, selfloops)
    m_train = np.sum(train_adj[ix_train])
    
    # Calculate number of possible edges in training set
    if directed:
        n_possible_train = n_train * n_train
        if not selfloops:
            n_possible_train -= n_train
    else:
        n_possible_train = n_train * (n_train - 1) // 2
        if selfloops:
            n_possible_train += n_train
    
    # Calculate edge density in training set
    edge_density = m_train / n_possible_train
    
    # Get test set properties
    n_test = test_R_list[0].shape[0]
    
    # Calculate number of possible edges in test set
    if directed:
        n_possible_test = n_test * n_test
        if not selfloops:
            n_possible_test -= n_test
    else:
        n_possible_test = n_test * (n_test - 1) // 2
        if selfloops:
            n_possible_test += n_test
    
    # Estimate total edges in test set
    m_test_est = edge_density * n_possible_test
    
    # Create combinatorial matrix for test set (uniform assumption)
    xi_test = np.full((n_test, n_test), m_test_est**2 / n_possible_test)
    
    # Calculate odds matrix for test set
    odds_mat = np.ones((n_test, n_test))
    for i, R in enumerate(test_R_list):
        odds_mat *= np.maximum(R, 1e-5) ** est_coef[i]
    
    # Get indices for valid edges in test set
    ix_test = mat2vec_ix(xi_test, directed, selfloops)
    
    if method == "exact":
        # Wallenius expectation method
        total_odds = np.sum(odds_mat[ix_test] * xi_test[ix_test])
        pred_adj = np.zeros((n_test, n_test))
        
        # Precompute indices for efficiency
        rows, cols = np.where(ix_test)
        for i in range(len(rows)):
            r, c = rows[i], cols[i]
            if total_odds > 0:
                pred_adj[r, c] = m_test_est * (odds_mat[r, c] * xi_test[r, c]) / total_odds
    else:
        # Approximate method (multivariate hypergeometric approximation)
        total_odds = np.sum(odds_mat[ix_test] * xi_test[ix_test])
        prob_mat = np.zeros((n_test, n_test))
        prob_mat[ix_test] = (odds_mat[ix_test] * xi_test[ix_test]) / total_odds
        pred_adj = m_test_est * prob_mat
    
    # Set invalid positions to zero
    pred_adj[~ix_test] = 0
    
    return pred_adj

def SMNR(R_list, A, xi=None, lambd=100, rho=100, tolerance=1e-4, 
          max_iter=10000, directed=False, selfloops=False, regular=False, 
          verbose=False, alpha=0.05, initial_theta=None):
    """
    Sparse Fused Multiplex Network Regression (SMNR)
    
    Parameters:
    R_list : list of ndarray
        List of relation matrices (covariates)
    A : ndarray
        Adjacency matrix of the interaction network
    xi : ndarray, optional
        Combinatorial matrix. If None, will be computed.
    lambd : float
        Regularization parameter for L1 penalty
    rho : float
        Augmented Lagrangian parameter
    tolerance : float
        Convergence tolerance
    max_iter : int
        Maximum number of iterations
    directed : bool
        Whether the graph is directed
    selfloops : bool
        Whether self-loops are allowed
    regular : bool
        Whether to use regular configuration model
    verbose : bool
        Whether to print progress
    alpha : float
        Significance level for confidence intervals (default 0.05)
    initial_theta : ndarray, optional
        Initial values for theta parameters
    
    Returns:
    dict with results:
        'theta': estimated parameters
        'std_errors': standard errors of parameters
        'p_values': p-values for parameters
        'conf_intervals': confidence intervals for parameters
        'cox_snell_R2': Cox Snell R-squared
        'omega': propensity matrix
        'loglikelihood': log-likelihood
        'AIC': Akaike Information Criterion
        'R2': McFadden R-squared
        'history': optimization history
    """
    # Initialize combinatorial matrix if not provided
    if xi is None:
        xi = compute_xi(A, directed, selfloops, regular)
    
    n = A.shape[0]
    ix = mat2vec_ix(A, directed, selfloops)
    m = np.sum(A[ix])
    
    # Initialize variables
    if initial_theta is not None:
        theta = np.array(initial_theta, dtype=float)
    else:
        theta = np.ones(len(R_list)) * 0.01  # Default initial parameter values
    z = theta.copy()
    sigma = np.ones(len(R_list)) * 100  # Dual variables
    
    # Optimization history
    history = {
        'theta': [theta.copy()],
        'z': [z.copy()],
        'sigma': [sigma.copy()],
        'obj': [compute_ApproxObj(theta, R_list, xi, A, lambd, directed, selfloops)]
    }
    
    # ADMM optimization
    for iter in range(max_iter):
        # Store previous values for convergence check
        theta_prev = theta.copy()
        
        # Update theta using gradient descent
        grad = fnM(theta, R_list, xi, A, sigma, rho, z, directed, selfloops)
        step_size = 0.5 / np.sqrt(iter + 1)  # Adaptive step size
        theta = theta - step_size * grad
        
        # Update z using soft thresholding
        z = soft_threshold(theta + sigma / rho, lambd / rho)
        
        # Update sigma
        sigma = sigma + rho * (theta - z)
        
        # Compute objective function value
        obj_val = compute_ApproxObj(z, R_list, xi, A, lambd, directed, selfloops)
        
        # Record history
        history['theta'].append(theta.copy())
        history['z'].append(z.copy())
        history['sigma'].append(sigma.copy())
        history['obj'].append(obj_val)
        
        # Check convergence
        theta_diff = np.linalg.norm(theta - theta_prev)
        if verbose:
            print(f"Iter {iter+1}: Obj={obj_val:.4f}, Diff={theta_diff:.6f}")
        
        if theta_diff < tolerance:
            if verbose:
                print(f"Converged at iteration {iter+1}")
            break
    
    # Compute final propensity matrix
    omega = compute_odds_matrices(R_list, z)
    
    # Compute log-likelihood
    loglik = logl(A, xi, omega, directed, selfloops, method='auto')
    
    # Compute degrees of freedom
    if regular:
        df = 2 * len(z) + 3
    else:
        if directed:
            df = 2 * len(z) + 2 * n + 2
        else:
            df = 2 * len(z) + n + 2
    
    # Compute AIC
    AIC = 2 * df - 2 * loglik
    
    # Compute R-squared (McFadden pseudo R-squared)
    # Null model: no covariates, only combinatorial effects
    R0_list = [np.ones_like(R_list[0])]  # Single all-ones relation
    theta0 = np.array([0.0])  # Single parameter for null model
    omega0 = compute_odds_matrices(R0_list, theta0)
    loglik_null = logl(A, xi, omega0, directed, selfloops)
    R2 = 1 - ((loglik - df) / loglik_null)
    
    # Compute Cox Snell R-squared
    cox_snell_R2 = 1 - np.exp((2 / m) * (loglik_null - loglik))
    
    # Compute Hessian matrix and standard errors
    H = compute_hessian(z, R_list, xi, A, rho, directed, selfloops)
    try:
        H_inv = np.linalg.inv(H)
        variances = np.diag(H_inv)
        std_errors = np.sqrt(variances)
        
        # Compute z-scores and p-values
        z_scores = z / std_errors
        p_values = 2 * (1 - scipy.stats.norm.cdf(np.abs(z_scores)))
        
        # Compute confidence intervals
        z_alpha = scipy.stats.norm.ppf(1 - alpha/2)
        conf_intervals = [
            [z[i] - z_alpha * std_errors[i], z[i] + z_alpha * std_errors[i]]
            for i in range(len(z))
        ]
    except np.linalg.LinAlgError:
        warnings.warn("Hessian matrix is singular. Cannot compute standard errors.")
        std_errors = np.full_like(z, np.nan)
        p_values = np.full_like(z, np.nan)
        conf_intervals = [[np.nan, np.nan]] * len(z)
    
    # Prepare results
    results = {
        'theta': theta,
        'z': z,
        'sigma': sigma,
        'std_errors': std_errors,
        'p_values': p_values,
        'conf_intervals': conf_intervals,
        'cox_snell_R2': cox_snell_R2,
        'omega': omega,
        'loglikelihood': loglik,
        'AIC': AIC,
        'R2': R2,
        'history': history,
        'n_iter': iter + 1,
        'converged': theta_diff < tolerance,
        'hessian': H
    }
    
    return results

def generate_multiplex_data(num_nodes=20, num_signals=3, num_noises=2, 
                            signal_strength=0.8, noise_strength=0.2,
                            edge_factor=10, seed=None, 
                            use_multiplex_threshold=50,
                            theta=None,
                            signal_structure='block',  # New parameter: specify signal layer structure
                            small_world_k=4,           # Small world parameter: number of neighbors
                            small_world_p=0.1,         # Small world parameter: rewiring probability
                            scale_free_m=2):           # Scale-free parameter: edges to add per node
    """
    Generates synthetic multiplex network data with various signal layer structures.
    
    Parameters:
    num_nodes : int
        Number of nodes in the network
    num_signals : int
        Number of signal relation layers
    num_noises : int
        Number of noise relation layers
    signal_strength : float
        Strength of signal relations
    noise_strength : float
        Strength of noise relations
    edge_factor : float
        Factor to determine number of edges (m = edge_factor * num_nodes)
    seed : int, optional
        Random seed for reproducibility
    use_multinomial_threshold : int
        Use multinomial approximation when num_nodes >= this threshold
    theta : list, optional
        Predefined theta values. If provided, overrides num_signals and num_noises.
        Theta length must be even (each base layer has two parameters)
    signal_structure : str or list
        'block' - Block community structure
        'small-world' - Small-world structure
        'scale-free' - Scale-free structure
        If list, length should match num_signals to specify structure per signal layer
    small_world_k : int
        Initial number of neighbors per node in small-world network
    small_world_p : float
        Rewiring probability (0-1) for small-world network
    scale_free_m : int
        Number of edges to add per new node in scale-free network

    Returns:
    dict with:
        'A': interaction network adjacency matrix
        'R_list': list of relation matrices (after zero-dummy transformation)
        'true_theta': true parameter values (expanded for zero-dummy matrices)
        'xi': combinatorial matrix
        'signal_mask': boolean mask indicating signal layers in R_list
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Process signal_structure parameter
    if isinstance(signal_structure, str):
        signal_structures = [signal_structure] * num_signals
    else:
        if len(signal_structure) != num_signals:
            raise ValueError("signal_structure list length must match num_signals")
        signal_structures = signal_structure
    
    # Use theta if provided, otherwise use num_signals and num_noises
    if theta is not None:
        if len(theta) % 2 != 0:
            raise ValueError("Theta length must be even (each base layer has two parameters)")
        num_base_layers = len(theta) // 2
        true_theta = np.array(theta)
        # Identify signal layers: where theta != 0 (excluding zero dummy layers)
        signal_mask = np.zeros(len(theta), dtype=bool)
        for i in range(num_base_layers):
            if theta[2*i] != 0:  # Only original matrix parameters can be non-zero
                signal_mask[2*i] = True
    else:
        num_base_layers = num_signals + num_noises
        true_theta = []
        signal_mask = []
        for i in range(num_signals):
            true_theta.append(signal_strength)  # Original signal matrix
            true_theta.append(0.0)              # Zero dummy matrix
            signal_mask.append(True)
            signal_mask.append(False)
        for i in range(num_noises):
            true_theta.append(0.0)              # Original noise matrix
            true_theta.append(0.0)              # Zero dummy matrix
            signal_mask.append(False)
            signal_mask.append(False)
        signal_mask = np.array(signal_mask)
    
    # Generate base relationship matrices
    base_R_list = []
    for i in range(num_base_layers):
        # Check if current layer is a signal layer
        is_signal = (theta is not None and true_theta[2*i] != 0) or (theta is None and i < num_signals)
        
        if is_signal:
            # Signal layer
            strength = signal_strength if theta is None else true_theta[2*i]
            struct_type = signal_structures[i] if theta is None else signal_structures[np.sum(signal_mask[:2*i])//2]
            
            if struct_type == 'block':
                # Block structure: create community structure
                mat = np.full((num_nodes, num_nodes), strength)
                
                # Add block structure (2-4 communities)
                n_blocks = np.random.randint(2, 5)
                block_size = num_nodes // n_blocks
                block_assignment = np.repeat(np.arange(n_blocks), block_size)
                if len(block_assignment) < num_nodes:
                    block_assignment = np.concatenate([
                        block_assignment, 
                        np.ones(num_nodes - len(block_assignment)) * (n_blocks - 1)
                    ])
                
                # Strengthen intra-community connections
                for i_idx in range(num_nodes):
                    for j_idx in range(num_nodes):
                        if i_idx < j_idx and block_assignment[i_idx] == block_assignment[j_idx]:
                            mat[i_idx, j_idx] = strength * np.random.uniform(1.5, 2.5)
            
            elif struct_type == 'small-world':
                # Small-world structure (Watts-Strogatz model)
                k = small_world_k  # Number of neighbors per node
                p = small_world_p  # Rewiring probability
                
                # Create small-world network
                G = nx.watts_strogatz_graph(num_nodes, k, p)
                mat = nx.to_numpy_array(G)
                
                # Set edge weights to signal strength
                mat = mat.astype(float) * strength
                
                # Randomly strengthen some edges (10%)
                n_edges = np.sum(mat > 0) // 2
                n_enhance = max(1, int(0.1 * n_edges))
                edge_indices = np.argwhere(np.triu(mat) > 0)
                enhance_indices = edge_indices[np.random.choice(len(edge_indices), n_enhance, replace=False)]
                
                for idx in enhance_indices:
                    i, j = idx
                    mat[i, j] = mat[j, i] = strength * np.random.uniform(1.5, 2.5)
            
            elif struct_type == 'scale-free':
                # Scale-free structure (Barabási-Albert model)
                m = scale_free_m  # Edges to add per new node
                
                # Create scale-free network
                G = nx.barabasi_albert_graph(num_nodes, m)
                mat = nx.to_numpy_array(G)
                
                # Set edge weights to signal strength
                mat = mat.astype(float) * strength
                
                # Strengthen hub connections (top 10% nodes by degree)
                degrees = dict(G.degree())
                sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
                hub_nodes = [node for node, deg in sorted_nodes[:max(1, num_nodes//10)]]
                
                for node in hub_nodes:
                    neighbors = list(G.neighbors(node))
                    for neighbor in neighbors:
                        mat[node, neighbor] = mat[neighbor, node] = strength * np.random.uniform(1.5, 2.5)
            
            else:
                raise ValueError(f"Unknown signal structure: {struct_type}")
                
            # Symmetrize and remove self-loops
            mat = np.triu(mat) + np.triu(mat, 1).T
            np.fill_diagonal(mat, 0)
        
        else:
            # Noise layer: uniform random values
            strength = noise_strength if theta is None else true_theta[2*i]
            mat = np.random.uniform(0, strength, (num_nodes, num_nodes))
            mat = np.triu(mat) + np.triu(mat, 1).T
            np.fill_diagonal(mat, 0)
        
        base_R_list.append(mat)
    
    # Apply zero dummy transformation to each relationship matrix
    R_list = []
    for mat in base_R_list:
        # Get zero dummy matrix
        adj_mat, zero_dummy = get_zero_dummy(mat)
        
        # Add to relationship list
        R_list.append(adj_mat)
        R_list.append(zero_dummy)
    
    # If theta is provided, we already have signal_mask
    if theta is None:
        signal_mask = np.array(signal_mask)
    
    # Compute combined advantage matrix using true_theta
    odds = np.ones((num_nodes, num_nodes))
    for i, mat in enumerate(R_list):
        odds *= mat ** true_theta[i]
    
    # Calculate expected number of edges
    m = int(edge_factor * num_nodes)
    
    # Create combination matrix (regular configuration)
    xi = np.full((num_nodes, num_nodes), m**2 / (num_nodes * (num_nodes - 1)))
    np.fill_diagonal(xi, 0)  # No self-loops
    
    # Generate interaction network
    ix = mat2vec_ix(xi, directed=False, selfloops=False)
    
    # Flatten relevant arrays
    xi_flat = xi[ix]
    odds_flat = odds[ix]
    
    # Convert xi to integers for Wallenius distribution
    xi_flat_int = np.round(xi_flat).astype(int)
    
    # Ensure non-negative values
    xi_flat_int = np.maximum(xi_flat_int, 0)
    odds_flat = np.maximum(odds_flat, 1e-10)
    
    # Total number of edges to sample
    n = m
    
    # Use Wallenius for small networks, multinomial for large networks
    if num_nodes < use_multiplex_threshold:
        try:
            # Sample from Wallenius distribution
            sample = rMWNCHypergeo(1, xi_flat_int, n, odds_flat)[0]
        except Exception as e:
            warnings.warn(f"Wallenius sampling failed: {str(e)}, using multinomial instead")
            p = xi_flat * odds_flat
            p = p / p.sum()
            sample = np.random.multinomial(n, p)
    else:
        # Use multinomial approximation for large networks
        p = xi_flat * odds_flat
        p = p / p.sum()
        sample = np.random.multinomial(n, p)
    
    # Create adjacency matrix
    A = np.zeros_like(xi)
    A[ix] = sample
    
    return {
        'A': A,
        'R_list': R_list,
        'true_theta': true_theta,
        'xi': xi,
        'signal_mask': signal_mask
    }

def cross_validate_SMNR(R_list, A, xi, lambd_values, rho_values, tolerance=1e-4,
                         directed=False, selfloops=False, n_folds=5, 
                         seed=None, initial_theta=None, apply_zero_dummy=True):
    """
    Performs grid search cross-validation to select the best lambda and rho for SMNR.
    
    Parameters:
    R_list : list of ndarray
        Relation matrices
    A : ndarray
        Interaction adjacency matrix
    xi : ndarray
        Combinatorial matrix
    lambd_values : list of float
        Lambda values to test
    rho_values : list of float
        Rho values to test
    n_folds : int
        Number of cross-validation folds
    seed : int, optional
        Random seed
    initial_theta : ndarray, optional
        Initial values for theta parameters
    apply_zero_dummy : bool
        Whether to apply zero-dummy transformation to relation matrices
    
    Returns:
    dict with:
        'best_lambda': best lambda value
        'best_rho': best rho value
        'best_val_mse': best validation MSE
        'results': dictionary of results for each (lambda, rho) pair
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Apply zero-dummy transformation if requested
    if apply_zero_dummy:
        new_R_list = []
        new_initial_theta = []
        
        # Apply transformation to each relation matrix
        for i, R in enumerate(R_list):
            adj_mat, zero_dummy = get_zero_dummy(R)
            new_R_list.append(adj_mat)
            new_R_list.append(zero_dummy)
            
            # Adjust initial theta if provided
            if initial_theta is not None:
                orig_theta = initial_theta[i]
                new_initial_theta.append(orig_theta)
                new_initial_theta.append(0.0)  # Zero-dummy matrix should have theta=0
        
        # If no initial theta provided, create default for new R_list
        if initial_theta is None:
            new_initial_theta = np.ones(len(new_R_list)) * 0.01
        else:
            new_initial_theta = np.array(new_initial_theta)
        
        R_list = new_R_list
        initial_theta = new_initial_theta
    
    # Prepare indices for cross-validation
    ix = mat2vec_ix(A, directed, selfloops)
    n_edges = A[ix].sum()
    edge_indices = np.where(ix)
    
    # Randomly permute edges
    perm = np.random.permutation(len(edge_indices[0]))
    fold_size = len(perm) // n_folds
    
    # Initialize results dictionary
    results = {}
    for lambd in lambd_values:
        for rho in rho_values:
            results[(lambd, rho)] = {
                'train_ll': [],
                'val_mse': []
            }
    
    best_lambda = None
    best_rho = None
    best_val_mse = float('inf')
    
    for fold in range(n_folds):
        # Create mask for validation set
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(perm)
        val_indices = perm[val_start:val_end]
        
        # Create training and validation masks
        train_ix = np.copy(ix)
        # Get indices for validation edges
        val_rows = edge_indices[0][val_indices]
        val_cols = edge_indices[1][val_indices]
        train_ix[val_rows, val_cols] = False
        
        val_ix = np.zeros_like(ix, dtype=bool)
        val_ix[val_rows, val_cols] = True
        
        # Create training and validation adjacency matrices
        A_train = np.copy(A)
        A_train[val_ix] = 0
        
        A_val = np.zeros_like(A)
        A_val[val_ix] = A[val_ix]
        
        # Compute xi for training set (or use provided xi)
        # In practice, we should recompute xi based on training set degrees
        # But for simplicity, we'll use the provided xi
        xi_train = xi
        
        # Train SMNR for each (lambda, rho) pair
        for lambd, rho in product(lambd_values, rho_values):
            try:
                # Train model
                model = SMNR(
                    R_list, A_train, xi=xi_train, lambd=lambd, 
                    rho=rho, tolerance=tolerance, directed=directed, 
                    selfloops=selfloops, initial_theta=initial_theta
                )
                
                # Compute training log-likelihood
                train_ll = model['loglikelihood']
                
                # Predict validation set using trained model
                pred_val = SMNR_predict(
                    A_train,  # Training adjacency
                    R_list,   # Same relations for validation
                    model['z'],  # Estimated coefficients
                    directed=directed,
                    selfloops=selfloops,
                    method="approximate"
                )
                
                # Calculate MSE on validation set
                val_mse = np.mean((pred_val[val_ix] - A_val[val_ix])**2)
                
                # Store results
                results[(lambd, rho)]['train_ll'].append(train_ll)
                results[(lambd, rho)]['val_mse'].append(val_mse)
                
                # Update best parameters if current MSE is better
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_lambda = lambd
                    best_rho = rho
                
            except Exception as e:
                print(f"Error for lambda={lambd}, rho={rho}, fold={fold}: {str(e)}")
                results[(lambd, rho)]['train_ll'].append(-np.inf)
                results[(lambd, rho)]['val_mse'].append(np.inf)
    
    # Compute average metrics for each parameter pair
    for params in results:
        results[params]['avg_train_ll'] = np.mean(results[params]['train_ll'])
        results[params]['avg_val_mse'] = np.mean(results[params]['val_mse'])
    
    return {
        'best_lambda': best_lambda,
        'best_rho': best_rho,
        'best_val_mse': best_val_mse,
        'results': results
    }

def dMWNCHypergeo(x, m, n, odds, precision=1e-7):
    """
    Probability mass function for Multivariate Wallenius' Noncentral Hypergeometric distribution
    
    Parameters:
    x : list[int] - Number of balls drawn of each color
    m : list[int] - Number of balls of each color in urn
    n : int       - Total number of balls drawn from urn
    odds : list[float] - Odds for each color
    precision : float  - Precision for numerical integration
    
    Returns:
    float - Probability of observing x
    """
    # Validate input dimensions
    k = len(m)
    if len(x) != k or len(odds) != k:
        raise ValueError("x, m, and odds must have the same length")
    if sum(x) != n:
        raise ValueError(f"Sum of x ({sum(x)}) must equal n ({n})")
    if any(xi < 0 for xi in x) or any(mi < 0 for mi in m):
        raise ValueError("Negative values not allowed")
    if n < 0 or n > sum(m):
        raise ValueError(f"n ({n}) must be between 0 and sum(m) ({sum(m)})")
    if any(wi < 0 for wi in odds):
        raise ValueError("Odds must be non-negative")
    
    # Calculate denominator D for the integral
    D = sum(odds[i] * (m[i] - x[i]) for i in range(k))
    if D <= 0:
        return 0.0  # Probability is zero when D is non-positive
    
    # Define the integrand function for numerical integration
    def integrand(t):
        product = 1.0
        for i in range(k):
            if odds[i] > 0 and m[i] > x[i]:
                exponent = odds[i] / D
                term = 1 - t**exponent
                product *= term**x[i]
        return product
    
    # Perform numerical integration using adaptive quadrature
    integral, _ = quad(integrand, 0, 1, epsabs=precision)
    return max(0.0, integral)  # Ensure non-negative probability

def rMWNCHypergeo(nran, m, n, odds, precision=1e-7):
    """
    Random variate generator for Multivariate Wallenius' Noncentral Hypergeometric distribution
    
    Parameters:
    nran : int      - Number of random variates to generate
    m : list[int]   - Number of balls of each color in urn
    n : int         - Total number of balls to draw
    odds : list[float] - Odds for each color
    precision : float  - Precision parameter (not used in this implementation)
    
    Returns:
    list[list[int]] - List of generated samples
    """
    k = len(m)  # Number of colors
    samples = []
    
    for _ in range(nran):
        # Initialize sample vector and remaining balls
        sample = [0] * k
        remaining = m.copy()
        to_draw = n
        
        # Sequential drawing process
        while to_draw > 0:
            # Calculate total remaining weight
            total_weight = sum(odds[i] * remaining[i] for i in range(k))
            if total_weight <= 0:
                break  # Stop if no valid draws remain
                
            # Calculate drawing probabilities for each color
            probs = [odds[i] * remaining[i] / total_weight for i in range(k)]
            
            # Draw one ball according to probabilities
            chosen_color = np.random.choice(k, p=probs)
            
            # Update counts
            sample[chosen_color] += 1
            remaining[chosen_color] -= 1
            to_draw -= 1
            
        samples.append(sample)
    
    return samples


def logl_wallenius(adj, xi, omega, directed, selfloops):
    """
    Computes log-likelihood using Wallenius' noncentral hypergeometric distribution.
    
    Parameters:
    adj : ndarray
        Adjacency matrix of the interaction network
    xi : ndarray
        Combinatorial matrix
    omega : ndarray
        Propensity matrix
    directed : bool
        Whether the graph is directed
    selfloops : bool
        Whether self-loops are allowed
        
    Returns:
    float: log-likelihood value
    """
    # Get vectorization indices
    ix = mat2vec_ix(adj, directed, selfloops)
    
    # Flatten the relevant arrays
    A_flat = adj[ix]
    xi_flat = xi[ix]
    omega_flat = omega[ix]
    
    # Total number of edges
    n = int(np.sum(A_flat))
    
    # Convert xi to integers for Wallenius distribution
    xi_flat_int = np.round(xi_flat).astype(int)
    
    # Ensure non-negative values
    xi_flat_int = np.maximum(xi_flat_int, 0)
    omega_flat = np.maximum(omega_flat, 1e-10)
    
    # Calculate probability using Wallenius distribution
    try:
        prob = dMWNCHypergeo(A_flat, xi_flat_int, n, omega_flat)
        return np.log(max(prob, 1e-300))  # Avoid log(0)
    except Exception as e:
        warnings.warn(f"Error computing Wallenius likelihood: {str(e)}")
        return -np.inf

def logl(adj, xi, omega, directed, selfloops, method='auto'):
    """
    Computes log-likelihood for ghype models.
    
    Parameters:
    adj : ndarray
        Adjacency matrix
    xi : ndarray
        Combinatorial matrix
    omega : ndarray
        Propensity matrix
    directed : bool
        Whether the graph is directed
    selfloops : bool
        Whether self-loops are allowed
    method : str
        'multinomial' for multinomial approximation,
        'wallenius' for Wallenius distribution,
        'auto' for automatic selection
        
    Returns:
    float: log-likelihood value
    """
    ix = mat2vec_ix(adj, directed, selfloops)
    n_edges = np.sum(adj[ix])
    
    # Automatic method selection
    if method == 'auto':
        if n_edges < 2000000:  # Use Wallenius for small networks
            return logl_wallenius(adj, xi, omega, directed, selfloops)
        else:  # Use multinomial for large networks
            return logl_multinomial(adj, xi, omega, directed, selfloops)
    
    # Manual method selection
    if method == 'wallenius':
        return logl_wallenius(adj, xi, omega, directed, selfloops)
    else:  # Default to multinomial
        return logl_multinomial(adj, xi, omega, directed, selfloops)

def check_graph_type(graph):
    """Check if the input is an adjacency matrix or an edge list"""
    if isinstance(graph, pd.DataFrame) and graph.shape[1] == 3:
        # Edge list validation
        if not (graph.iloc[:, 0].dtype in ['object', 'category'] and 
                graph.iloc[:, 1].dtype in ['object', 'category']):
            raise ValueError("The first two columns must be node IDs (string or category)")
        if not (graph.iloc[:, 2].dtype in ['int', 'float']):
            raise ValueError("The third column must be numeric (edge weight)")
        return False
    elif isinstance(graph, np.ndarray) and graph.ndim == 2:
        # Adjacency matrix validation
        if graph.shape[0] != graph.shape[1]:
            raise ValueError("Adjacency matrix must be a square matrix")
        return True
    else:
        raise TypeError("Input must be an adjacency matrix (numpy array) or an edge list (DataFrame)")

def reciprocity_stat(graph, nodes=None, zero_val=None):
    """Calculate reciprocity statistics"""
    is_matrix = check_graph_type(graph)
    
    if not is_matrix:
        # Convert edge list to adjacency matrix
        el = graph
        if nodes is None:
            nodes = sorted(set(el.iloc[:, 0]).union(set(el.iloc[:, 1])))
        adj = el2adj(el, nodes)
    else:
        adj = graph
        if nodes is None:
            nodes = np.arange(adj.shape[0])
    
    # Calculate reciprocal matrix (transpose)
    recip_mat = adj.T.copy()
    
    # Zero value processing
    if zero_val is not None:
        recip_mat[recip_mat == 0] = zero_val
    
    return recip_mat

def homophily_stat(variable, nodes, type='categorical', 
                   these_categories_only=None, zero_val=None):
    """Calculate homophily statistics"""
    var_array = np.array(variable)
    
    if type == 'categorical':
        # Categorical variable processing
        unique_vals = np.unique(var_array)
        prime_map = {val: i+1 for i, val in enumerate(unique_vals)}
        
        # Create block IDs
        block_ids = np.array([prime_map[val] for val in var_array])
        blocks = np.outer(block_ids, block_ids)
        
        # Create homophily matrix
        if these_categories_only is None:
            homophily_mat = np.where(
                blocks == np.diag(block_ids)**2, np.e, 1
            )
        else:
            selected_primes = [prime_map[val] for val in these_categories_only]
            homophily_mat = np.where(
                np.isin(blocks, [p**2 for p in selected_primes]), np.e, 1
            )
    
    elif type == 'absdiff':
        # Continuous variable processing (absolute difference)
        homophily_mat = np.zeros((len(nodes), len(nodes)))
        for i, val_i in enumerate(var_array):
            for j, val_j in enumerate(var_array):
                homophily_mat[i, j] = abs(val_i - val_j)
    else:
        raise ValueError("Type must be 'categorical' or 'absdiff'")
    
    # Zero value processing
    if zero_val is not None:
        homophily_mat[homophily_mat == 0] = zero_val
    
    return homophily_mat

# Helper functions ========================================================
def el2adj(edgelist, nodes):
    """Convert edge list to adjacency matrix"""
    node_index = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n))
    
    for _, row in edgelist.iterrows():
        src = node_index[row[0]]
        tgt = node_index[row[1]]
        weight = row[2]
        adj[src, tgt] = weight
    
    return adj

def adj2el(adj, directed):
    """Convert adjacency matrix to edge list"""
    rows, cols = np.where(adj > 0)
    data = adj[rows, cols]
    
    el = pd.DataFrame({
        'sender': rows,
        'target': cols,
        'weight': data
    })
    
    if not directed:
        # Remove duplicates for undirected graph
        el = el[el['sender'] <= el['target']]
    
    return el

# if __name__ == "__main__":
#     print("=== Sparse Fused Multiplex Network Regression (SMNR) Demo ===")
    
#     # Example 1: Using predefined theta to generate block structure in signal layers but not in noise layers
#     print("\n--- Example 1: Block data generation with custom theta ---")
#     custom_theta = [3.0, 0, 2.0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0]  # 6 base layers
    
#     data_block = generate_multiplex_data(
#         num_nodes=100,
#         theta=custom_theta,
#         edge_factor=10,
#         signal_structure='block',  # Block structure
#         seed=42
#     )
    
#     A = data_block['A']
#     R_list = data_block['R_list']
#     true_theta = data_block['true_theta']
#     xi = data_block['xi']
#     signal_mask = data_block['signal_mask']
    
#     print(f"Generated block data with {len(R_list)} relation layers")
#     print(f"Signal layers: {np.sum(signal_mask)}")
#     print(f"Noise layers: {np.sum(~signal_mask)}")
    
#     # Grid search for lambda and rho
#     print("\n--- Example 1: Grid search for lambda and rho ---")
#     lambd_values = np.exp(np.linspace(np.log(0.01), np.log(20), num=10))
#     rho_values = np.exp(np.linspace(np.log(0.01), np.log(20), num=10))
    
#     cv_results = cross_validate_SMNR(
#         R_list, A, xi, 
#         lambd_values=lambd_values,
#         rho_values=rho_values,
#         directed=False,
#         selfloops=False,
#         n_folds=3,
#         seed=42
#     )
    
#     best_lambda = cv_results['best_lambda']
#     best_rho = cv_results['best_rho']
#     best_mse = cv_results['best_val_mse']
    
#     print(f"Best lambda: {best_lambda}, Best rho: {best_rho}, Best MSE: {best_mse:.4f}")
    
#     # Train final model with best parameters
#     print("\nTraining final model with best parameters...")
#     model = SMNR(
#         R_list, A, xi=xi, lambd=best_lambda, rho=best_rho,
#         directed=False, selfloops=False, verbose=True
#     )
    
#     print("\nModel results:")
#     print(f"Converged: {model['converged']} in {model['n_iter']} iterations")
#     print(f"Log-likelihood: {model['loglikelihood']:.2f}")
#     print(f"AIC: {model['AIC']:.2f}")
#     print(f"McFadden R²: {model['R2']:.4f}")
#     print(f"Cox Snell R²: {model['cox_snell_R2']:.4f}")
    
#     # Compare estimated theta with true values
#     print("\nParameter comparison:")
#     for i, (est, true) in enumerate(zip(model['z'], true_theta)):
#         layer_type = "Signal" if signal_mask[i] else "Noise"
#         print(f"Layer {i+1} ({layer_type}): True={true:.2f}, Estimated={est:.4f}")

#     print("\nEstimated parameters with statistical significance:")
#     for i, (theta_est, std_err, p_val, ci) in enumerate(zip(
#         model['z'], model['std_errors'], model['p_values'], model['conf_intervals']
#     )):
#         layer_type = "Signal" if signal_mask[i] else "Noise"
#         sig_star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
#         print(f"  Layer {i+1} ({layer_type}):")
#         print(f"    theta = {theta_est:.4f} ± {std_err:.4f} {sig_star}")
#         print(f"    p-value = {p_val:.4f}")
#         print(f"    95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]")

#     # Analyze results
#     signal_est = model['z'][signal_mask]
#     noise_est = model['z'][~signal_mask]
#     true_signal = np.array(true_theta)[signal_mask]
#     true_noise = np.array(true_theta)[~signal_mask]
    
#     print("\n=== Performance Summary ===")
#     print(f"Signal layers: Mean absolute error = {np.mean(np.abs(signal_est - true_signal)):.4f}")
#     print(f"Noise layers: Mean absolute error = {np.mean(np.abs(noise_est - true_noise)):.4f}")
#     print(f"Regularization effectiveness: {np.sum(np.abs(noise_est) < 0.001)}/{len(noise_est)} noise parameters shrunk near zero")

    
#     # Example 2: Using predefined theta to generate small-world structure in signal layers but not in noise layers
#     print("\n--- Example 2: Small-world data generation with custom theta ---")
#     custom_theta = [3.0, 0, 2.0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0]  # 6 base layers
    
#     data_smallworld = generate_multiplex_data(
#         num_nodes=100,
#         theta=custom_theta,
#         edge_factor=10, 
#         signal_structure='small-world',  # Small-world structure
#         small_world_k=4,                # Initial neighbors per node
#         small_world_p=0.15,             # Rewiring probability
#         seed=42
#     )
    
#     A = data_smallworld['A']
#     R_list = data_smallworld['R_list']
#     true_theta = data_smallworld['true_theta']
#     xi = data_smallworld['xi']
#     signal_mask = data_smallworld['signal_mask']
    
#     print(f"Generated small-world data with {len(R_list)} relation layers")
#     print(f"Signal layers: {np.sum(signal_mask)}")
#     print(f"Noise layers: {np.sum(~signal_mask)}")
    
#     # Grid search for lambda and rho
#     print("\n--- Example 2: Grid search for lambda and rho ---")
#     lambd_values = np.exp(np.linspace(np.log(0.01), np.log(20), num=10))
#     rho_values = np.exp(np.linspace(np.log(0.01), np.log(20), num=10))
    
#     cv_results = cross_validate_SMNR(
#         R_list, A, xi, 
#         lambd_values=lambd_values,
#         rho_values=rho_values,
#         directed=False,
#         selfloops=False,
#         n_folds=3,
#         seed=42
#     )
    
#     best_lambda = cv_results['best_lambda']
#     best_rho = cv_results['best_rho']
#     best_mse = cv_results['best_val_mse']
    
#     print(f"Best lambda: {best_lambda}, Best rho: {best_rho}, Best MSE: {best_mse:.4f}")
    
#     # Train final model with best parameters
#     print("\nTraining final model with best parameters...")
#     model = SMNR(
#         R_list, A, xi=xi, lambd=best_lambda, rho=best_rho,
#         directed=False, selfloops=False, verbose=True
#     )
    
#     print("\nModel results:")
#     print(f"Converged: {model['converged']} in {model['n_iter']} iterations")
#     print(f"Log-likelihood: {model['loglikelihood']:.2f}")
#     print(f"AIC: {model['AIC']:.2f}")
#     print(f"McFadden R²: {model['R2']:.4f}")
#     print(f"Cox Snell R²: {model['cox_snell_R2']:.4f}")
    
#     # Compare estimated theta with true values
#     print("\nParameter comparison:")
#     for i, (est, true) in enumerate(zip(model['z'], true_theta)):
#         layer_type = "Signal" if signal_mask[i] else "Noise"
#         print(f"Layer {i+1} ({layer_type}): True={true:.2f}, Estimated={est:.4f}")

#     print("\nEstimated parameters with statistical significance:")
#     for i, (theta_est, std_err, p_val, ci) in enumerate(zip(
#         model['z'], model['std_errors'], model['p_values'], model['conf_intervals']
#     )):
#         layer_type = "Signal" if signal_mask[i] else "Noise"
#         sig_star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
#         print(f"  Layer {i+1} ({layer_type}):")
#         print(f"    theta = {theta_est:.4f} ± {std_err:.4f} {sig_star}")
#         print(f"    p-value = {p_val:.4f}")
#         print(f"    95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]")

#     # Analyze results
#     signal_est = model['z'][signal_mask]
#     noise_est = model['z'][~signal_mask]
#     true_signal = np.array(true_theta)[signal_mask]
#     true_noise = np.array(true_theta)[~signal_mask]
    
#     print("\n=== Performance Summary ===")
#     print(f"Signal layers: Mean absolute error = {np.mean(np.abs(signal_est - true_signal)):.4f}")
#     print(f"Noise layers: Mean absolute error = {np.mean(np.abs(noise_est - true_noise)):.4f}")
#     print(f"Regularization effectiveness: {np.sum(np.abs(noise_est) < 0.001)}/{len(noise_est)} noise parameters shrunk near zero")

    
#     # Example 3: Using predefined theta to generate scale-free structure in signal layers but not in noise layers
#     print("\n--- Example 3: Scale-free data generation with custom theta ---")
#     custom_theta = [3.0, 0, 2.0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0]  # 6 base layers
    
#     data_scalefree = generate_multiplex_data(
#         num_nodes=100,
#         theta=custom_theta,
#         edge_factor=10, 
#         signal_structure='scale-free',  # Scale-free structure
#         scale_free_m=2,                 # Edges to add per node
#         seed=42
#     )
    
#     A = data_scalefree['A']
#     R_list = data_scalefree['R_list']
#     true_theta = data_scalefree['true_theta']
#     xi = data_scalefree['xi']
#     signal_mask = data_scalefree['signal_mask']
    
#     print(f"Generated scale-free data with {len(R_list)} relation layers")
#     print(f"Signal layers: {np.sum(signal_mask)}")
#     print(f"Noise layers: {np.sum(~signal_mask)}")
    
#     # Grid search for lambda and rho
#     print("\n--- Example 3: Grid search for lambda and rho ---")
#     lambd_values = np.exp(np.linspace(np.log(0.01), np.log(20), num=10))
#     rho_values = np.exp(np.linspace(np.log(0.01), np.log(20), num=10))
    
#     cv_results = cross_validate_SMNR(
#         R_list, A, xi, 
#         lambd_values=lambd_values,
#         rho_values=rho_values,
#         directed=False,
#         selfloops=False,
#         n_folds=3,
#         seed=42
#     )
    
#     best_lambda = cv_results['best_lambda']
#     best_rho = cv_results['best_rho']
#     best_mse = cv_results['best_val_mse']
    
#     print(f"Best lambda: {best_lambda}, Best rho: {best_rho}, Best MSE: {best_mse:.4f}")
    
#     # Train final model with best parameters
#     print("\nTraining final model with best parameters...")
#     model = SMNR(
#         R_list, A, xi=xi, lambd=best_lambda, rho=best_rho,
#         directed=False, selfloops=False, verbose=True
#     )
    
#     print("\nModel results:")
#     print(f"Converged: {model['converged']} in {model['n_iter']} iterations")
#     print(f"Log-likelihood: {model['loglikelihood']:.2f}")
#     print(f"AIC: {model['AIC']:.2f}")
#     print(f"McFadden R²: {model['R2']:.4f}")
#     print(f"Cox Snell R²: {model['cox_snell_R2']:.4f}")
    
#     # Compare estimated theta with true values
#     print("\nParameter comparison:")
#     for i, (est, true) in enumerate(zip(model['z'], true_theta)):
#         layer_type = "Signal" if signal_mask[i] else "Noise"
#         print(f"Layer {i+1} ({layer_type}): True={true:.2f}, Estimated={est:.4f}")

#     print("\nEstimated parameters with statistical significance:")
#     for i, (theta_est, std_err, p_val, ci) in enumerate(zip(
#         model['z'], model['std_errors'], model['p_values'], model['conf_intervals']
#     )):
#         layer_type = "Signal" if signal_mask[i] else "Noise"
#         sig_star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
#         print(f"  Layer {i+1} ({layer_type}):")
#         print(f"    theta = {theta_est:.4f} ± {std_err:.4f} {sig_star}")
#         print(f"    p-value = {p_val:.4f}")
#         print(f"    95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]")

#     # Analyze results
#     signal_est = model['z'][signal_mask]
#     noise_est = model['z'][~signal_mask]
#     true_signal = np.array(true_theta)[signal_mask]
#     true_noise = np.array(true_theta)[~signal_mask]
    
#     print("\n=== Performance Summary ===")
#     print(f"Signal layers: Mean absolute error = {np.mean(np.abs(signal_est - true_signal)):.4f}")
#     print(f"Noise layers: Mean absolute error = {np.mean(np.abs(noise_est - true_noise)):.4f}")
#     print(f"Regularization effectiveness: {np.sum(np.abs(noise_est) < 0.001)}/{len(noise_est)} noise parameters shrunk near zero")
    
        
#     # Example 4: Using predefined theta to generate mixed structure in signal layers but not in noise layers
#     print("\n--- Example 4: Mixed data generation with custom theta ---")
#     custom_theta = [3.0, 0, 2.0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0]  # 6 base layers
    
#     data_mixed = generate_multiplex_data(
#         num_nodes=100,
#         theta=custom_theta,
#         edge_factor=10, 
#         signal_structure=['block', 'small-world', 'scale-free'],  # Mixed structure
#         small_world_k=3,
#         small_world_p=0.2,
#         scale_free_m=3,
#         seed=42
#     )
    
#     A = data_mixed['A']
#     R_list = data_mixed['R_list']
#     true_theta = data_mixed['true_theta']
#     xi = data_mixed['xi']
#     signal_mask = data_mixed['signal_mask']
    
#     print(f"Generated mixed data with {len(R_list)} relation layers")
#     print(f"Signal layers: {np.sum(signal_mask)}")
#     print(f"Noise layers: {np.sum(~signal_mask)}")
    
#     # Grid search for lambda and rho
#     print("\n--- Example 4: Grid search for lambda and rho ---")
#     lambd_values = np.exp(np.linspace(np.log(0.01), np.log(20), num=10))
#     rho_values = np.exp(np.linspace(np.log(0.01), np.log(20), num=10))
    
#     cv_results = cross_validate_SMNR(
#         R_list, A, xi, 
#         lambd_values=lambd_values,
#         rho_values=rho_values,
#         directed=False,
#         selfloops=False,
#         n_folds=3,
#         seed=42
#     )
    
#     best_lambda = cv_results['best_lambda']
#     best_rho = cv_results['best_rho']
#     best_mse = cv_results['best_val_mse']
    
#     print(f"Best lambda: {best_lambda}, Best rho: {best_rho}, Best MSE: {best_mse:.4f}")
    
#     # Train final model with best parameters
#     print("\nTraining final model with best parameters...")
#     model = SMNR(
#         R_list, A, xi=xi, lambd=best_lambda, rho=best_rho,
#         directed=False, selfloops=False, verbose=True
#     )
    
#     print("\nModel results:")
#     print(f"Converged: {model['converged']} in {model['n_iter']} iterations")
#     print(f"Log-likelihood: {model['loglikelihood']:.2f}")
#     print(f"AIC: {model['AIC']:.2f}")
#     print(f"McFadden R²: {model['R2']:.4f}")
#     print(f"Cox Snell R²: {model['cox_snell_R2']:.4f}")
    
#     # Compare estimated theta with true values
#     print("\nParameter comparison:")
#     for i, (est, true) in enumerate(zip(model['z'], true_theta)):
#         layer_type = "Signal" if signal_mask[i] else "Noise"
#         print(f"Layer {i+1} ({layer_type}): True={true:.2f}, Estimated={est:.4f}")

#     print("\nEstimated parameters with statistical significance:")
#     for i, (theta_est, std_err, p_val, ci) in enumerate(zip(
#         model['z'], model['std_errors'], model['p_values'], model['conf_intervals']
#     )):
#         layer_type = "Signal" if signal_mask[i] else "Noise"
#         sig_star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
#         print(f"  Layer {i+1} ({layer_type}):")
#         print(f"    theta = {theta_est:.4f} ± {std_err:.4f} {sig_star}")
#         print(f"    p-value = {p_val:.4f}")
#         print(f"    95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]")

#     # Analyze results
#     signal_est = model['z'][signal_mask]
#     noise_est = model['z'][~signal_mask]
#     true_signal = np.array(true_theta)[signal_mask]
#     true_noise = np.array(true_theta)[~signal_mask]
    
#     print("\n=== Performance Summary ===")
#     print(f"Signal layers: Mean absolute error = {np.mean(np.abs(signal_est - true_signal)):.4f}")
#     print(f"Noise layers: Mean absolute error = {np.mean(np.abs(noise_est - true_noise)):.4f}")
#     print(f"Regularization effectiveness: {np.sum(np.abs(noise_est) < 0.001)}/{len(noise_est)} noise parameters shrunk near zero")