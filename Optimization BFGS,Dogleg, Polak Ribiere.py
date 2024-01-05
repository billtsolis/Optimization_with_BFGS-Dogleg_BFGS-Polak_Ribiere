import numpy as np
import pandas as pd

# Read the CSV files
df1 = pd.read_csv('data_ComputersElectronicProducts.csv', skiprows=6, delimiter='\t')
df2 = pd.read_csv('data_DefenseCapitalGoods.csv', skiprows=6, delimiter='\t')
df3 = pd.read_csv('data_MotorVehiclesParts.csv', skiprows=6, delimiter='\t')
df4 = pd.read_csv('data_PrimaryMetals.csv', skiprows=6, delimiter='\t')

# Διαχωρίζουμε τη στήλη "Period,Value" σε δύο στήλες "Period" και "Value"
df1[['Period', 'Value']] = df1['Period,Value'].str.split(',', expand=True)
df2[['Period', 'Value']] = df2['Period,Value'].str.split(',', expand=True)
df3[['Period', 'Value']] = df3['Period,Value'].str.split(',', expand=True)
df4[['Period', 'Value']] = df4['Period,Value'].str.split(',', expand=True)

# Μετατροπή της στήλης "Value" σε αριθμητικές τιμές
df1['Value'] = pd.to_numeric(df1['Value'], errors='coerce')
df2['Value'] = pd.to_numeric(df2['Value'], errors='coerce')
df3['Value'] = pd.to_numeric(df3['Value'], errors='coerce')
df4['Value'] = pd.to_numeric(df4['Value'], errors='coerce')

# Αφαίρεση των γραμμών με απουσιάζουσες τιμές
df1 = df1.dropna()
df2 = df2.dropna()
df3 = df3.dropna()
df4 = df4.dropna()

R1 = (df1['Value'].values / df1['Value'].values[0]) - 1
R2 = (df2['Value'].values / df2['Value'].values[0]) - 1
R3 = (df3['Value'].values / df3['Value'].values[0]) - 1
R4 = (df4['Value'].values / df4['Value'].values[0]) - 1

R = [R1, R2, R3, R4]
#Calculate mean R
def calculate_mean_relative_return(R):
    n = len(R)
    mean_R = []

    for i in range(n):
        mean_R_i = np.mean(R[i])
        mean_R.append(mean_R_i)

    return mean_R

mean_R = np.array(calculate_mean_relative_return(R))
#Calculate covariance matrix
def calculate_covariance_matrix(R):
    num_variables = len(R)  # Number of variables
    

    M = np.zeros((num_variables, num_variables))

    for i in range(num_variables):
        for j in range(num_variables):
            cov_ij = np.cov(R[i], R[j])[0, 1]
            M[i, j] = cov_ij

    return M
# Call the function with your R values
M = calculate_covariance_matrix(R)
#Calculate objective function
def objective_function(x, mean_R, M, lambd):
    total_sum = np.sum(x)
    
    w= x / total_sum if total_sum != 0 else x
    
    sum1 = -np.dot(w, mean_R) 
    
    sum2 = np.sum(np.outer(w,w) * M)
    
    return sum1 + lambd*sum2
#Calculate gradient
def gradient(x, mean_R, M, lambd):
    def partial_derivative_x1(x, mean_R, M, lambd):
        total_sum = np.sum(x)
        part1 = - ( - ((mean_R[3] - mean_R[0]) * x[3] + (mean_R[2] - mean_R[0]) * x[2] + (mean_R[1] - mean_R[0]) * x[1])/total_sum**2)
        part2 = 2 * (M[0,0] * x[0] * (x[1]+x[2]+x[3])/(total_sum)**3 - M[0,1] * x[1]*(x[0]-x[1]-x[2]-x[3])/(total_sum)**3 - M[0,2] * x[2]*(x[0]-x[1]-x[2]-x[3])/(total_sum)**3 - M[0,3] * x[3]*(x[0]-x[1]-x[2]-x[3])/(total_sum)**3)
        part3 = 2 * (- M[3,3] * x[3]**2/(total_sum)**3 - M[2,2] * x[2]**2/(total_sum)**3 - M[1,1] * x[1]**2/(total_sum)**3)
        part4 = 4 * (- M[1,2] * x[1]*x[2]/(total_sum)**3 - M[1,3] * x[1]*x[3]/(total_sum)**3 - M[2,3] * x[2]*x[3]/(total_sum)**3)       
   
        return (part1 + lambd * (part2 + part3 + part4))

    def partial_derivative_x2(x, mean_R, M, lambd):
        total_sum = np.sum(x)
        part1 = - ( - ((mean_R[3] - mean_R[1]) * x[3] + (mean_R[2] - mean_R[1]) * x[2] + (mean_R[0] - mean_R[1]) * x[0])/total_sum**2)
        part2 = 2 * (M[1,1] * x[1] * (x[0]+x[2]+x[3])/(total_sum)**3 - M[1,0] * x[0]*(x[1]-x[0]-x[2]-x[3])/(total_sum)**3 - M[1,2] * x[2]*(x[0]-x[1]-x[2]-x[3])/(total_sum)**3 - M[1,3] * x[3]*(x[1]-x[0]-x[2]-x[3])/(total_sum)**3)
        part3 = 2 * (- M[3,3] * x[3]**2/(total_sum)**3 - M[2,2] * x[2]**2/(total_sum)**3 - M[0,0] * x[0]**2/(total_sum)**3)
        part4 = 4 * (- M[0,2] * x[0]*x[2]/(total_sum)**3 - M[0,3] * x[0]*x[3]/(total_sum)**3 - M[2,3] * x[2]*x[3]/(total_sum)**3)       

        return (part1 + lambd * (part2 + part3 + part4))

    def partial_derivative_x3(x, mean_R, M, lambd):
        total_sum = np.sum(x)
        part1 = - ( - ((mean_R[3] - mean_R[2]) * x[3] + (mean_R[1] - mean_R[2]) * x[1] + (mean_R[0] - mean_R[2]) * x[0])/total_sum**2)
        part2 = 2 * (M[2,2] * x[2] * (x[0]+x[1]+x[3])/(total_sum)**3 - M[2,0] * x[0]*(x[2]-x[0]-x[1]-x[3])/(total_sum)**3 - M[2,1] * x[1]*(x[2]-x[0]-x[1]-x[3])/(total_sum)**3 - M[2,3] * x[3]*(x[2]-x[0]-x[1]-x[3])/(total_sum)**3)
        part3 = 2 * (- M[3,3] * x[3]**2/(total_sum)**3 - M[1,1] * x[1]**2/(total_sum)**3 - M[0,0] * x[0]**2/(total_sum)**3)
        part4 = 4 * (- M[0,1] * x[0]*x[1]/(total_sum)**3 - M[0,3] * x[0]*x[3]/(total_sum)**3 - M[1,3] * x[1]*x[3]/(total_sum)**3)       
   
        return (part1 + lambd * (part2 + part3 + part4))

    def partial_derivative_x4(x, mean_R, M, lambd):
        total_sum = np.sum(x)
        part1 = - ((mean_R[3] - mean_R[2]) * x[2] + (mean_R[3] - mean_R[1]) * x[1] + (mean_R[3] - mean_R[0]) * x[0])/total_sum**2
        part2 = 2 * (M[3,3] * x[3] * (x[0]+x[1]+x[2])/(total_sum)**3 - M[3,0] * x[0]*(x[3]-x[0]-x[1]-x[2])/(total_sum)**3 - M[3,1] * x[1]*(x[3]-x[0]-x[1]-x[2])/(total_sum)**3 - M[3,2] * x[2]*(x[3]-x[0]-x[1]-x[2])/(total_sum)**3)
        part3 = 2 * (- M[2,2] * x[2]**2/(total_sum)**3 - M[1,1] * x[1]**2/(total_sum)**3 - M[0,0] * x[0]**2/(total_sum)**3)
        part4 = 4 * (- M[0,1] * x[0]*x[1]/(total_sum)**3 - M[0,2] * x[0]*x[2]/(total_sum)**3 - M[1,2] * x[1]*x[2]/(total_sum)**3)       

        return (part1 + lambd * (part2 + part3 + part4))

    # Επιστροφή ενός πίνακα NumPy αντί για ένα tuple
    return np.array([partial_derivative_x1(x, mean_R, M, lambd),
                     partial_derivative_x2(x, mean_R, M, lambd),
                     partial_derivative_x3(x, mean_R, M, lambd),
                     partial_derivative_x4(x, mean_R, M, lambd)])
    
   
#line search, zoom, bisection, BFGS
def line_search_wolfe(f, g, x, d, c1=1e-5, c2=0.9, max_iter=100):
    alpha = 1  # Αρχικό μήκος βήματος
    phi0 = f(x)  # Αρχική τιμή της συνάρτησης
    alpha_max = 2  # Μέγιστο μήκος βήματος
    phi_prime0 = np.dot(g(x), d)  # Αρχική παράγωγος της συνάρτησης στην κατεύθυνση d
    alpha_prev = 0.0  # Προηγούμενο μήκος βήματος
    phi_prev = phi0  # Προηγούμενη τιμή της συνάρτησης
    
    for i in range(max_iter):
        x_new = x + alpha * d  # Υπολογισμός νέας θέσης
        phi = f(x_new)  # Υπολογισμός τιμής συνάρτησης στο νέο σημείο
        phi_prime = np.dot(g(x_new), d)  # Υπολογισμός παραγώγου συνάρτησης στο νέο σημείο

        # Συνθήκη Armijo
        if phi > phi0 + c1 * alpha * phi_prime0 or (i > 0 and phi >= phi_prev):
            return zoom(f, g, x, d, alpha_prev, alpha)

        # Συνθήκη Curvature
        if abs(phi_prime) <= -c2 * phi_prime0:
            return alpha

        # Συνθήκη για αρνητική παράγωγο
        if phi_prime >= 0:
            return zoom(f, g, x, d, alpha, alpha_prev)

        alpha_prev = alpha
        phi_prev = phi
        alpha *= 2.0  # Διπλασιασμός μήκους βήματος

        # Εφαρμογή ορίου για το μέγιστο μήκος βήματος
        if alpha > alpha_max:
            return alpha_max

    return alpha

def zoom(f, g, x, d, alpha_lo, alpha_hi, c1=1e-4, c2=0.9, max_iter=100):
    phi0 = f(x)  # Αρχική τιμή της συνάρτησης
    phi_prime0 = np.dot(g(x), d)  # Αρχική παράγωγος της συνάρτησης στην κατεύθυνση d

    for i in range(max_iter):
        alpha = 0.5 * (alpha_lo + alpha_hi)  # Υπολογισμός νέου μήκους βήματος
        x_new = x + alpha * d  # Υπολογισμός νέας θέσης
        phi = f(x_new)  # Υπολογισμός τιμής συνάρτησης στο νέο σημείο
        phi_prime = np.dot(g(x_new), d)  # Υπολογισμός παραγώγου συνάρτησης στο νέο σημείο

        # Συνθήκες Armijo και Curvature
        if phi > phi0 + c1 * alpha * phi_prime0 or phi >= f(x + alpha_lo * d):
            alpha_hi = alpha
        else:
            if abs(phi_prime) <= -c2 * phi_prime0:
                return alpha
            if phi_prime * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha

    return alpha

def bisection(f, a, b, epsilon):
    k = 0
    stop = 0
    while stop == 0:
        x = 0.5 * (b + a)
        fx = f(x)  # Calculate the value of f(x)
        if np.any(fx == 0):  # Check if any element of fx is zero
            stop = 1
        else:
            if np.any(f(a) * fx < 0):  # Check if any element of f(a) * fx is negative
                a_new = a
                b_new = x
            else:
                a_new = x
                b_new = b
            k += 1
            if b_new - a_new < epsilon:
                stop = 1
        a = a_new
        b = b_new
    return 0.5 * (b + a)

def bfgs_algorithm(x0, mean_R, M, lambd, max_iter=100, tol=1e-6):
    n = len(x0)
    x = x0.copy()
    B = np.eye(n)  # Initial approximation of the Hessian matrix
    g = gradient(x, mean_R, M, lambd)  # Calculate the gradient
    k = 0
    
    for _ in range(max_iter):
        d = -np.dot(np.linalg.pinv(B), g)  # Calculate the search direction
        
        alpha = line_search_wolfe(lambda x: objective_function(x, mean_R, M, lambd),
                                  lambda x: gradient(x, mean_R, M, lambd),
                                  x, d)  # Calculate the step size
        
        x_new = x + alpha * d  # Update the solution
        #Call Bisection
        x_new = np.array([bisection(lambda x: objective_function(x, mean_R, M, lambd), 0, 1, 1e-8) if xi < 0 or xi > 1 else xi for xi in x_new])
        # Υπολογισμός του νέου gradient
        g_new = gradient(x_new, mean_R, M, lambd)  # Calculate the new gradient
        # Υπολογισμός των διανυσμάτων s και y
        s = x_new - x
        y = g_new - g
        #Update BFGS
        denominator = np.dot(y, s)
        if denominator != 0.0:
            B = B - np.outer(np.dot(B, s), np.dot(s, B)) / denominator + np.outer(y, y) / np.dot(y, s)
        
        x = x_new
        
        k += 1
       
    return x

bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
R = [R1, R2, R3, R4]
lambd = 1.5
num_initial_points = 10
results = []
used_initial_points = set()  # To keep track of used initial points

best_solution = None
best_objective_value = float('inf')
best_x0 = None

# Execute the algorithm from 10 independently and randomly selected initial points
for i in range(num_initial_points):
    while True:
        x0 = np.random.uniform(bounds[0][0], bounds[0][1], size=4)
        # Convert the array to a tuple for set comparison
        x0_tuple = tuple(x0)
        if x0_tuple not in used_initial_points:
            used_initial_points.add(x0_tuple)
            break
    
    result = bfgs_algorithm(x0, mean_R, M, lambd)
    results.append(result)
    
    # Calculate the total sum and weight vector
    total_sum = np.sum(result)
    w = (result / total_sum)

    # Print information for each run
    print ("BGFS with line search")
    print(f"Run {i + 1}:")
    print("Initial Point (x0):", x0)
    print("Solution:", result)
    print("w=", w)
    print("Expected Return:", np.dot(w, mean_R))
    print("Risk:", np.sum(np.outer(w, w) * M))
    objective_value = -np.dot(w, mean_R) + lambd * np.sum(np.outer(w, w) * M)
    print("Objective Function Value:", objective_value)
    
    print("\n")

    # Check if the current solution is better than the previous best
    if objective_value < best_objective_value:
        best_solution = result
        best_objective_value = objective_value
        best_x0 = x0

# Print information about the best solution
print ("Best BGFS line search")
print("Best Solution:")
print("Initial Point (x0):", best_x0)
print("Solution:", best_solution)
best_w = best_solution / np.sum(best_solution)
print("Best w:", best_w)
print("Expected Return:", np.dot(best_w, mean_R))
print("Risk:", np.sum(np.outer(best_w, best_w) * M))
print("Objective Function Value:", best_objective_value)
print("\n")
#BFGS DODLEG
def dogleg_bfgs(x0, mean_R, M, lambd, max_iter=100, tol=1e-6, delta=0.1):
    n = len(x0)
    x = x0.copy()
    B = np.eye(n)  # Initial approximation of the Hessian matrix
    g = gradient(x, mean_R, M, lambd)  # Calculate the gradient
    k = 0
    
    while np.linalg.norm(g) > tol and k < max_iter:
        d_B = -np.dot(np.linalg.pinv(B), g)
        d_U = -g * np.dot(g, g) / np.dot(g, np.dot(B, g))
        
        if np.linalg.norm(d_B) <= delta:
            d = d_B  # Take the Newton direction
        elif np.linalg.norm(d_U) >= delta:
            d = (delta / np.linalg.norm(d_U)) * d_U  # Take the steepest descent direction
        else:
            a = np.linalg.norm(d_U) ** 2
            b = 2 * np.dot(d_U, d_B - d_U)
            c = np.linalg.norm(d_B - d_U) ** 2 - delta ** 2
            
            # Check if the value inside the square root is non-negative
            if b ** 2 - 4 * a * c < 0:
                t = 0  # Set t to 0 if the value is negative
            else:
                t = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)  # Compute the intersection point
            
            d = d_U + t * (d_B - d_U)  # Take the dogleg direction
        
            
        x_new = x + d  # Update the solution
        x_new = np.array([bisection(lambda x: objective_function(x, mean_R, M, lambd), 0, 1, 1e-8) if xi < 0 or xi > 1 else xi for xi in x_new])
        g_new = gradient(x_new, mean_R, M, lambd)  # Calculate the new gradient
        

        
        denominator_rho = -np.dot(g, d) - 0.5 * np.dot(d, np.dot(B, d))
        if denominator_rho != 0.0:
            rho = (objective_function(x, mean_R, M, lambd) - objective_function(x_new, mean_R, M, lambd)) / denominator_rho
        else:
            rho = 0.0
            
        if rho < 0.25:
            delta *= 0.25
        elif rho > 0.75 and np.linalg.norm(d) == delta:
            delta = min(2 * delta, 1.0)
        else:
            delta = delta
        
        if rho > 0.1:
            # BFGS update
            y = g_new - g
            s = x_new - x
            denominator = np.dot(y, s)
            if denominator != 0.0:
                B = B - np.outer(np.dot(B, s), np.dot(s, B)) / denominator + np.outer(y, y) / np.dot(y, s)
            x = x_new
      
        g = g_new
        k += 1
    
    return x

best_solution2 = None
best_objective_value2 = float('inf')
best_x02 = None

# Execute the algorithm from 10 independently and randomly selected initial points
for i in range(num_initial_points):
    while True:
        x0 = np.random.uniform(bounds[0][0], bounds[0][1], size=4)
        # Convert the array to a tuple for set comparison
        x0_tuple = tuple(x0)
        if x0_tuple not in used_initial_points:
            used_initial_points.add(x0_tuple)
            break
    
    result = dogleg_bfgs(x0, mean_R, M, lambd)
    results.append(result)
    # Calculate the total sum and weight vector
    total_sum = np.sum(result)
    w = result / total_sum

    # Print information for each run
    print("BGFS DOGLEG")
    print(f"Run {i + 1}:")
    print("Initial Point (x0):", x0)
    print("Solution:", result)
    print("w=", w)
    print("Expected Return:", np.dot(w, mean_R))
    print("Risk:", np.sum(np.outer(w, w) * M))
    objective_value = -np.dot(w, mean_R) + lambd * np.sum(np.outer(w, w) * M)
    print("Objective Function Value:", objective_value)
    print("\n")

    # Check if the current solution is better than the previous best
    if objective_value < best_objective_value2:
        best_solution2 = result
        best_objective_value2 = objective_value
        best_x02 = x0

# Print information about the best solution
print("Best BGFS DOGLEG")
print("Best Solution:")
print("Best Initial Point (x0):", best_x02)
print("Solution:", best_solution2)
best_w = best_solution2 / np.sum(best_solution2) 
print("Best w:", best_w)
print("Expected Return:", np.dot(best_w, mean_R))
print("Risk:", np.sum(np.outer(best_w, best_w) * M))
print("Objective Function Value:", best_objective_value2)

#Polak Ribiere
def line_search_wolfe2(f, g, x, d, c1=1e-4, c2=0.9, max_iter=100):
    alpha = 0.1
    phi0 = f(x)
    alpha_max = 2 
    phi_prime0 = np.dot(g(x), d)
    alpha_prev = 0.0
    phi_prev = phi0
    
    
    for i in range(max_iter):
        x_new = x + alpha * d
        phi = f(x_new)
        phi_prime = np.dot(g(x_new), d)
        
        if phi > phi0 + c1 * alpha * phi_prime0 or (i > 1 and phi >= phi_prev):
            return zoom2(f, g, x, d, alpha_prev, alpha)
        
        if abs(phi_prime) <= -c2 * phi_prime0:
            return alpha
        
        if phi_prime >= 0:
            return zoom2(f, g, x, d, alpha, alpha_prev)
        
        alpha_prev = alpha
        phi_prev = phi
        
        alpha *= 2.0
        if alpha > alpha_max:
            return alpha_max  # Εφαρμόζεται το `amax` ως όριο
    return alpha

def zoom2(f, g, x, d, alpha_lo, alpha_hi, c1=1e-4, c2=0.9, max_iter=100):
    phi0 = f(x)
    phi_prime0 = np.dot(g(x), d)
    
    for i in range(max_iter):
        alpha = 0.5 * (alpha_lo + alpha_hi)
        x_new = x + alpha * d
        phi = f(x_new)
        phi_prime = np.dot(g(x_new), d)
        
        if phi > phi0 + c1 * alpha * phi_prime0 or phi >= f(x + alpha_lo * d):
            alpha_hi = alpha
        else:
            if abs(phi_prime) <= -c2 * phi_prime0:
                return alpha
            if phi_prime * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha
        
    return alpha

def polak_ribiere_cg(x0, mean_R, M, lambd, max_iter=100, tol=1e-6, restart_interval=14):
    x = x0.copy()
    g = gradient(x, mean_R, M, lambd)  # Calculate the gradient
    d = -g  # Initial search direction
    
    restart_counter = 0
    
    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        alpha = line_search_wolfe2(lambda x: objective_function(x, mean_R, M, lambd),
                              lambda x: gradient(x, mean_R, M, lambd),
                              x, d)  # Initial step size
        x_new = x + alpha * d  # Update x
        x_new = np.array([bisection(lambda x: objective_function(x, mean_R, M, lambd), 0, 1, 1e-6) if xi < 0 or xi > 1 else xi for xi in x_new])
        g_new = gradient(x_new, mean_R, M, lambd)  # Calculate the new gradient
        
        
        
        beta = max(np.dot(g_new, g_new - g) / np.dot(g, g), 0)  # Calculate the Polak-Ribiere coefficient
        
        if restart_counter >= restart_interval:
            d = -g_new  # Restart the search direction
            restart_counter = 0
        else:
            d = -g_new + beta * d  # Update the search direction
        
        x = x_new
        g = g_new
        restart_counter += 1
    
    return x

best_solution3 = None
best_objective_value3 = float('inf')
best_x03 = None
# Execute the algorithm from 10 independently and randomly selected initial points
for i in range(num_initial_points):
    while True:
        x0 = np.random.uniform(bounds[0][0], bounds[0][1], size=4)
        # Convert the array to a tuple for set comparison
        x0_tuple = tuple(x0)
        if x0_tuple not in used_initial_points:
            used_initial_points.add(x0_tuple)
            break
    
    result = polak_ribiere_cg(x0, mean_R, M, lambd)
    results.append(result)
    # Calculate the total sum and weight vector
    total_sum = np.sum(result)
    w = result / total_sum 

    # Print information for each run
    print("Polak Ribiere")
    print(f"Run {i + 1}:")
    print("Initial Point (x0):", x0)
    print("Solution:", result)
    print ("w=", w)
    print("Expected Return:", np.dot(w, mean_R))
    print("Risk:", np.sum(np.outer(w, w) * M))
    objective_value = -np.dot(w, mean_R) + lambd * np.sum(np.outer(w, w) * M)
    print("Objective Function Value:", objective_value)
    
    print("\n")

    # Check if the current solution is better than the previous best
    if objective_value < best_objective_value3:
        best_solution3 = result
        best_objective_value3 = objective_value
        best_x03 = x0

# Print information about the best solution
print("Best Polak Ribiere")
print("Best Solution:")
print("Best Initial Point (x0):", best_x03)
print("Solution:", best_solution3)
best_w = best_solution3 / np.sum(best_solution3) 
print("Best w:", best_w)
print("Expected Return:", np.dot(best_w, mean_R))
print("Risk:", np.sum(np.outer(best_w, best_w) * M))
print("Objective Function Value:", best_objective_value3)