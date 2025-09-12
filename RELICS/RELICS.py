#*******************************************
# ENERGYPLUS PLUGIN: MANUAL LIGHTING CONTROL - Apartment  
#*******************************************

"""
Created on Fri April 1 2022

@author: LVanThillo
"""

#----------------------
# INPUT ENERGYPLUS FILE
#----------------------

# Required input EnergyPlus per zone: 
    
#   Schedule:Constant,ScheduleLightZone,Fraction,0;

#   PythonPlugin:Variables,
#     PythonPluginGlobalVariables,   !- Name
#     PreviousStateLightingZone,     !- Variable Name 1

#  Required input EnergyPlus per zone with daylight enterance:  

#   Daylighting:Controls,
#     DaylightingControlZone,  !- Name
#     Zone,                    !- Zone or Space Name
#     SplitFlux,               !- Daylighting Method
#     ScheduleAlways0,         !- Availability Schedule Name
#     Continuous,              !- Lighting Control Type
#     0.3,                     !- Minimum Input Power Fraction for Continuous or ContinuousOff Dimming Control
#     0.2,                     !- Minimum Light Output Fraction for Continuous or ContinuousOff Dimming Control
#     ,                        !- Number of Stepped Control Steps
#     1,                       !- Probability Lighting will be Reset When Needed in Manual Stepped Control
#     ReferencePointZone,      !- Glare Calculation Daylighting Reference Point Name
#     ,                        !- Glare Calculation Azimuth Angle of View Direction Clockwise from Zone y-Axis {deg}
#     22,                      !- Maximum Allowable Discomfort Glare index
#     ,                        !- DElight Gridding Resolution {m2}
#     ReferencePointZone,      !- Daylighting Reference Point 1 Name
#     1,                       !- Fraction of Lights Controlled by Reference Point 1
#     500;                     !- Illuminance Setpoint at Reference Point 1 {lux}

# Daylighting:ReferencePoint,
#     ReferencePointZone,   !- Name
#     Zone,                 !- Zone or Space Name
#     ,                     !- X-Coordinate of Reference Point {m}
#     ,                     !- Y-Coordinate of Reference Point {m}
#     0.8;                  !- Z-Coordinate of Reference Point {m}

#------------------------------------
# PYTHON CODE MANUAL LIGHTING CONTROL
#------------------------------------

from pyenergyplus.plugin import EnergyPlusPlugin
import math
import importlib.util
import pandas as pd
import numpy as np
import re
from scipy.optimize import differential_evolution
from scipy.optimize import least_squares
import random

class LightingControl(EnergyPlusPlugin): 

    # PROVIDE INFORMATION ABOUT THE BUILDING GEOMETRY IN ENERGYPLUS
    rooms_lighting = [] # List all zone names with a defined presence pattern (in Occupancy.csv) and Corridor
    rooms_daylighting = [] # Give all the rooms where one daylighting reference point is formulated        
    rooms_daylighting_2 = [] # Give the rooms where 2 daylighting reference points are formulated 
    # Automatically generated
    rooms_with_daylighting = set(rooms_daylighting) | set(rooms_daylighting_2)
    rooms_without_daylighting = [room for room in rooms_lighting if room not in rooms_with_daylighting] # Lists all rooms without windows
    bedrooms = [room for room in rooms_lighting if 'Bedroom' in room] # Sum the bedrooms
    
    # IMPORT OCCUPANT BEHAVIOUR
    directory_occupancy = r'' # Refer to the directory in which the occupancy profiles have been saved. 
    occupancy_dataframe = pd.read_csv(directory_occupancy + '/Occupancy.csv').round({'Time': 2})
    occupancy_dataframe['Building'] = occupancy_dataframe.iloc[:,3:].sum(axis = 1) # Add a column with the total amount of present inhabitants
    occupancy_dataframe['LivingKitchen'] = occupancy_dataframe['Kitchen'] + occupancy_dataframe['Living']  # Add a column containing the sum of Living and Kitchen
    occupancy = occupancy_dataframe.to_dict()
    asleep_bedroom_dataframe = pd.read_csv(directory_occupancy + '/AsleepBedroom.csv').round({'Time': 2})
    asleep_bedroom = asleep_bedroom_dataframe.to_dict()
    bedroom_columns = [column for column in asleep_bedroom_dataframe.columns if 'Bedroom' in column]
    asleep_dataframe = asleep_bedroom_dataframe[bedroom_columns].sum(axis = 1)
    asleep = asleep_dataframe.to_dict()
    task_activities_dataframe = pd.read_csv(directory_occupancy+'/TaskAndActivities.csv').round({'Time' : 2})
    task_activities = task_activities_dataframe.to_dict()
    lighting_requirements_dataframe = pd.read_csv(directory_occupancy+'/PresenceLightingDesign2.csv').round({'Time' : 2})
    lighting_requirements_dataframe['LivingKitchen'] = lighting_requirements_dataframe[['Kitchen', 'Living']].apply(lambda row: 1 if (row['Kitchen'] == 1 or row['Living'] == 1) else 0, axis=1) # Add a column containing the information for an open kitchen
    lighting_requirements = lighting_requirements_dataframe.to_dict() 
    directory_data = r'' # Give the directory in which te data is saved. 
    
    
    def get_probability(rnd, prob, p_type='cum'):
        '''
        Find the x-value in a given comulative probability 'prob_cum' based on a
        given random y-value 'rnd'.
        '''
        if p_type != 'cum':
            prob = np.cumsum(prob)
            prob /= max(prob)
        idx = 1
        while rnd >= prob[idx - 1]:
            idx += 1
        return idx   
        
    def get_probability_interpolate(rnd, prob, p_type='cum'):
        '''
        Find the x-value in a given comulative probability 'prob_cum' based on a
        given random y-value 'rnd'.
        '''
        if p_type != 'cum':
            prob = np.cumsum(prob)
            prob /= max(prob)
        idx = 0
        while rnd >= prob[idx][1]:
            idx += 1
        prev_value = prob[idx - 1][0]
        prev_prob = prob[idx - 1][1]
        next_value = prob[idx][0]
        next_prob = prob[idx][1]
        value = prev_value + (next_value - prev_value)*(rnd - prev_prob)/(next_prob - prev_prob)
        return value

    def generate_random_exponential_probability_function(lower_expr, upper_expr, interval_range=(0, 1), steepness_range=(0, 1), increasing=True, constrain_point=None, max_iterations = 10):
        """
        Generates an exponential function within given bounds with broad curvature variation.
        Additional constraints in the generation are foreseen to reduce the steepness and interval in which the generated curve should be generated.
        Uses optimization to generate all possible intermediate curves.
        The function is constrained to the x-interval [0,1].
        Optionally, it can be constrained to pass through a specific point (x_0, y_0).
        """
        
        def exponential_function(a, b, c, x, d = None):
            """
            Computes an exponential function.
            
            Supports:
              - (a, b, c, x) → f(x) = a + c * exp(b * x)
              - (a, b, c, d, x) → f(x) = a * exp(b * x + c) + d
            """
            if d is None:
                # 3 parameters: f(x) = a + c * exp(b * x)
                return a + c * np.exp(b * x)
            else:
                # 4 parameters: f(x) = a * exp(b * x + c) + d
                return a * np.exp(b * x + c) + d
            
        def parse_exponential_function(expression):
            """Parses an exponential function string of the form 'a*exp(b*x+c)+d' and extracts a, b, c, d."""
            expression = expression.replace(" ", "")
            match = re.match(
                    r'^\s*([\d\.\-eE]+)\s*\*\s*exp\s*\(\s*([\d\.\-eE]+)\s*\*\s*x\s*([+-]\s*[\d\.\-eE]+)?\s*\)\s*([+-]\s*[\d\.\-eE]+)?\s*$',
                    expression
                )
            def evaluate_value(value):
                """Converts 'e^x' to a float or returns a numeric value directly."""
                if value and 'e^' in value:
                    return np.exp(float(value.split('^')[1]))
                return float(value) if value else 0.0  

            if match:
                a, b, c, d = match.groups()
                a = evaluate_value(a) if a else 1.0
                b = evaluate_value(b) if b else 1.0
                c = evaluate_value(c)
                d = evaluate_value(d)
                
                a_3 = d
                b_3 = b
                c_3 = a*np.exp(c) 
                return {'a': a_3, 'b': b_3, 'c': c_3}

            if expression == '0': 
                return {'a': 0.0, 'b': 1.0, 'c': 0.0}
            elif expression == '1': 
                return {'a': 0.0, 'b': 1.0, 'c': 0.0}

            raise ValueError("Expression format must be 'a*exp(b*x+c)+d'")

        x = np.linspace(0, 1, 100)
        lower_params = parse_exponential_function(lower_expr) if isinstance(lower_expr, str) else lower_expr
        upper_params = parse_exponential_function(upper_expr) if isinstance(upper_expr, str) else upper_expr
        

        def steepness(a, b, c, x_val):
            """Calculate steepness at x_val."""
            return c * b * np.exp(b * x_val)
        
        def loss_function(params):
            a, b, c = params
            fx = exponential_function(a, b, c, x)
            lower_bound = exponential_function(**lower_params, x=x)
            upper_bound = exponential_function(**upper_params, x=x)
            corrected_lower = np.minimum(lower_bound, upper_bound)
            corrected_upper = np.maximum(lower_bound, upper_bound)
            new_lower_bound = corrected_lower + (corrected_upper - corrected_lower) * interval_range[0]
            new_upper_bound = corrected_lower + (corrected_upper - corrected_lower) * interval_range[1]
            
            # Hard constraint: outside the bounds → infinite penalty
            if np.any(fx < new_lower_bound) or np.any(fx > new_upper_bound):
                return np.inf
            
            # Soft quadratic penalty near the bounds
            penalty = np.sum(np.maximum(0, new_lower_bound - fx)**2) + np.sum(np.maximum(0, fx - new_upper_bound)**2)

            # Extra constraint: function must pass through (x0, y0) if specified
            if constrain_point is not None:
                x0, y0 = constrain_point
                y_pred = exponential_function(a, b, c, x0)
                penalty += 1000 * (y_pred - y0) ** 2  # Heavy penalty for deviation
            
            x_interval = np.linspace(0, 1, 100)
            # Average steepness over the interval
            avg_steepness = np.mean([steepness(a, b, c, x_val) for x_val in x_interval])

            # Penalty adjustment based on steepness
            if avg_steepness > steepness_range[1]:
                new_upper_bound += (avg_steepness - steepness_range[1]) * 0.1  # Raise upper bound
            elif avg_steepness < steepness_range[0]:
                new_lower_bound -= (steepness_range[0] - avg_steepness) * 0.1  # Lower lower bound

            penalty += np.sum(np.maximum(0, fx - new_upper_bound))  # Above new upper bound
            penalty += np.sum(np.maximum(0, new_lower_bound - fx))  # Below new lower bound

            return penalty


        def residuals(params, x, y):
            return exponential_function(params[0], params[1], params[2], x) - y


        bounds = [
            (min(lower_params['a'], upper_params['a']), max(lower_params['a'], upper_params['a'])),
            (min(lower_params['b'], upper_params['b']), max(lower_params['b'], upper_params['b'])),
            (min(lower_params['c'], upper_params['c']), max(lower_params['c'], upper_params['c']))
        ]

        success = False
        for iteration in range(max_iterations): 
            result = differential_evolution(loss_function, bounds, strategy='best1bin', tol=1e-5, maxiter=5000)
            if result.success: 
                a_opt, b_opt, c_opt = result.x
                success = True
                break
                    
            if iteration == max_iterations - 1:                 
                def estimate_initialguess(x, y):
                    """
                    Approximation method for exponential fitting: y ≈ a + c * exp(b * x)
                    """
                    x = np.asarray(x, dtype=np.float64)
                    y = np.asarray(y, dtype=np.float64)

                    sort_idx = np.argsort(x)
                    x = x[sort_idx]
                    y = y[sort_idx]

                    # Step 1: Numerical integral approximation S (trapezoidal rule)
                    dx = np.diff(x)
                    dy = y[:-1] + y[1:]
                    S = np.zeros_like(x)
                    S[1:] = np.cumsum(0.5 * dx * dy)

                    # Solve A · [b, d, e] = B
                    n = len(x)
                    S2 = S**2
                    A = np.array([
                        [np.sum(S2),     np.sum(S * x),  np.sum(S)],
                        [np.sum(S * x),  np.sum(x**2),   np.sum(x)],
                        [np.sum(S),      np.sum(x),      n]
                    ])
                    B = np.array([
                        np.sum(S * y),
                        np.sum(x * y),
                        np.sum(y)
                    ])

                    try:
                        coeffs = np.linalg.solve(A, B)
                        b = coeffs[0]
                    except np.linalg.LinAlgError as e:
                        raise ValueError("Failed to solve for parameter b: singular matrix.") from e

                    # Step 2: Use b to compute exp(bx)
                    exp_bx = np.exp(b * x)
                    sum_exp = np.sum(exp_bx)
                    sum_exp2 = np.sum(exp_bx ** 2)
                    sum_y = np.sum(y)
                    sum_exp_y = np.sum(exp_bx * y)

                    A2 = np.array([
                        [n,         sum_exp],
                        [sum_exp,   sum_exp2]
                    ])
                    B2 = np.array([
                        sum_y,
                        sum_exp_y
                    ])

                    try:
                        a, c = np.linalg.solve(A2, B2)
                    except np.linalg.LinAlgError as e:
                        raise ValueError("Failed to solve for parameters a and c: singular matrix.") from e

                    return a, b, c

                x = np.linspace(0, 1, 100)
                lower_params = parse_exponential_function(lower_expr) if isinstance(lower_expr, str) else lower_expr
                upper_params = parse_exponential_function(upper_expr) if isinstance(upper_expr, str) else upper_expr

                lower_func = exponential_function(**lower_params, x=x)
                upper_func = exponential_function(**upper_params, x=x)

                lower_bound = np.minimum(lower_func, upper_func)
                upper_bound = np.maximum(lower_func, upper_func)
                delta = upper_bound - lower_bound

                for i in range(max_iterations + 5): 
                    alpha = np.random.uniform(interval_range[0], interval_range[1])
                    f_target = lower_bound + alpha * delta
                    interpolated_a, interpolated_b, interpolated_c = estimate_initialguess(x, f_target)
                 
                    fx = interpolated_a + interpolated_c * np.exp(interpolated_b * x)

                    # Check if fx stays within bounds with tolerance
                    tolerance = 0.05
                    cross_mask = delta <= 1e-2  # nearly equal or negative
                    adaptive_tol = np.ones_like(x) * tolerance
                    adaptive_tol[cross_mask] *= 3.0
                    absolute_tol = 1e-2
                    
                    lower_limit = lower_bound - np.maximum(absolute_tol, lower_bound * adaptive_tol)
                    upper_limit = upper_bound + np.minimum(absolute_tol, upper_bound * adaptive_tol)
                
                    if np.all((fx >= lower_limit) & (fx <= upper_limit)):
                        a_opt = interpolated_a
                        b_opt = interpolated_b
                        c_opt = interpolated_c
                        success = True
                        break

        if success:
            lower_bound = exponential_function(lower_params['a'], lower_params['b'], lower_params['c'], x)
            upper_bound = exponential_function(upper_params['a'], upper_params['b'], upper_params['c'], x)
            delta = upper_bound - lower_bound
            fx = exponential_function(a_opt, b_opt, c_opt, x)
            
            # Determine pointwise min/max since lower and upper may cross
            min_bound = np.minimum(lower_bound, upper_bound)
            max_bound = np.maximum(lower_bound, upper_bound)
            
            # 5% tolerance
            tolerance = 0.05
            cross_mask = delta <= 1e-2
            adaptive_tol = np.ones_like(x) * tolerance
            adaptive_tol[cross_mask] *= 3.0
            absolute_tol = 1e-2
                    
            lower_limit = min_bound - np.maximum(absolute_tol, lower_bound * adaptive_tol)
            upper_limit = max_bound + np.minimum(absolute_tol, upper_bound * adaptive_tol)
            
            # Only check if both bounds lie within [0, 1]
            valid_range_mask = (lower_limit >= 0) & (upper_limit <= 1)
            check_mask = valid_range_mask & ((fx < lower_limit) | (fx > upper_limit))
            
            if np.any(check_mask):
                idx = np.where(check_mask)[0][0]
                raise ValueError(
                    f"Generated function value fx[{idx}] = {fx[idx]:.4f} is outside the "
                    f"5% tolerance bounds [{lower_limit[idx]:.4f}, {upper_limit[idx]:.4f}] "
                    f"within the [0,1] validity range."
                )
                        
            result = {'a': a_opt, 'b': b_opt, 'c': c_opt}
        else: 
            raise ValueError("Generated function is outside 5% tolerance bounds.")

    def generate_random_logarithmic_probability_function(lower_expr, upper_expr, target_time, interval_range=(0, 1), max_deviation = 0.1, max_iterations = 5):
        """
        Generates a logarithmic function within given bounds with broad curvature variation.
        Additional constraints in the generation are foreseen to reduce the steepness and interval in which the generated curve should be generated.
        Furthermore, the a time target point is set as inflection point which should be approximated within 10%. 
        Uses optimization to generate all possible intermediate curves.
        The function is constrained to the x-interval [1,100].
        """
        
        def parse_logarithmic_function(expression):
            """
            Parses a logarithmic function string of the form 'a*log(b*x+c)+d' 
            and extracts parameters a, b, c, d.

            Supports cases where c and d are optional.
            Also allows a, b, c, d to be powers of e (e.g., 'e^2').
            """
            expression = expression.replace(" ", "")  # Remove spaces

            # Regex with optional c and d
            match = re.match(
                r'^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)?\s*\*?\s*log\(\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)?\s*\*?\s*x\s*([+-]\s*\d*\.?\d+(?:[eE][+-]?\d+)?)?\s*\)\s*([+-]\s*\d*\.?\d+(?:[eE][+-]?\d+)?)?\s*$',
                expression
            )

            def evaluate_value(value):
                """Convert 'e^x' to float or directly return numeric values."""
                if value and 'e^' in value:
                    return np.exp(float(value.split('^')[1]))  # Convert 'e^x' into exp(x)
                return float(value) if value else 0.0  # Default 0.0 if missing

            if match:
                a, b, c, d = match.groups()
                a = evaluate_value(a) if a else 1.0  # Default 1.0 if missing
                b = evaluate_value(b) if b else 1.0  # Default 1.0 if missing
                c = evaluate_value(c)  # Default 0.0 if missing
                d = evaluate_value(d)  # Default 0.0 if missing
                a_3 = d + a * np.log(b)
                b_3 = a
                c_3 = c / b
                return {'a': a_3, 'b': b_3, 'c': c_3}

            # Special cases for constant functions
            if expression == '0':
                return {'a': 0.0, 'b': 0.0, 'c': 1.0}
            elif expression == '1':
                return {'a': 1.0, 'b': 0.0, 'c': 1.0}

            raise ValueError("Expression format must be 'a*log(b*x+c)+d'")

        def logarithmic_function(a, b, c, x):
            return a + b * np.log(x + c)
            
        def second_derivative_zero(a, b, c):
            return -c + abs(b) / np.sqrt(2)  # Point of maximum curvature
        
        x = np.linspace(1, 100, 1000)
        x_interval = np.linspace(1, 60, 1000)
        lower_params = parse_logarithmic_function(lower_expr) if isinstance(lower_expr, str) else lower_expr
        upper_params = parse_logarithmic_function(upper_expr) if isinstance(upper_expr, str) else upper_expr
        
        def loss_function(params):
            a, b, c = params
            
            if b < 0 or c < 0:
                return 1e10  # Large penalty for invalid parameter values

            # Restrict evaluation to x ∈ [1, 60]
            x_interval = np.linspace(1, 60, 1000)
            
            fx = logarithmic_function(a, b, c, x_interval)
            lower_bound = logarithmic_function(lower_params['a'], lower_params['b'], lower_params['c'], x_interval)
            upper_bound = logarithmic_function(upper_params['a'], upper_params['b'], upper_params['c'], x_interval)
            new_lower_bound = lower_bound + (upper_bound - lower_bound) * interval_range[0]
            new_upper_bound = lower_bound + (upper_bound - lower_bound) * interval_range[1]

            # Penalise curve outside bounds
            penalty = 0
            penalty += np.sum(np.maximum(0, fx - new_upper_bound))  # Above upper bound
            penalty += np.sum(np.maximum(0, new_lower_bound - fx))  # Below lower bound

            # Penalise mismatch in target bending point
            x_bend = second_derivative_zero(a, b, c)
            if not (0.9 * target_time <= x_bend <= 1.1 * target_time) and x_bend <= 60:
                penalty += 1000 * abs(x_bend - target_time) / target_time

            return penalty
        
        def residuals(params, x, y):
            return logarithmic_function(params[0], params[1], params[2], x) - y

        # Define optimisation parameter bounds
        bounds = [
            (max(lower_params['a'], 0.01), upper_params['a']),
            (max(lower_params['b'], 0.1), upper_params['b']),
            (max(lower_params['c'], 0), upper_params['c'])
        ]
        bounds = [(min(bound), max(bound)) for bound in bounds]

        for iteration in range(max_iterations): 
            success = False
            result = differential_evolution(loss_function, bounds, strategy='best1bin', tol=1e-5, popsize=50, maxiter=2000)
            if result.success: 
                a_opt, b_opt, c_opt = result.x
                success = True
                break

            if iteration == max_iterations - 1: 
                def estimate_initialguess_log(x, y):
                    """
                    Estimate initial guess for log fitting: y ≈ A + B * log(x + C).

                    Parameters
                    ----------
                    x : array-like
                        Independent variable
                    y : array-like
                        Dependent variable

                    Returns
                    -------
                    A, B, C : float
                        Estimated parameters
                    """
                    x = np.asarray(x, dtype=np.float64)
                    y = np.asarray(y, dtype=np.float64)
                
                    # Filter: only keep values within [0, 1]
                    mask = (y >= 0.0) & (y <= 1.0)
                    if not np.any(mask):
                        raise ValueError("No valid y-values within [0, 1].")
                    x = x[mask]
                    y = y[mask]
                
                    # Sort by y (similar to exponential version sorted by x)
                    sort_idx = np.argsort(y)
                    y = y[sort_idx]
                    x = x[sort_idx]
                
                    # Step 1: Approximate integral S (trapezoidal rule over y)
                    dy = np.diff(y)
                    dx = x[:-1] + x[1:]
                    S = np.zeros_like(y)
                    S[1:] = np.cumsum(0.5 * dy * dx)
                
                    # Matrix solve for b'
                    n = len(y)
                    S2 = S**2
                    A = np.array([
                        [np.sum(S2),       np.sum(S * y),  np.sum(S)],
                        [np.sum(S * y),    np.sum(y**2),   np.sum(y)],
                        [np.sum(S),        np.sum(y),      n]
                    ])
                    B = np.array([
                        np.sum(S * x),
                        np.sum(y * x),
                        np.sum(x)
                    ])
                
                    try:
                        coeffs = np.linalg.solve(A, B)
                        b_prime = coeffs[0]
                    except np.linalg.LinAlgError as e:
                        raise ValueError("Failed to solve for parameter b': singular matrix.") from e
                
                    # Step 2: Use b' to compute exp(b'y)
                    exp_bY = np.exp(b_prime * y)
                    sum_exp = np.sum(exp_bY)
                    sum_exp2 = np.sum(exp_bY**2)
                    sum_x = np.sum(x)
                    sum_exp_x = np.sum(exp_bY * x)
                
                    A2 = np.array([
                        [n,        sum_exp],
                        [sum_exp,  sum_exp2]
                    ])
                    B2 = np.array([
                        sum_x,
                        sum_exp_x
                    ])
                
                    try:
                        a_prime, c_prime = np.linalg.solve(A2, B2)
                    except np.linalg.LinAlgError as e:
                        raise ValueError("Failed to solve for a' and c': singular matrix.") from e
                
                    # Back-transform to (A, B, C)
                    C = -a_prime
                    B = 1.0 / b_prime
                    if c_prime <= 0:
                        c_prime = np.finfo(float).eps
                    A = -B * np.log(c_prime)
                
                    return A, B, C
                    
                # Evaluate lower and upper functions over the x range
                lower_func = logarithmic_function(**lower_params, x=x)
                upper_func = logarithmic_function(**upper_params, x=x)

                # Determine pointwise minimum and maximum bounds
                lower_bound = np.minimum(lower_func, upper_func)
                upper_bound = np.maximum(lower_func, upper_func)
                delta = upper_bound - lower_bound

                # Generate intermediate function within bounds using random interpolation
                for i in range(max_iterations):
                    alpha = np.random.uniform(interval_range[0], interval_range[1])
                    f_target = lower_bound + alpha * delta
                    interpolated_a, interpolated_b, interpolated_c = estimate_initialguess_log(x, f_target)
                    
                    # Recalculate function values
                    fx = interpolated_a + interpolated_b * np.log(x + interpolated_c)

                    # Check if fx lies within the bounds with tolerance
                    tolerance = 0.05
                    cross_mask = delta <= 1e-2  # nearly equal or negative
                    adaptive_tol = np.ones_like(x) * tolerance
                    adaptive_tol[cross_mask] *= 3.0  # e.g., 3× wider around crossings
                    absolute_tol = 1e-2
                    
                    lower_limit = lower_bound - np.maximum(absolute_tol, lower_bound * adaptive_tol)
                    upper_limit = upper_bound + np.maximum(absolute_tol, upper_bound * adaptive_tol)

                    # Only check where bounds are effectively within [0,1]
                    inside_bounds_mask = (lower_bound >= 0) & (upper_bound <= 1)

                    # Apply check only within valid region
                    if np.all(
                        ~inside_bounds_mask |  # outside valid region → automatically ok
                        ((fx >= lower_limit) & (fx <= upper_limit))
                    ):
                        a_opt = interpolated_a
                        b_opt = interpolated_b
                        c_opt = interpolated_c
                        success = True
                        break

                if success:
                    # Recalculate functions and deltas for final solution
                    lower_func = logarithmic_function(lower_params['a'], lower_params['b'], lower_params['c'], x)
                    upper_func = logarithmic_function(upper_params['a'], upper_params['b'], upper_params['c'], x)
                    delta = upper_func - lower_func
                    fx = logarithmic_function(a_opt, b_opt, c_opt, x)

                    # Determine pointwise minimum and maximum bounds (lower/upper may cross)
                    lower_bound = np.minimum(lower_func, upper_func)
                    upper_bound = np.maximum(lower_func, upper_func)

                    # Allow 5% tolerance
                    tolerance = 0.05
                    cross_mask = delta <= 1e-2  # nearly equal or negative
                    adaptive_tol = np.ones_like(x) * tolerance
                    adaptive_tol[cross_mask] *= 3.0  # e.g., 3× wider around crossings
                    absolute_tol = 1e-2

                    lower_limit = lower_bound - np.maximum(absolute_tol, lower_bound * adaptive_tol)
                    upper_limit = upper_bound + np.maximum(absolute_tol, upper_bound * adaptive_tol)

                    # Vectorized check - only where both bounds lie within [0,1]
                    inside_bounds_mask = (lower_bound >= 0) & (upper_bound <= 1)
                    # Identify points to check
                    check_mask = inside_bounds_mask & ((fx < lower_limit) | (fx > upper_limit))

                    if np.any(check_mask):
                        idx = np.where(check_mask)[0][0]  # first error index for debugging
                        raise ValueError(
                            f"Generated function value fx[{idx}] = {fx[idx]:.4f} is outside the "
                            f"5% tolerance bounds [{lower_limit[idx]:.4f}, {upper_limit[idx]:.4f}] "
                            f"within the [0,1] validity range."
                        )

                    return {'a': a_opt, 'b': b_opt, 'c': c_opt}
                else:
                    raise ValueError(
                        "Generated function is outside 5% tolerance bounds."
                    )

    
    def reflect_exponential_function(a, b, c, d): 
        ''' Reflect the exponential function around x = 0.5.'''
        return {'a': a, 'b': -b, 'c': c * np.exp(b)}
 
    
    def calculate_solar_altitude(day, hour):
        '''
        Calculates the solar altitude angle for Brussels (Belgium) 
        '''
        timezone = 1
        longitude = 4.53
        latitude = 50.90
       
        delta = 23.45 * math.sin(math.radians((360 / 365) * (day + 284)))
        B = math.radians((360 / 365) * (day - 81))
        ET = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)
        standard_meridian = timezone * 15  # 15° per timezone
        LST = hour + (4 * (longitude - standard_meridian) + ET) / 60
        h = math.radians(15 * (LST - 12))
        phi = math.radians(latitude)
        delta = math.radians(delta)
        alpha = math.degrees(math.asin(math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.cos(h)))
            
        return alpha
        
    def generate_exponential_function(parameters):
        def f(x): 
            a, b, c = parameters['a'], parameters['b'], parameters['c']
            return a + c * np.exp(b * x)
        return f
            
 
    # ASSIGN VARIABLES
    
    # Step 1. Define the habits at family level
    # Step 1a. Define the family in relation to their light switching routines
    # Scale 1 to 5 with 5 being the most effective. 
    probs = np.loadtxt(directory_data +'/family_habits.txt', float)
    family_characterised = get_probability(np.random.random(), probs)
    # Step 1b. Define the family habits in relation to the moment they switch the lighting on and off
    # (0) (In)sufficient daylighting, (1) switching solar shading and (2) entering or leaving the room 
    probs = np.loadtxt(directory_data +'/reasons_on_living.txt', float)
    reason_on_living = get_probability(np.random.random(), probs)
    reasons_on_living = np.zeros(3)
    if reason_on_living in [1, 4, 5, 7]: 
        reasons_on_living[0] = 1
    if reason_on_living in [2, 4, 6, 7]: 
        reasons_on_living[1] = 1
    if reason_on_living in [3, 5, 6, 7]:
        reasons_on_living[2] = 1
    probs = np.loadtxt(directory_data+'/reasons_off_living.txt', float)
    reason_off_living = get_probability(np.random.random(), probs[:,reason_on_living - 1])
    reasons_off_living = np.zeros(3)
    if reason_off_living in [1, 4, 5, 7]: 
        reasons_off_living[0] = 1
    if reason_off_living in [2, 4, 6, 7]: 
        reasons_off_living[1] = 1
    if reason_off_living in [3, 5, 6, 7]:
        reasons_off_living[2] = 1
    # Step 1c. Define the family habits in relation to not switching off the lighting. 
    # (0) Consistent switching off, (1) forgetting, (2) laziness, (3) returning and (4) security_lighting. 
    probs = np.loadtxt(directory_data+'/not_switching_off.txt', float)
    reason_not_off = get_probability(np.random.random(), probs[:,family_characterised - 1])
    reasons_not_off = np.zeros(5)
    if reason_not_off in [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]: 
        reasons_not_off[0] = 1
    if reason_not_off in [2, 6, 10, 11, 12, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26]: 
        reasons_not_off[1] = 1
    if reason_not_off in [3, 7, 10, 13, 14, 16, 17, 19, 20, 23, 24, 26, 27, 28, 29]:
        reasons_not_off[2] = 1
    if reason_not_off in [4, 8, 11, 13, 15, 16, 18, 19, 21, 23, 25, 26, 27, 29, 30]: 
        reasons_not_off[3] = 1
    if reason_not_off in [5, 9, 12, 14, 15, 17, 18, 19, 22, 24, 25, 26, 28, 29, 30]: 
        reasons_not_off[4] = 1
    # Step 1d. Define the household in relation to their illuminance requirements. 
    # A value between 0 and 1, with 0 corresponding to low illuminance levels and 1 to high illuminances. 
    household_requirements = np.random.random()
    
    # Step 2. Define the habits at room level
    # Step 2a. When does the family switch on the ligthing in the different rooms? 
    # (0) Insufficient daylighting, (1) closing solar shading and (2) entering the room 
    # Bedroom
    probs = np.loadtxt(directory_data+'/reasons_on_bedroom.txt', float)
    reason_on_bedroom = get_probability(np.random.random(), probs[:,reason_on_living - 1])
    reasons_on_bedroom = np.zeros(3)
    if reason_on_bedroom in [1, 4, 5, 7]: 
        reasons_on_bedroom[0] = 1
    if reason_on_bedroom in [2, 4, 6, 7]: 
        reasons_on_bedroom[1] = 1
    if reason_on_bedroom in [3, 5, 6, 7]:
        reasons_on_bedroom[2] = 1
    # Bathroom
    probs = np.loadtxt(directory_data+'/reasons_on_bathroom.txt', float)
    reason_on_bathroom = get_probability(np.random.random(), probs[:,reason_on_living - 1])
    reasons_on_bathroom = np.zeros(3)
    if reason_on_bathroom in [1, 4, 5, 7]: 
        reasons_on_bathroom[0] = 1
    if reason_on_bathroom in [2, 4, 6, 7]: 
        reasons_on_bathroom[1] = 1
    if reason_on_bathroom in [3, 5, 6, 7]:
        reasons_on_bathroom[2] = 1
    # Toilet and storage
    probs = np.loadtxt(directory_data+'/reasons_on_toilet.txt', float)
    reason_on_toilet = get_probability(np.random.random(), probs[:,reason_on_living - 1])
    reasons_on_toilet = np.zeros(3)
    if reason_on_toilet in [1, 4, 5, 7]: 
        reasons_on_toilet[0] = 1
    if reason_on_toilet in [2, 4, 6, 7]: 
        reasons_on_toilet[1] = 1
    if reason_on_toilet in [3, 5, 6, 7]:
        reasons_on_toilet[2] = 1
    # Hallway
    probs = np.loadtxt(directory_data+'/reasons_on_hallway.txt', float)
    reason_on_hallway = get_probability(np.random.random(), probs[:,reason_on_living - 1])
    reasons_on_hallway = np.zeros(3)
    if reason_on_hallway in [1, 4, 5, 7]: 
        reasons_on_hallway[0] = 1
    if reason_on_hallway in [2, 4, 6, 7]: 
        reasons_on_hallway[1] = 1
    if reason_on_hallway in [3, 5, 6, 7]:
        reasons_on_hallway[2] = 1
    reasons_on = {'living': reasons_on_living, 'bathroom': reasons_on_bathroom, 'bedroom': reasons_on_bedroom, 'toilet': reasons_on_toilet, 'hallway': reasons_on_hallway}
    # Step 2b. When does the family switch off the lighting in the different rooms? 
    # (0) Insufficient daylighting, (1) opening solar shading and (2) leaving the room 
    # Bedroom
    probs = np.loadtxt(directory_data+'/reasons_off_bedroom.txt', float)
    reason_off_bedroom = get_probability(np.random.random(), probs[:,reason_off_living - 1])
    reasons_off_bedroom = np.zeros(3)
    if reason_off_bedroom in [1, 4, 5, 7]: 
        reasons_off_bedroom[0] = 1
    if reason_off_bedroom in [2, 4, 6, 7]: 
        reasons_off_bedroom[1] = 1
    if reason_off_bedroom in [3, 5, 6, 7]:
        reasons_off_bedroom[2] = 1
    # Bathroom
    probs = np.loadtxt(directory_data+'/reasons_off_bathroom.txt', float)
    reason_off_bathroom = get_probability(np.random.random(), probs[:,reason_off_living - 1])
    reasons_off_bathroom = np.zeros(3)
    if reason_off_bathroom in [1, 4, 5, 7]: 
        reasons_off_bathroom[0] = 1
    if reason_off_bathroom in [2, 4, 6, 7]: 
        reasons_off_bathroom[1] = 1
    if reason_off_bathroom in [3, 5, 6, 7]:
        reasons_off_bathroom[2] = 1
    # Toilet and storage
    probs = np.loadtxt(directory_data+'/reasons_off_toilet.txt', float)
    reason_off_toilet = get_probability(np.random.random(), probs[:,reason_off_living - 1])
    reasons_off_toilet = np.zeros(3)
    if reason_off_toilet in [1, 4, 5, 7]: 
        reasons_off_toilet[0] = 1
    if reason_off_toilet in [2, 4, 6, 7]: 
        reasons_off_toilet[1] = 1
    if reason_off_toilet in [3, 5, 6, 7]:
        reasons_off_toilet[2] = 1
    # Hallway
    probs = np.loadtxt(directory_data+'/reasons_off_hallway.txt', float)
    reason_off_hallway = get_probability(np.random.random(), probs[:,reason_off_living - 1])
    reasons_off_hallway = np.zeros(3)
    if reason_off_hallway in [1, 4, 5, 7]: 
        reasons_off_hallway[0] = 1
    if reason_off_hallway in [2, 4, 6, 7]: 
        reasons_off_hallway[1] = 1
    if reason_off_hallway in [3, 5, 6, 7]:
        reasons_off_hallway[2] = 1
    reasons_off = {'living': reasons_off_living, 'bathroom': reasons_off_bathroom, 'bedroom': reasons_off_bedroom, 'toilet': reasons_off_toilet, 'hallway': reasons_off_hallway}
    # Step 2c. How often do they forget to switch off the lighting? 
    # (1) Never, (2) several times per month, (3) several times per week, (4) multiple times per week, (5) daily
    # Living room/kitchen/office
    unnecessary_living = family_characterised
    # Bedroom
    probs = np.loadtxt(directory_data+'/unnecessary_bedroom.txt', float)
    unnecessary_bedroom = get_probability(np.random.random(), probs[:,unnecessary_living - 1])
    # Bathroom
    probs = np.loadtxt(directory_data+'/unnecessary_bathroom.txt', float)
    unnecessary_bathroom = get_probability(np.random.random(), probs[:,unnecessary_living - 1])
    # Toilet and storage
    probs = np.loadtxt(directory_data+'/unnecessary_toilet.txt', float)
    unnecessary_toilet = get_probability(np.random.random(), probs[:,unnecessary_living - 1])
    # Hallway
    probs = np.loadtxt(directory_data+'/unnecessary_hallway.txt', float)
    unnecessary_hallway = get_probability(np.random.random(), probs[:,unnecessary_living - 1])
    unnecessary = {'living': unnecessary_living, 'bedroom': unnecessary_bedroom, 'bathroom': unnecessary_bathroom, 'toilet': unnecessary_toilet, 'hallway': unnecessary_hallway}
    
    # Step 3. Set the probability function per critical moment and room. 
    # Define the function of the bedrooms: Which bedrooms are full-time used as office (and should be analysed as living room)? 
    asleep_bedroom_max = asleep_bedroom_dataframe.max(axis=0)
    offices_list = []
    bedrooms_list = []
    for bedroom in bedrooms: 
        if asleep_bedroom_max[bedroom] == 0: 
            offices_list.append(bedroom) 
        else: 
            bedrooms_list.append(bedroom)
    # Import the lower and upper probability functions. 
    on_entering_bounds = pd.read_csv(directory_data + '/on_entering.txt', delimiter='\t', dtype=str, comment='#', header = None)
    on_during_bounds = pd.read_csv(directory_data+'/on_during.txt', delimiter='\t', dtype=str, comment='#', header = None)
    off_leaving_bounds = pd.read_csv(directory_data+'/off_leaving.txt', delimiter='\t', dtype=str, comment='#', header = None)
    off_during_bounds = pd.read_csv(directory_data+'/off_during.txt', delimiter='\t', dtype=str, comment='#', header = None)
    probability_on_entering = {}
    probability_on_during = {}
    probability_off_leaving = {}
    probability_off_during = {}
    probability_on_solar = {}
    probability_off_solar = {}
    probability_on_leaving_occupation = {}
    probability_off_leaving_occupation = {}
    probability_on_entering_occupation = {}
    probability_off_entering_occupation = {}
    probability_off_entering = {}
    probability_on_leaving = {}
    probability_off_sleeping = {}
    # Repeat the following steps for every room/zone in the simulation. 
    for zone_name in rooms_lighting: 
        # Define the type of room and corresponding index
        # (1) Bathroom, (2) Bedroom, (3) Hallway, (4) Kitchen, (5) Living, (6) Office and (7) Toilet, Storage
        if 'Bathroom' in zone_name: 
            room_type = 1
            room_habits = 'bathroom'
        elif zone_name in bedrooms_list: 
            room_type = 2
            room_habits = 'bedroom'
        elif 'Corridor' in zone_name or 'Hallway' in zone_name: 
            room_type = 3
            room_habits = 'hallway'
        elif 'Kitchen' in zone_name and not 'Living' in zone_name: 
            room_type = 4
            room_habits = 'living'
        elif 'Living' in zone_name and not 'Kitchen' in zone_name: 
            room_type = 5
            room_habits = 'living'
        elif zone_name in offices_list: 
            room_type = 6
            room_habits = 'living'
        elif 'Toilet' in zone_name or 'Storage' in zone_name: 
            room_type = 7
            room_habits = 'toilet'
        elif 'LivingKitchen' in zone_name:
            # Assumption: the gathered data for Living and Kitchen are combined and the borders are set in to the extrema. 
            room_type = 8
            room_habits = 'living'
        # Step 3a. Link the survey data with the measurement campaign. 
        # Switch-on probabilities
        # When the first occupant enters the room
        # Assumption: If 'entering the room' is included in the household routines, the a constraint for the switching-on probability interval of 40-100% is formulated. If not, the constraint is set to 0-60%. 
        if reasons_on[room_habits][2] == 0: 
            interval_on_entering = (0,0.6)
        else:
            interval_on_entering = (0.4,1)  
        # Assumption: The steepness of the graph is formulated in relation to the extent of effectiveness in responding to triggers.
        if family_characterised == 1: 
            steepness_on = (0,0.25)
        elif family_characterised == 2: 
            steepness_on = (0.15,0.45)
        elif family_characterised == 3: 
            steepness_on = (0.35,0.65)
        elif family_characterised == 4: 
            steepness_on = (0.55,0.85)
        elif family_characterised == 5: 
            steepness_on = (0.75,1)
        # During occupantion
        # Assumption: The habit of switching the lighting on in function of the daylighting availability is evolved; if this trigger is included for the household, a constraint of 40-100% of the interval is formulated. If not, the constraint is set to 0-60%. 
        if reasons_on[room_habits][0] == 0: 
            interval_on_during = (0,0.6)
        else:
            interval_on_during = (0.4,1)
        # Assumption: The steepness of the graph is formulated in relation to the extent of effectiveness in responding to triggers.
        # These values are already set above by entering the room. 
        # Switch-off probabilities
        # Constant fixing the time interval during which the lighting remains on during between two visits
        # Minutes, rounded at 2 to match the time steps. 
        probs = np.loadtxt(directory_data+'/interval_lighton.txt', float)
        interval_lighton = get_probability_interpolate(np.random.random(), probs[:,[0, family_characterised]])
        # Last occupant leaving the room
        # Assumption: If 'leaving the room' is included in the household routines, the a constraint for the switching-off probability interval of 40-100% is formulated. If not, the constraint is set to 0-60%.    
        if reasons_off[room_habits][2] == 0: 
            interval_off_leaving = (0,0.6)
        else:
            interval_off_leaving = (0.4,1)     
        # Assumption: The average number of times that an household forgets to switch off the lighting is as follows included: the interval is set to 75-100% for an indication of never, 55-85% for several times per month, 35-64% for a several times per week, 15-45% for multiple times per week and 0-25% for daily. 
        # This is used as a correction on the formulated interval above.
        interval_lower, interval_upper = interval_off_leaving
        if unnecessary[room_habits] == 1: 
            interval_off_leaving = (interval_lower + 0.75 * (interval_upper-interval_lower), interval_lower + 1*(interval_upper-interval_lower))
        elif unnecessary[room_habits] == 2: 
            interval_off_leaving = (interval_lower + 0.55 * (interval_upper-interval_lower), interval_lower + 0.85*(interval_upper-interval_lower))
        elif unnecessary[room_habits] == 3: 
            interval_off_leaving = (interval_lower + 0.35 * (interval_upper-interval_lower), interval_lower + 0.65*(interval_upper-interval_lower))
        elif unnecessary[room_habits] == 4: 
            interval_off_leaving = (interval_lower + 0.15 * (interval_upper-interval_lower), interval_lower + 0.45*(interval_upper-interval_lower))
        else:  
            interval_off_leaving = (interval_lower + 0 * (interval_upper-interval_lower), interval_lower + 0.25*(interval_upper-interval_lower))
        # Assumption: A target point is set in accordance to the generated time interval. This target point corresponds to the inflection point of the probability function and should be matched within 10%. 
        target_time = interval_lighton
        # During occupantion
        # Assumption: The habit of switching the lighting off in function of the daylighting availability is evolved; if this trigger is included for the household, a constraint of 40-100% of the interval is formulated. If not, the constraint is set to 0-60%. 
        if reasons_off[room_habits][0] == 0: 
            interval_off_during = (0,0.6)
        else:
            interval_off_during = (0.4,1)
        # Step 3b. Define the relations for switching on. 
        # First occupant entering the room (in function of indoor illuminance)
        on_entering_lower = on_entering_bounds.iloc[room_type - 1,0]
        on_entering_upper = on_entering_bounds.iloc[room_type - 1,1]
        probability_on_entering[zone_name] = generate_random_exponential_probability_function(on_entering_lower, on_entering_upper, interval_on_entering, steepness_on, increasing = False)
        # During occupation (in function of indoor illuminance)
        on_during_lower = on_during_bounds.iloc[room_type - 1,0]
        on_during_upper = on_during_bounds.iloc[room_type - 1,1]
        probability_on_during[zone_name] = generate_random_exponential_probability_function(on_during_lower, on_during_upper, interval_on_during, steepness_on, increasing = False) 
        # Step 3c. Define the relations for switching off. 
        # Last occupant leaving the room (in function of time until next occupation)
        off_leaving_lower = off_leaving_bounds.iloc[room_type - 1,0]
        off_leaving_upper = off_leaving_bounds.iloc[room_type - 1,1]
        probability_off_leaving[zone_name] = generate_random_logarithmic_probability_function(off_leaving_lower, off_leaving_upper, target_time, interval_off_leaving)
        # During occupation (in function of indoor illuminance)
        off_during_lower = off_during_bounds.iloc[room_type - 1,0]
        off_during_upper = off_during_bounds.iloc[room_type - 1,1]
        probability_off_during[zone_name] = generate_random_exponential_probability_function(off_during_lower, off_during_upper, interval_off_during)
        # Step 3d. Formulate probability functions for additional (critical) moments (based on assumptions).
        # Critical moment: switching solar shading
        if reasons_off[room_habits][1] == 1: 
            # Assumption: the eyes don't have the time to gradually adapt to the darkness and the person in moving already (however, it can still require effort to go the the switch). Since both have an opposite effect, it is assumed that this corresponds to the probability of entering. 
            probability_on_solar[zone_name] = probability_on_entering[zone_name]
            # Assumption: Opening the solar shading results in a probability to switch off the lighting that is increased in comparison to during occupation. It is assumed that this probability is increasing with increasing indoor illuminance and is located between the off-probability for during + 15% and the mirrored on-probability (around y = 0.5) - 15%.  
            probability_off_solar[zone_name] = generate_random_exponential_probability_function(probability_off_during[zone_name], reflect_exponential_function(**probability_on_entering[zone_name]), (0.15,0.85))
        else: 
            probability_on_solar[zone_name] = 0
            probability_off_solar[zone_name] = 0
        # Critical moment: first occupant enters the room after the lighting was not switched off. 
        # Assumption: The probability to switch off the lighting when entering as first is assumed to be inbetween probability_off_entering_occupation and the reflected probability_on_entering. An interval of (0,0.5) is proposed to set the probability function. 
        if occupancy_dataframe['Building'].max() > 1: 
            probability_off_entering[zone_name] = generate_random_exponential_probability_function(probability_off_during[zone_name], reflect_exponential_function(**probability_on_entering[zone_name]), (0,0.5))
        else: 
            probability_lower = generate_random_exponential_probability_function(probability_off_during[zone_name], reflect_exponential_function(**probability_on_entering[zone_name]), (0,0.5))
            point_constrain = min(generate_exponential_function(probability_off_during[zone_name])(0), generate_exponential_function(reflect_exponential_function(**probability_on_entering[zone_name]))(0))
            probability_off_entering[zone_name] = generate_random_exponential_probability_function(probability_lower, reflect_exponential_function(**probability_on_entering[zone_name]), interval_range = (0,0.5), constrain_point =(0, point_constrain))
        # Critical moment: additional occupant entering the room
        if occupancy_dataframe['Building'].max() > 1: 
            # Assumption: this person interacts independently. 
            probability_on_entering_occupation[zone_name] = probability_on_entering[zone_name]
            # Assumption: the occupant is expected to interact more effectively than during occupation. It is assumed that this probability is increasing with increasing indoor illuminance and is located between the off-probability for during + 15% and the off probability when entering - 15%.  
            point_constrain = min(generate_exponential_function(probability_off_during[zone_name])(0), generate_exponential_function(reflect_exponential_function(**probability_on_entering[zone_name]))(0))
            probability_off_entering_occupation[zone_name] = generate_random_exponential_probability_function(probability_off_during[zone_name], probability_off_entering[zone_name], interval_range =  (0.15,0.85), constrain_point =(0,point_constrain))
        # Critical moment: occupant leaving the room while at least one person stays in the room
        if occupancy_dataframe['Building'].max() > 1: 
            # Assumption: the probability is located between the probability when entering the room as first and interacting during occupation, with the difference that this person is now not acting for themself. Therefore, a probability function in the lowest quart of the interval between both functions is suggested.  
            probability_on_leaving_occupation[zone_name] = generate_random_exponential_probability_function(probability_on_during[zone_name], probability_on_entering[zone_name], (0,0.25), increasing = False)
            # Assumption: an occupant will occasionally switch off the lighting when leaving as this requires limited additional effort, but affects the comfort of his household members. It is assumed that this probability is increasing with increasing indoor illuminance and is located between the off-probability for during + 15% and the mirrored on-probability (around y = 0.5) - 15%.  
            point_constrain = min(generate_exponential_function(probability_off_during[zone_name])(0), generate_exponential_function(reflect_exponential_function(**probability_on_entering[zone_name]))(0))
            probability_off_leaving_occupation[zone_name] = generate_random_exponential_probability_function(probability_off_during[zone_name], reflect_exponential_function(**probability_on_entering[zone_name]), interval_range = (0.15,0.85), constrain_point = (0,point_constrain))
        # Critical moment: last occupant leaves the room when the lighting is switched off. 
        # Assumption: This person will not switch on the lighting. 
        probability_on_leaving[zone_name] = {'a': 0, 'b': 0, 'c': 0, 'd': 0}
        # Occupant is going to sleep. 
        # Assumption: lighting will be switched off between 95 and 99.99% of the occurances. 
        if 'Bedroom' in zone_name: 
            probability_off = random.uniform(0.95, 0.9999)
            probability_off_sleeping[zone_name] = {'a': 0, 'b': 0, 'c': 0, 'd': probability_off}
    # Step 3d. Set the average standard deviation 
    standard_deviation = np.loadtxt(directory_data+'/correction_deviation.txt')
        
    # Step 4. Define the household preferences in relation to illuminance
    # Step 4a. Generate a dictionary containing the desired illuminance per room and per timestep. 
    # rows: (1) Cooking, dishes, computer and office work, (2) Cleaning and ironing, (3) Personal hygiene and laundry and (4) Entertainment and passageways
    illuminance_ranges = np.loadtxt(directory_data +'/illuminance_ranges.txt')
    # Set the illuminance thresholds per activity and per room. A 5% variation is allowed on the family constant. 
    def interpolate_illuminance(illuminance_lower, illuminance_upper, household_requirements): 
        household_requirements_adjusted = household_requirements + random.uniform(-household_requirements * 0.05, household_requirements * 0.05)
        return (1 - household_requirements_adjusted) * illuminance_lower + household_requirements_adjusted * illuminance_upper
    # Kitchen
    illuminance_kitchen_cooking = interpolate_illuminance(illuminance_ranges[0,0], illuminance_ranges[0,1], household_requirements)
    illuminance_kitchen_dishes = interpolate_illuminance(illuminance_ranges[0,0], illuminance_ranges[0,1], household_requirements)
    illuminance_kitchen_else = interpolate_illuminance(illuminance_ranges[3,0], illuminance_ranges[3,1], household_requirements)
    # Living
    illuminance_living_pc = interpolate_illuminance(illuminance_ranges[0,0], illuminance_ranges[0,1], household_requirements)
    illuminance_living_adm = interpolate_illuminance(illuminance_ranges[0,0], illuminance_ranges[0,1], household_requirements)
    illuminance_living_vacuum = interpolate_illuminance(illuminance_ranges[1,0], illuminance_ranges[1,1], household_requirements)
    illuminance_living_iron = interpolate_illuminance(illuminance_ranges[1,0], illuminance_ranges[1,1], household_requirements)
    illuminance_living_else = interpolate_illuminance(illuminance_ranges[3,0], illuminance_ranges[3,1], household_requirements)
    # Bedroom 1
    illuminance_bedroom1_pc = interpolate_illuminance(illuminance_ranges[0,0], illuminance_ranges[0,1], household_requirements)
    illuminance_bedroom1_adm = interpolate_illuminance(illuminance_ranges[0,0], illuminance_ranges[0,1], household_requirements)
    illuminance_bedroom1_else = interpolate_illuminance(illuminance_ranges[3,0], illuminance_ranges[3,1], household_requirements)
    # Bedroom 2
    illuminance_bedroom2_pc = interpolate_illuminance(illuminance_ranges[0,0], illuminance_ranges[0,1], household_requirements)
    illuminance_bedroom2_adm = interpolate_illuminance(illuminance_ranges[0,0], illuminance_ranges[0,1], household_requirements)
    illuminance_bedroom2_else = interpolate_illuminance(illuminance_ranges[3,0], illuminance_ranges[3,1], household_requirements)
    # Bedroom 3
    illuminance_bedroom3_pc = interpolate_illuminance(illuminance_ranges[0,0], illuminance_ranges[0,1], household_requirements)
    illuminance_bedroom3_adm = interpolate_illuminance(illuminance_ranges[0,0], illuminance_ranges[0,1], household_requirements)
    illuminance_bedroom3_else = interpolate_illuminance(illuminance_ranges[3,0], illuminance_ranges[3,1], household_requirements)
    # Corridor
    illuminance_corridor = interpolate_illuminance(illuminance_ranges[3,0], illuminance_ranges[3,1], household_requirements)
    # Bathroom
    illuminance_bathroom = interpolate_illuminance(illuminance_ranges[2,0], illuminance_ranges[2,1], household_requirements)
    # Storage
    illuminance_storage = interpolate_illuminance(illuminance_ranges[2,0], illuminance_ranges[2,1], household_requirements)   
    # ToiletGround
    illuminance_toiletground = interpolate_illuminance(illuminance_ranges[3,0], illuminance_ranges[3,1], household_requirements)
    # ToiletFirst
    illuminance_toiletfirst = interpolate_illuminance(illuminance_ranges[3,0], illuminance_ranges[3,1], household_requirements)
    
    # Step 4b. Analyse activities and tasks in relation to the required minimum illuminance per room and per timestep
    minimum_illuminance = {'Living':{}, 'Kitchen':{}, 'LivingKitchen': {}, 'Bedroom1':{}, 'Bedroom2':{}, 'Bedroom3':{}, 'Corridor':{}, 'Bathroom':{}, 'Storage': {}, 'ToiletGround': {}, 'ToiletFirst': {}}
    for i in range(task_activities_dataframe.shape[0]): 
        # Kitchen
        if task_activities['cook'][i] > 0: 
            minimum_illuminance['Kitchen'].update({i:illuminance_kitchen_cooking})
        elif task_activities['dishes'][i] > 0:
            minimum_illuminance['Kitchen'].update({i:illuminance_kitchen_dishes})
        else:
            minimum_illuminance['Kitchen'].update({i:illuminance_kitchen_else})
        # Living
        if task_activities['pcDayzone'][i] > 0: 
            minimum_illuminance['Living'].update({i:illuminance_living_pc})
        elif task_activities['admDayzone'][i] > 0: 
            minimum_illuminance['Living'].update({i:illuminance_living_adm})
        elif task_activities['vacuum'][i] > 0:
            minimum_illuminance['Living'].update({i:illuminance_living_vacuum})
        elif task_activities['iron'][i] > 0: 
            minimum_illuminance['Living'].update({i:illuminance_living_iron})
        else: 
            minimum_illuminance['Living'].update({i:illuminance_living_else})
        # LivingKitchen
        minimum_illuminance['LivingKitchen'].update({i: max(minimum_illuminance['Living'].get(i, 0), minimum_illuminance['Kitchen'].get(i,0))})
        # Bedroom1
        if task_activities['pcBedroom1'][i] > 0: 
            minimum_illuminance['Bedroom1'].update({i:illuminance_bedroom1_pc})
        elif task_activities['admBedroom1'][i] > 0: 
            minimum_illuminance['Bedroom1'].update({i:illuminance_bedroom1_adm})
        else: 
            minimum_illuminance['Bedroom1'].update({i:illuminance_bedroom1_else})
        # Bedroom2
        if task_activities['pcBedroom2'][i] > 0: 
            minimum_illuminance['Bedroom2'].update({i:illuminance_bedroom2_pc})
        elif task_activities['admBedroom2'][i] > 0: 
            minimum_illuminance['Bedroom2'].update({i:illuminance_bedroom2_adm})
        else: 
            minimum_illuminance['Bedroom2'].update({i:illuminance_bedroom2_else})
        # Bedroom3
        if task_activities['pcBedroom3'][i] > 0: 
            minimum_illuminance['Bedroom3'].update({i:illuminance_bedroom3_pc})
        elif task_activities['admBedroom3'][i] > 0: 
            minimum_illuminance['Bedroom3'].update({i:illuminance_bedroom3_adm})
        else: 
            minimum_illuminance['Bedroom3'].update({i:illuminance_bedroom3_else})
        # Corridor
        minimum_illuminance['Corridor'].update({i:illuminance_corridor})
        # Bathroom
        minimum_illuminance['Bathroom'].update({i:illuminance_bathroom})
        # Storage
        minimum_illuminance['Storage'].update({i:illuminance_storage})
        #ToiletGround
        minimum_illuminance['ToiletGround'].update({i:illuminance_toiletground})
        # ToiletFirst
        minimum_illuminance['ToiletFirst'].update({i:illuminance_toiletfirst})
        
    # Step 5. Generate the habits that are used to model the practical habits
    # Stap 5a. Define 'security_lighting lighting' for the household
    # Assumption: Light will be switched on in during periods of absence and darkness
    # Rooms
    security_lighting_rooms_overview = ['Living', 'Kitchen', 'Corridor']
    security_lighting_rooms = random.sample(security_lighting_rooms_overview, random.randint(1, len(security_lighting_rooms_overview)))
    # Habits
    security_lighting_habits_overview = ['dark and absence', 'dark and absence/asleep', 'anticipate dark and absence', 'anticipate dark and absence/asleep']
    security_lighting_habits = random.choice(security_lighting_habits_overview)
    # Step 5b. Calculate timestep sunrise and sunset per day. 
    sunset = {}
    sunrise = {}
    for day in range(1,366): 
        timestep = 0
        while calculate_solar_altitude(day, timestep) > 0 or calculate_solar_altitude(day, timestep + 1/30) < 0: # 2-minute timesteps
            timestep += 1/30
        sunrise[day] = timestep - 1/30
        while calculate_solar_altitude(day, timestep) < 0 or calculate_solar_altitude(day, timestep + 1/30) > 0: # 2-minute timesteps
            timestep += 1/30
        sunset[day] = timestep + 1/30
        
    # Step 5c. Probability habits
    # Assumption: There is a 0-10% chance the inhabitants do not follow their habits (e.g. they are returning later than expected). 
    probability_security_lighting = random.uniform(0.90, 1.00)
                   
    # CONTROL PROGRAM
    
    def on_begin_timestep_before_predictor(self,state) -> int: 

        def correcting_lighting(zone, timestep, hour, day, sunrise, sunset, occupancy, lighting_requirements, asleep, illuminance, previousstate_lighting) -> float: 
            
            # Correcting factor dusk: daylighting illuminance is zero during this period
            # Assumption: during dusk times people are more/less likely to switch their lighting
            # Average duration of civil dusk in Brussels: 36 minutes [source: Koninklijke sterrenwacht van België]
            average_dusk = 0.6 # Hour
            if hour <= sunrise[day] - average_dusk or hour >= sunset[day] + average_dusk: 
                if previousstate_lighting == 0:
                    # More likely to switch lighting on                
                    k_dusk = 1
                else:
                    # Less likely to switch lighting off
                    k_dusk = - 1
            elif sunrise[day] - average_dusk < hour < sunrise[day]:
                # Assumption: Linear interpolation between first dusk and sunrise
                if previousstate_lighting == 0: 
                    # Less likely to switch lighting on
                    k_dusk = (sunrise[day]-hour)/average_dusk 
                else: 
                    # More likely to switch lighting off
                    k_dusk = (sunrise[day]-hour)/average_dusk - 1
            elif sunset[day] < hour < sunset[day] + average_dusk: 
                # Assumption: Linear interpolation between sunset and last dusk
                if previousstate_lighting == 0: 
                    # More likely to switch lighting on
                    k_dusk = (hour-sunset[day])/average_dusk
                else: 
                    # Less likely to switch lighting off
                    k_dusk = -(hour-sunset[day])/average_dusk
            else: 
                k_dusk = 0
            
            
            # Correcting factor during - occupancy
            if lighting_requirements[zone][timestep -1] > 0 and lighting_requirements[zone][timestep + 1] > 0:
                # Assumption: The probability of interacting increases with the period of expected occupation
                i = 1
                while lighting_requirements[zone][timestep + i] == 1 and timestep + i <= 262800 and i <= 10: 
                    i += 1
                # Assumption: the probability is interpolated between -1 and 0 according to the time until leaving with a maximum of 20 minutes.  
                k_occpros = i/10 - 1
                # Assumption: The occupant is less likely to interact right after entering the room, followed by a fast increase in probability and a gradual decrease as the eyes adapt to the environment. 
                i = 1
                while lighting_requirements[zone][timestep - i] == 1 and timestep - i > 0 and i <= 10: 
                    i += 1
                # Assumption: the probability is interpolated between -1 and 1 according to the time after entering (8 minutes) and afterwards between 1 and 0 with a maximum of 20 minutes.
                if i <= 4: 
                    k_occprev = i/2 - 1
                else: 
                    k_occprev = 1 - (i-4)/12
                k_occ = min(k_occpros, k_occprev)
            else: 
                k_occ = 0
            # Assumption: the chances increases when the person just wakes up
            if 'Bedroom' in zone: 
                if occupancy[zone][timestep -1] == asleep[zone][timestep-1] and occupancy[zone][timestep] > asleep[zone][timestep]:
                    k_occ = 1
            
            # Covering the 95% interval of the normal distribution
            # Assumption: dusk and occupancy are considered equally during decision-making. 
            k = (0.5*k_dusk+ 0.5*k_occ)*2
            
            return k
        
        def stochastic_switching(probability_situation: float, previousstate) -> int:
            ''' 
            Simulates a stochastic state switch based on a given probability.
            '''
            if not (0 <= probability_situation <= 1): 
                if probability_situation < 0: 
                    probability_situation = 0
                elif probability_situation > 1: 
                    probability_situation = 1
            chance = np.random.random()
            if chance <= probability_situation: # switching the lighting
                return 1 - previousstate
            else: 
                return previousstate
            
        def generate_logarithmic_function(parameters): 
            '''
            Generate the logarithmic probability function. 
            '''
            def f(x):
                a, b, c = parameters['a'], parameters['b'], parameters['c']
                return a + b * np.log(x + c)
            return f
            
        def horizontal_dilation(parameters, minimum_illuminance): 
            '''
            Scales the probability function in relation to the minimum illuminance that is required during the timestep.
            
            '''
            def f(x): 
                a, b, c = parameters['a'], parameters['b'], parameters['c']
                return a + c * np.exp(b * 1/minimum_illuminance * x)
            return f

        hour = round(self.api.exchange.current_time(state),2)
        day = self.api.exchange.day_of_year(state)
        timestep_simulation = self.api.exchange.zone_time_step(state)
        timestep = round((day-1)*720+(hour/0.033333)-1) # simulation runs at two minute timesteps
        sun_up = self.api.exchange.sun_is_up(state)
        
        shading_interaction = 0 # Depends on the shading interaction during a particular timestep

        previousstate_lighting = {}
        illuminance={}
        for zone_name in self.rooms_lighting: 
            
            # Exchange input information with EnergyPlus
            previousstate_lighting['PreviousStateLighting{}'.format(zone_name)] = self.api.exchange.get_global_value(state, self.api.exchange.get_global_handle(state, 'PreviousStateLighting'+zone_name))
            if zone_name in self.rooms_daylighting:
                illuminance['Illuminance{}'.format(zone_name)] = self.api.exchange.get_variable_value(state, self.api.exchange.get_variable_handle(state, 'Daylighting Reference Point 1 Illuminance', 'DaylightingControl'+zone_name))
            elif zone_name == self.rooms_daylighting_2: # Average of two Daylighting Reference Points
                illuminance['Illuminance{}'.format(zone_name)] = (self.api.exchange.get_variable_value(state, self.api.exchange.get_variable_handle(state, 'Daylighting Reference Point 1 Illuminance', 'DaylightingControl'+zone_name))+self.api.exchange.get_variable_value(state, self.api.exchange.get_variable_handle(state, 'Daylighting Reference Point 2 Illuminance', 'DaylightingControl'+zone_name)))/2
            
            state_lighting = 0
            
            # # EVALUATION PRACTICAL REASONS (I.E. security_lighting)
            
            # Household shows a habit to switch on lighting for security_lighting reasons
            if self.reasons_not_off[4] == 1: 
                # security_lighting habits are only maintained in a limited selection of the rooms. 
                if zone_name in self.security_lighting_rooms or (zone_name == 'LivingKitchen' and ('Living' in self.security_lighting_rooms or 'Kitchen' in self.security_lighting_rooms)):
                    if timestep-1 >= 0:
                        # Switching lighting on
                        if previousstate_lighting['PreviousStateLighting{}'.format(zone_name)] == 0: 
                            # Assumption: lighting will be switched on during periods of absence when it is dark. 
                            if self.security_lighting_habits == 'dark and absence': 
                                # Last person has left the building while it is dark
                                if sun_up == False and self.occupancy['Building'][timestep] == 0 and self.occupancy['Building'][timestep-1] > 0: 
                                    state_lighting = stochastic_switching(self.probability_security_lighting, 0)
                            elif self.security_lighting_habits == 'dark and absence/asleep': 
                            # Assumption: lighting will be switched on during periods of absence and when all occupants are in their bedrooms. 
                                occupancy_bedrooms = 0
                                occupancy_bedrooms_previous = 0
                                for bedroom in self.bedrooms_list: 
                                    occupancy_bedrooms += self.occupancy[bedroom][timestep]
                                    occupancy_bedrooms_previous += self.occupancy[bedroom][timestep-1]
                                if sun_up == False and self.occupancy['Building'][timestep] == 0 and self.occupancy['Building'][timestep-1] > 0: 
                                    state_lighting = stochastic_switching(self.probability_security_lighting, 0) 
                                elif sun_up == False and self.occupancy['Building'][timestep] == occupancy_bedrooms and self.occupancy['Building'][timestep-1] > occupancy_bedrooms_previous: 
                                    state_lighting = stochastic_switching(self.probability_security_lighting, 0)
                            elif self.security_lighting_habits == 'anticipate dark and absence': 
                            # Assumption: lighting will be switched on during periods of absence when it is dark, the occupants also anticipate when leaving the house while the sun is still up. 
                                # Last person leaves the building
                                if self.occupancy['Building'][timestep] == 0 and self.occupancy['Building'][timestep-1] > 0: 
                                    # Dark when leaving
                                    if sun_up == False: 
                                        state_lighting = stochastic_switching(self.probability_security_lighting, 0)
                                    else: 
                                        # Check whether they return before darkness
                                        i = timestep + 1
                                        while self.occupancy['Building'][i] == 0 and (i - (day - 1)*24*30)/30 < self.sunset[day]: 
                                            i += 1
                                        if (i - (day - 1)*24*30)/30 >= self.sunset[day]: 
                                            state_lighting = stochastic_switching(self.probability_security_lighting, 0)
                            elif self.security_lighting_habits == 'anticipate dark and absence/asleep': 
                            # Assumption: lighting will be switched on during periods of absence and when all occupants are in their bedrooms. The occupants also anticipate when leaving the house while the sun is still up.
                                occupancy_bedrooms = 0
                                occupancy_bedrooms_previous = 0
                                for bedroom in self.bedrooms_list: 
                                    occupancy_bedrooms += self.occupancy[bedroom][timestep]
                                    occupancy_bedrooms_previous += self.occupancy[bedroom][timestep-1]                                    
                                # Last person leaves the building
                                if self.occupancy['Building'][timestep] == 0 and self.occupancy['Building'][timestep-1] > 0: 
                                    # Dark when leaving
                                    if sun_up == False: 
                                        state_lighting = stochastic_switching(self.probability_security_lighting, 0)
                                    else: 
                                        # Check whether they return before darkness
                                        i = timestep + 1
                                        while self.occupancy['Building'][i] == 0 and (i - (day - 1)*24*30)/30 < self.sunset[day]: 
                                            i += 1
                                        if (i - (day - 1)*24*30)/30 >= self.sunset[day]: 
                                            state_lighting = stochastic_switching(self.probability_security_lighting, 0)
                                # Last occupant going to sleep
                                elif self.occupancy['Building'][timestep] == occupancy_bedrooms and self.occupancy['Building'][timestep-1] > occupancy_bedrooms_previous: 
                                    if sun_up == False: 
                                        state_lighting = stochastic_switching(self.probability_security_lighting, 0)
                                    else: 
                                        i = timestep + 1
                                        while self.occupancy['Building'][i] == occupancy_bedrooms and (i - (day - 1)*24*30)/30 < self.sunset[day]: 
                                            i += 1
                                        if (i - (day - 1)*24*30)/30 >= self.sunset[day]: 
                                            state_lighting = stochastic_switching(self.probability_security_lighting, 0)
                        # Switching lighting off
                        else: 
                            # First occupant coming home
                            if self.occupancy['Building'][timestep - 1] == 0 and self.occupancy['Building'][timestep] > 0:
                                # Nobody present in the room: lighting is switched off with a probability of 95% (probability_security_lighting). 
                                if self.occupancy[zone_name][timestep] == 0: 
                                    state_lighting = stochastic_switching(self.probability_security_lighting, 1)
                                    previousstate_lighting['PreviousStateLighting{}'.format(zone_name)] = state_lighting
                            # First occupant leaving the bedroom
                            elif self.security_lighting_habits in ['dark and absence/asleep', 'anticipate dark and absence/asleep']: 
                                occupancy_bedrooms = 0
                                occupancy_bedrooms_previous = 0
                                for bedroom in self.bedrooms_list: 
                                    occupancy_bedrooms += self.occupancy[bedroom][timestep]
                                    occupancy_bedrooms_previous += self.occupancy[bedroom][timestep-1]       
                                if self.occupancy['Building'][timestep] > occupancy_bedrooms and self.occupancy['Building'][timestep-1] == occupancy_bedrooms_previous:
                                    # Nobody present in the room: lighting is switched off with a probability of 95% (probability_security_lighting). 
                                    if self.occupancy[zone_name][timestep] == 0: 
                                        state_lighting = stochastic_switching(self.probability_security_lighting, 1)
                                        previousstate_lighting['PreviousStateLighting{}'.format(zone_name)] = state_lighting

            # EVALUATION PSYCHOLOGICAL REASONS
            
            # Evaluation only runs when lighting is not switched on due to security_lighting reasons
            if not state_lighting == 1:
                # Light ON during the previous timestep
                if previousstate_lighting['PreviousStateLighting{}'.format(zone_name)] == 1:
                    
                    # Nobody present in the room
                    if self.lighting_requirements[zone_name][timestep] == 0: 
                        
                        # Check for critical moments
                        if timestep-1 >= 0: 
                            # Last occupant has left the room since last timestep
                            if self.lighting_requirements[zone_name][timestep-1] > 0:              
                            
                                # Calculate the period of time until an occupant returns
                                time_next_occupancy = 2
                                index = timestep + 1
                                while self.lighting_requirements[zone_name][index] == 0 and index < 262799: 
                                    index += 1
                                    time_next_occupancy += 2
                                # And evaluate whether the occupant will switch off the lighting when leaving 
                                k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting['PreviousStateLighting{}'.format(zone_name)])
                                probability_timestep = generate_logarithmic_function(self.probability_off_leaving[zone_name])(time_next_occupancy) + k*self.standard_deviation
                                state_lighting = stochastic_switching(probability_timestep, previousstate_lighting['PreviousStateLighting{}'.format(zone_name)])
                            # Nobody present in the room, the lighting remains on  
                            else: 
                                state_lighting = previousstate_lighting['PreviousStateLighting{}'.format(zone_name)]
                        else: 
                            state_lighting = previousstate_lighting['PreviousStateLighting{}'.format(zone_name)]
            
                    #Occupants present in the room
                    else:         
                    
                        # Check for critical moments
                        if timestep-1 >= 0:
                            # The occupants have interacted with the solar shading installation
                            probability_timestep_shading = 0
                            if shading_interaction == 1: 
                                k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting['PreviousStateLighting{}'.format(zone_name)])
                                illuminance_timestep = illuminance['Illuminance{}'.format(zone_name)]
                                probability_timestep_shading = horizontal_dilation((self.probability_off_solar), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation
                            # The first occupant has entered the room 
                            if self.lighting_requirements[zone_name][timestep-1] == 0:
                                # Rooms without sufficient daylight entrance 
                                if zone_name in self.rooms_without_daylighting:
                                    illuminance_timestep = 0
                                    probability_timestep = max(probability_timestep_shading, horizontal_dilation((self.probability_off_entering[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep))
                                # Rooms with sufficient daylight entrance and presence pattern defined in Occupancy
                                else: 
                                    k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting['PreviousStateLighting{}'.format(zone_name)])
                                    illuminance_timestep = illuminance['Illuminance{}'.format(zone_name)]
                                    probability_timestep = max(probability_timestep_shading, horizontal_dilation((self.probability_off_entering[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation)
                            # An additional occupant has entered the room
                            elif zone_name in self.occupancy_dataframe.columns and self.occupancy[zone_name][timestep] > self.occupancy[zone_name][timestep-1]: 
                                k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting['PreviousStateLighting{}'.format(zone_name)])
                                illuminance_timestep = illuminance['Illuminance{}'.format(zone_name)]
                                probability_timestep = max(probability_timestep_shading,horizontal_dilation((self.probability_off_entering_occupation[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation)
                            # An occupant has left the room
                            elif zone_name in self.occupancy_dataframe.columns and self.occupancy[zone_name][timestep] < self.occupancy[zone_name][timestep-1]: 
                                k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting['PreviousStateLighting{}'.format(zone_name)])
                                illuminance_timestep = illuminance['Illuminance{}'.format(zone_name)]
                                probability_timestep = max(probability_timestep_shading,horizontal_dilation((self.probability_off_leaving_occupation[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation)
                            # No external critical moment detected 
                            else: 
                                # Rooms without sufficient daylight entrance
                                if zone_name in self.rooms_without_daylighting: 
                                    illuminance_timestep = 0
                                    probability_timestep = max(probability_timestep_shading,horizontal_dilation((self.probability_off_during[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep))
                                # Rooms, except bedrooms, with sufficient daylight entrance and presence pattern defined in Occupancy
                                elif (zone_name in self.rooms_daylighting or zone_name in self.rooms_daylighting_2) and zone_name not in self.bedrooms_list: 
                                    k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting['PreviousStateLighting{}'.format(zone_name)])
                                    illuminance_timestep = illuminance['Illuminance{}'.format(zone_name)]
                                    probability_timestep = max(probability_timestep_shading,horizontal_dilation((self.probability_off_during[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation)
                                # Bedrooms with sufficient daylight entrance and presence pattern defined in Occupancy
                                elif zone_name in self.bedrooms_list: 
                                    illuminance_timestep = illuminance['Illuminance{}'.format(zone_name)] 
                                    # Person has entered the previous timestep and went immediately to sleep
                                    if timestep-2 >=0  and self.occupancy[zone_name][timestep - 2] == 0 and self.asleep_bedroom[zone_name][timestep-1] == self.occupancy[zone_name][timestep - 1] and self.asleep_bedroom[zone_name][timestep] == self.occupancy[zone_name][timestep]: 
                                        probability_timestep = max(probability_timestep_shading, horizontal_dilation((self.probability_off_sleeping[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep))                          
                                    # Person awake
                                    elif self.asleep_bedroom[zone_name][timestep] != self.occupancy[zone_name][timestep]: 
                                        k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting['PreviousStateLighting{}'.format(zone_name)])
                                        probability_timestep = max(probability_timestep_shading, horizontal_dilation((self.probability_off_during[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) +k*self.standard_deviation)
                                    # Person is going to sleep 
                                    elif self.asleep_bedroom[zone_name][timestep] > self.asleep_bedroom[zone_name][timestep-1]: 
                                        probability_timestep = max(probability_timestep_shading, horizontal_dilation((self.probability_off_sleeping[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep))
                                    # All occupants in the bedroom are asleep, making interactions impossible
                                    else: 
                                        probability_timestep = 0
                            state_lighting = stochastic_switching(probability_timestep, previousstate_lighting['PreviousStateLighting{}'.format(zone_name)])
                        else: 
                            state_lighting = previousstate_lighting['PreviousStateLighting{}'.format(zone_name)]     
                        
                        
                # Light OFF during the previous timestep                
                else: 
                    # Nobody present in the room     
                    if self.lighting_requirements[zone_name][timestep] == 0: 
                        # Last occupant has left the room since the last timestep.
                        if timestep-1 > 0 and self.lighting_requirements[zone_name][timestep] == 1: 
                            # Rooms without sufficient daylight entrance
                            if zone_name in self.rooms_without_daylighting:
                                illuminance_timestep = 0
                                probability_timestep = horizontal_dilation((self.probability_on_leaving[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep)
                            # Rooms with daylight entrance
                            else: 
                                k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting['PreviousStateLighting{}'.format(zone_name)])
                                illuminance_timestep = illuminance['Illuminance{}'.format(zone_name)]
                                probability_timestep = horizontal_dilation((self.probability_on_leaving[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation
                            state_lighting = stochastic_switching(probability_timestep, previousstate_lighting['PreviousStateLighting{}'.format(zone_name)])
                        # Nobody present in the room, the lighting remains off
                        else: 
                            state_lighting = previousstate_lighting['PreviousStateLighting{}'.format(zone_name)]
                        
                    # Occupants present in the room
                    else: 
                        
                        # Check for critical moments
                        if timestep - 1 >= 0: 
                            # The occupants have interacted with the solar shading installation
                            probability_timestep_shading = 0
                            if shading_interaction == 1: 
                                illuminance_timestep = illuminance['Illuminance{}'.format(zone_name)]
                                probability_timestep_shading = horizontal_dilation((self.probability_on_solar), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep)      
                            # First occupant has entered the room
                            if self.lighting_requirements[zone_name][timestep-1] == 0: 
                                # Rooms without sufficient daylight entrance
                                if zone_name in self.rooms_without_daylighting: 
                                    illuminance_timestep = 0
                                    probability_timestep = max(probability_timestep_shading, horizontal_dilation((self.probability_on_entering[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep))
                                # Rooms, except bedrooms, with sufficient daylight entrance and presence pattern defined in Occupancy
                                elif (zone_name in self.rooms_daylighting or zone_name in self.rooms_daylighting_2) and zone_name not in self.bedrooms_list: 
                                    k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting['PreviousStateLighting{}'.format(zone_name)])
                                    illuminance_timestep = illuminance['Illuminance{}'.format(zone_name)]
                                    probability_timestep = max(probability_timestep_shading, horizontal_dilation((self.probability_on_entering[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation)
                                # Bedrooms with sufficient daylight entrance and presence pattern defined in Occupancy
                                elif zone_name in self.bedrooms_list:
                                    illuminance_timestep = illuminance['Illuminance{}'.format(zone_name)] 
                                    # Person awake
                                    if self.asleep_bedroom[zone_name][timestep] != self.occupancy[zone_name][timestep]:
                                        k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting['PreviousStateLighting{}'.format(zone_name)])
                                        probability_timestep = max(probability_timestep_shading, horizontal_dilation((self.probability_on_entering[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation)
                                    # All occupants in the bedroom are asleep, making interactions impossible
                                    else: 
                                        probability_timestep = 0
                            # An additional occupant has entered the room
                            elif zone_name not in ['ToiletGround', 'ToiletFirst', 'Toilet', 'Corridor'] and self.occupancy[zone_name][timestep] > self.occupancy[zone_name][timestep-1]: 
                                k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting['PreviousStateLighting{}'.format(zone_name)])
                                illuminance_timestep = illuminance['Illuminance{}'.format(zone_name)]
                                probability_timestep = max(probability_timestep_shading, horizontal_dilation((self.probability_on_entering_occupation[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation)
                            # An occupant has left the room
                            elif zone_name not in ['ToiletGround', 'ToiletFirst', 'Toilet', 'Corridor'] and self.occupancy[zone_name][timestep] < self.occupancy[zone_name][timestep-1]: 
                                k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting['PreviousStateLighting{}'.format(zone_name)])
                                illuminance_timestep = illuminance['Illuminance{}'.format(zone_name)]
                                probability_timestep = max(probability_timestep_shading, horizontal_dilation((self.probability_on_leaving_occupation[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation)
                            # No external critical moment detected
                            else: 
                                # Rooms without sufficient daylight entrance
                                if zone_name in self.rooms_without_daylighting: 
                                    illuminance_timestep = 0
                                    probability_timestep = max(probability_timestep_shading,horizontal_dilation((self.probability_on_during[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep))
                                # Rooms, except bedrooms, with sufficient daylight entrance and presence pattern defined in Occupancy
                                elif (zone_name in self.rooms_daylighting or zone_name in self.rooms_daylighting_2) and zone_name not in self.bedrooms_list: 
                                    k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting['PreviousStateLighting{}'.format(zone_name)])
                                    illuminance_timestep = illuminance['Illuminance{}'.format(zone_name)]
                                    probability_timestep = max(probability_timestep_shading,horizontal_dilation((self.probability_on_during[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation)
                                # Bedrooms with sufficient daylight entrance and presence pattern defined in Occupancy
                                elif zone_name in self.bedrooms_list:  
                                    illuminance_timestep = illuminance['Illuminance{}'.format(zone_name)] 
                                    # Person awake
                                    if self.asleep_bedroom[zone_name][timestep] != self.occupancy[zone_name][timestep]: 
                                        k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting['PreviousStateLighting{}'.format(zone_name)])
                                        probability_timestep = max(probability_timestep_shading,horizontal_dilation((self.probability_on_during[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation)
                                    # All occupants in the bedroom are asleep, making interactions impossible
                                    else: 
                                        probability_timestep = 0
                            state_lighting = stochastic_switching(probability_timestep, previousstate_lighting['PreviousStateLighting{}'.format(zone_name)])
                        else: 
                            state_lighting = previousstate_lighting['PreviousStateLighting{}'.format(zone_name)]            
            
            self.api.exchange.set_actuator_value(state, self.api.exchange.get_actuator_handle(state, "Schedule:Constant", "Schedule Value", "ScheduleLight"+zone_name), state_lighting)
            self.api.exchange.set_global_value(state, self.api.exchange.get_global_handle(state, 'PreviousStateLighting'+zone_name), state_lighting)
            
        return 0        
       