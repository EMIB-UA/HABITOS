#*******************************************************
# ENERGYPLUS PLUGIN: MANUAL LIGHTING AND SHADING CONTROL 
#*******************************************************

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
#     ActionPracticalDarknessZone,   !- Variable Name 2
#     ActionPracticalPrivacyZone,    !- Variable Name 3
#     ActionPracticalSecurityZone,   !- Variable Name 4
#     TimestepPreviousShadingZone,   !- Variable Name 5
#     TimestepInteractionZone;       !- Variable Name 6

# Required input EnergyPlus per zone and per orientation: 

#   PythonPlugin:Variables,
#     PythonPluginGlobalVariables,   !- Name
#     PreviousStateShadingZone.Orientation,  !- Variable Name 1

# Required input EnergyPlus per zone, per orientation and per occlusion rate with shading: 
    
#   Schedule:Constant,ScheduleShadingControlZone.Orientation_OcclusionRate,Fraction,0;

#   WindowShadingControl,
#     ControlScreensZoneOrientationOcclusionRate,  !- Name
#     Zone,                    !- Zone Name
#     ,                        !- Shading Control Sequence Number
#     ExteriorScreen,          !- Shading Type
#     ,                        !- Construction with Shading Name
#     OnIfScheduleAllows,      !- Shading Control Type
#     ScheduleShadingControlZone.Orientation_OcclusionRate,  !- Schedule Name
#     ,                        !- Setpoint {W/m2, W or deg C}
#     Yes,                     !- Shading Control Is Scheduled
#     No,                      !- Glare Control Is Active
#     MaterialExteriorScreen,  !- Shading Device Material Name
#     ,                        !- Type of Slat Angle Control for Blinds
#     ,                        !- Slat Angle Schedule Name
#     ,                        !- Setpoint 2 {W/m2 or deg C}
#     ,                        !- Daylighting Control Object Name
#     Group,                   !- Multiple Surface Control Type
#     WindowsZoneOrientationOcclusionRate;   !- Fenestration Surface 1 Name


#  Required input EnergyPlus per zone with daylight enterance:  

#   Daylighting:Controls,
#     DaylightingControlZone,  !- Name
#     Zone,                    !- Zone or Space Name
#     SplitFlux,               !- Daylighting Method
#     ScheduleAlways0,         !- Availability Schedule Name
#     Continuous,              !- Lighting Control Type
#     0,                     !- Minimum Input Power Fraction for Continuous or ContinuousOff Dimming Control
#     0,                     !- Minimum Light Output Fraction for Continuous or ContinuousOff Dimming Control
#     ,                        !- Number of Stepped Control Steps
#     1,                       !- Probability Lighting will be Reset When Needed in Manual Stepped Control
#     ReferencePointZone,      !- Glare Calculation Daylighting Reference Point Name
#     ,                        !- Glare Calculation Azimuth Angle of View Direction Clockwise from Zone y-Axis {deg}
#     ,                      !- Maximum Allowable Discomfort Glare index
#     ,                        !- DElight Gridding Resolution {m2}
#     ReferencePointZone,      !- Daylighting Reference Point 1 Name
#     1,                       !- Fraction of Lights Controlled by Reference Point 1
#     IlluminanceZone;                     !- Illuminance Setpoint at Reference Point 1 {lux}

# Daylighting:ReferencePoint,
#     ReferencePointZone,   !- Name
#     Zone,                 !- Zone or Space Name
#     ,                     !- X-Coordinate of Reference Point {m}
#     ,                     !- Y-Coordinate of Reference Point {m}
#     0.8;                  !- Z-Coordinate of Reference Point {m}

#------------------------------------------------
# PYTHON CODE MANUAL LIGHTING AND SHADING CONTROL
#------------------------------------------------

import math
import importlib.util
import re
import os
import glob
import random
import pandas as pd
import numpy as np
import builtins
import csv
from datetime import datetime
from eppy import modeleditor
from eppy.modeleditor import IDF
from eppy.runner.run_functions import run
from scipy.optimize import differential_evolution
from scipy.optimize import least_squares
from pyenergyplus.plugin import EnergyPlusPlugin

def parse_datetime(month, date, time): 
    date_format = "%m-%d %H:%M"
    return datetime.strptime(str(month) + "-" + str(date) +" "+ str(int(time)-1) + ":00", date_format)
    
class LightingAndShadingControl(EnergyPlusPlugin): 

    def __init__(self): 
        super().__init__()

        # PROVIDE INFORMATION ABOUT THE BUILDING GEOMETRY IN ENERGYPLUS
        rooms_lighting = [] # List all zone names with a defined presence pattern (in Occupancy.csv) and Corridor
        rooms_daylighting = [] # Give all the rooms where one daylighting reference point is formulated        
        rooms_daylighting_2 = [] # Give the rooms where 2 daylighting reference points are formulated 
        rooms_shading = [] # List all zone names with solar shading.
        rooms_3_orientations = [] # List the zone names with windows in three orientations.
        rooms_2_orientations = [] # List the zone names with windows in two orientations
        overview_windows = {} # Define a specific window name for each orientation of the rooms
        # Automatically generated
        rooms_with_daylighting = set(rooms_daylighting) | set(rooms_daylighting_2)
        rooms_without_daylighting = [room for room in rooms_lighting if room not in rooms_with_daylighting] # Lists all rooms without windows
        bedrooms = [room for room in rooms_lighting if 'Bedroom' in room] # Sum the bedrooms
        self.rooms_lighting = rooms_lighting
        self.rooms_daylighting = rooms_daylighting
        self.rooms_daylighting_2 = rooms_daylighting_2
        self.rooms_without_daylighting = rooms_without_daylighting
        self.rooms_shading = rooms_shading
        self.rooms_3_orientations = rooms_3_orientations
        self.rooms_2_orientations = rooms_2_orientations
        self.overview_windows = overview_windows

        # IMPORT OCCUPANT BEHAVIOUR
        directory_occupancy = r'' # Refer to the directory in which the occupancy profiles have been saved. 
        occupancy_dataframe = pd.read_csv(directory_occupancy + '/Occupancy.csv').round({'Time': 2})
        occupancy_dataframe['Building'] = occupancy_dataframe.iloc[:,3:].sum(axis = 1) # Add a column with the total amount of present inhabitants
        occupancy_dataframe['LivingKitchen'] = occupancy_dataframe['Kitchen'] + occupancy_dataframe['Living']  # Add a column containing the sum of Living and Kitchen
        self.occupancy_dataframe = occupancy_dataframe
        self.occupancy = occupancy_dataframe.to_dict()
        asleep_bedroom_dataframe = pd.read_csv(directory_occupancy + '/AsleepBedroom.csv').round({'Time': 2})
        self.asleep_bedroom = asleep_bedroom_dataframe.to_dict()
        bedroom_columns = [column for column in asleep_bedroom_dataframe.columns if 'Bedroom' in column]
        asleep_dataframe = asleep_bedroom_dataframe[bedroom_columns].sum(axis = 1)
        self.asleep = asleep_dataframe.to_dict()
        task_activities_dataframe = pd.read_csv(directory_occupancy+'/TaskAndActivities.csv').round({'Time' : 2})
        task_activities = task_activities_dataframe.to_dict()
        lighting_requirements_dataframe = pd.read_csv(directory_occupancy+'/PresenceLightingDesign.csv').round({'Time' : 2})
        lighting_requirements_dataframe['LivingKitchen'] = lighting_requirements_dataframe[['Kitchen', 'Living']].apply(lambda row: 1 if (row['Kitchen'] == 1 or row['Living'] == 1) else 0, axis=1) # Add a column containing the information for an open kitchen
        self.lighting_requirements = lighting_requirements_dataframe.to_dict() 
        directory_data_lighting = r'' # Give the directory in which the lighting data is saved. 
        directory_data_shading = r'' # Give the directory in which the shading data is saved.
        
        # PROVIDE INFORMATION ABOUT ENERGYPLUS INSTALLATION AND SEPARATE IDF FILE TO CALCULATE ILLUMINANCE
        idd_path = r'' # Specify the .idd path of the EnergyPlus installation
        idf_path = r'' # Specify the .idf path where a copy of the simulation file is saved that can be used for daylighting calculations
        weather_path = r'' # Specify the path to the weather file
        output_path = r'' # Specify the directory in which te daylighting simulation results will be saved
        os.makedirs(output_path, exist_ok = True)
        
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
            
        # DEFINE LIGHTING BEHAVIOUR
        def generate_random_exponential_probability_function_lighting(lower_expr, upper_expr, interval_range=(0, 1), steepness_range=(0, 1), increasing=True, constrain_point=None, max_iterations = 10):
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
                            
                return {'a': a_opt, 'b': b_opt, 'c': c_opt}
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

        
        def reflect_exponential_function(a, b, c): 
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
        probs = np.loadtxt(directory_data_lighting +'/family_habits.txt', float)
        family_characterised = get_probability(np.random.random(), probs)
        # Step 1b. Define the family habits in relation to the moment they switch the lighting on and off
        # (0) (In)sufficient daylighting, (1) switching solar shading and (2) entering or leaving the room 
        probs = np.loadtxt(directory_data_lighting +'/reasons_on_living.txt', float)
        reason_on_living = get_probability(np.random.random(), probs)
        reasons_on_living = np.zeros(3)
        if reason_on_living in [1, 4, 5, 7]: 
            reasons_on_living[0] = 1
        if reason_on_living in [2, 4, 6, 7]: 
            reasons_on_living[1] = 1
        if reason_on_living in [3, 5, 6, 7]:
            reasons_on_living[2] = 1
        probs = np.loadtxt(directory_data_lighting+'/reasons_off_living.txt', float)
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
        probs = np.loadtxt(directory_data_lighting+'/not_switching_off.txt', float)
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
        self.reasons_not_off = reasons_not_off
        # Step 1d. Define the household in relation to their illuminance requirements. 
        # A value between 0 and 1, with 0 corresponding to low illuminance levels and 1 to high illuminances. 
        household_requirements = np.random.random()
        
        # Step 2. Define the habits at room level
        # Step 2a. When does the family switch on the ligthing in the different rooms? 
        # (0) Insufficient daylighting, (1) closing solar shading and (2) entering the room 
        # Bedroom
        probs = np.loadtxt(directory_data_lighting+'/reasons_on_bedroom.txt', float)
        reason_on_bedroom = get_probability(np.random.random(), probs[:,reason_on_living - 1])
        reasons_on_bedroom = np.zeros(3)
        if reason_on_bedroom in [1, 4, 5, 7]: 
            reasons_on_bedroom[0] = 1
        if reason_on_bedroom in [2, 4, 6, 7]: 
            reasons_on_bedroom[1] = 1
        if reason_on_bedroom in [3, 5, 6, 7]:
            reasons_on_bedroom[2] = 1
        # Bathroom
        probs = np.loadtxt(directory_data_lighting+'/reasons_on_bathroom.txt', float)
        reason_on_bathroom = get_probability(np.random.random(), probs[:,reason_on_living - 1])
        reasons_on_bathroom = np.zeros(3)
        if reason_on_bathroom in [1, 4, 5, 7]: 
            reasons_on_bathroom[0] = 1
        if reason_on_bathroom in [2, 4, 6, 7]: 
            reasons_on_bathroom[1] = 1
        if reason_on_bathroom in [3, 5, 6, 7]:
            reasons_on_bathroom[2] = 1
        # Toilet and storage
        probs = np.loadtxt(directory_data_lighting+'/reasons_on_toilet.txt', float)
        reason_on_toilet = get_probability(np.random.random(), probs[:,reason_on_living - 1])
        reasons_on_toilet = np.zeros(3)
        if reason_on_toilet in [1, 4, 5, 7]: 
            reasons_on_toilet[0] = 1
        if reason_on_toilet in [2, 4, 6, 7]: 
            reasons_on_toilet[1] = 1
        if reason_on_toilet in [3, 5, 6, 7]:
            reasons_on_toilet[2] = 1
        # Hallway
        probs = np.loadtxt(directory_data_lighting+'/reasons_on_hallway.txt', float)
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
        probs = np.loadtxt(directory_data_lighting+'/reasons_off_bedroom.txt', float)
        reason_off_bedroom = get_probability(np.random.random(), probs[:,reason_off_living - 1])
        reasons_off_bedroom = np.zeros(3)
        if reason_off_bedroom in [1, 4, 5, 7]: 
            reasons_off_bedroom[0] = 1
        if reason_off_bedroom in [2, 4, 6, 7]: 
            reasons_off_bedroom[1] = 1
        if reason_off_bedroom in [3, 5, 6, 7]:
            reasons_off_bedroom[2] = 1
        # Bathroom
        probs = np.loadtxt(directory_data_lighting+'/reasons_off_bathroom.txt', float)
        reason_off_bathroom = get_probability(np.random.random(), probs[:,reason_off_living - 1])
        reasons_off_bathroom = np.zeros(3)
        if reason_off_bathroom in [1, 4, 5, 7]: 
            reasons_off_bathroom[0] = 1
        if reason_off_bathroom in [2, 4, 6, 7]: 
            reasons_off_bathroom[1] = 1
        if reason_off_bathroom in [3, 5, 6, 7]:
            reasons_off_bathroom[2] = 1
        # Toilet and storage
        probs = np.loadtxt(directory_data_lighting+'/reasons_off_toilet.txt', float)
        reason_off_toilet = get_probability(np.random.random(), probs[:,reason_off_living - 1])
        reasons_off_toilet = np.zeros(3)
        if reason_off_toilet in [1, 4, 5, 7]: 
            reasons_off_toilet[0] = 1
        if reason_off_toilet in [2, 4, 6, 7]: 
            reasons_off_toilet[1] = 1
        if reason_off_toilet in [3, 5, 6, 7]:
            reasons_off_toilet[2] = 1
        # Hallway
        probs = np.loadtxt(directory_data_lighting+'/reasons_off_hallway.txt', float)
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
        probs = np.loadtxt(directory_data_lighting+'/unnecessary_bedroom.txt', float)
        unnecessary_bedroom = get_probability(np.random.random(), probs[:,unnecessary_living - 1])
        # Bathroom
        probs = np.loadtxt(directory_data_lighting+'/unnecessary_bathroom.txt', float)
        unnecessary_bathroom = get_probability(np.random.random(), probs[:,unnecessary_living - 1])
        # Toilet and storage
        probs = np.loadtxt(directory_data_lighting+'/unnecessary_toilet.txt', float)
        unnecessary_toilet = get_probability(np.random.random(), probs[:,unnecessary_living - 1])
        # Hallway
        probs = np.loadtxt(directory_data_lighting+'/unnecessary_hallway.txt', float)
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
        self.offices_list = offices_list
        self.bedrooms_list = bedrooms_list             
        
        # Import the lower and upper probability functions. 
        on_entering_bounds = pd.read_csv(directory_data_lighting + '/on_entering.txt', delimiter='\t', dtype=str, comment='#', header = None)
        on_during_bounds = pd.read_csv(directory_data_lighting+'/on_during.txt', delimiter='\t', dtype=str, comment='#', header = None)
        off_leaving_bounds = pd.read_csv(directory_data_lighting+'/off_leaving.txt', delimiter='\t', dtype=str, comment='#', header = None)
        off_during_bounds = pd.read_csv(directory_data_lighting+'/off_during.txt', delimiter='\t', dtype=str, comment='#', header = None)
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
            probs = np.loadtxt(directory_data_lighting+'/interval_lighton.txt', float)
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
            probability_on_entering[zone_name] = generate_random_exponential_probability_function_lighting(on_entering_lower, on_entering_upper, interval_on_entering, steepness_on, increasing = False)
            # During occupation (in function of indoor illuminance)
            on_during_lower = on_during_bounds.iloc[room_type - 1,0]
            on_during_upper = on_during_bounds.iloc[room_type - 1,1]
            probability_on_during[zone_name] = generate_random_exponential_probability_function_lighting(on_during_lower, on_during_upper, interval_on_during, steepness_on, increasing = False) 
            # Step 3c. Define the relations for switching off. 
            # Last occupant leaving the room (in function of time until next occupation)
            off_leaving_lower = off_leaving_bounds.iloc[room_type - 1,0]
            off_leaving_upper = off_leaving_bounds.iloc[room_type - 1,1]
            probability_off_leaving[zone_name] = generate_random_logarithmic_probability_function(off_leaving_lower, off_leaving_upper, target_time, interval_off_leaving)
            # During occupation (in function of indoor illuminance)
            off_during_lower = off_during_bounds.iloc[room_type - 1,0]
            off_during_upper = off_during_bounds.iloc[room_type - 1,1]
            probability_off_during[zone_name] = generate_random_exponential_probability_function_lighting(off_during_lower, off_during_upper, interval_off_during)
            # Step 3d. Formulate probability functions for additional (critical) moments (based on assumptions).
            # Critical moment: switching solar shading
            if reasons_off[room_habits][1] == 1: 
                # Assumption: the eyes don't have the time to gradually adapt to the darkness and the person in moving already (however, it can still require effort to go the the switch). Since both have an opposite effect, it is assumed that this corresponds to the probability of entering. 
                probability_on_solar[zone_name] = probability_on_entering[zone_name]
                # Assumption: Opening the solar shading results in a probability to switch off the lighting that is increased in comparison to during occupation. It is assumed that this probability is increasing with increasing indoor illuminance and is located between the off-probability for during + 15% and the mirrored on-probability (around y = 0.5) - 15%.  
                probability_off_solar[zone_name] = generate_random_exponential_probability_function_lighting(probability_off_during[zone_name], reflect_exponential_function(**probability_on_entering[zone_name]), (0.15,0.85))
            else: 
                probability_on_solar[zone_name] = 0
                probability_off_solar[zone_name] = 0
            # Critical moment: first occupant enters the room after the lighting was not switched off. 
            # Assumption: The probability to switch off the lighting when entering as first is assumed to be inbetween probability_off_entering_occupation and the reflected probability_on_entering. An interval of (0,0.5) is proposed to set the probability function. 
            if occupancy_dataframe['Building'].max() > 1: 
                probability_off_entering[zone_name] = generate_random_exponential_probability_function_lighting(probability_off_during[zone_name], reflect_exponential_function(**probability_on_entering[zone_name]), (0,0.5))
            else: 
                probability_lower = generate_random_exponential_probability_function_lighting(probability_off_during[zone_name], reflect_exponential_function(**probability_on_entering[zone_name]), (0,0.5))
                point_constrain = min(generate_exponential_function(probability_off_during[zone_name])(0), generate_exponential_function(reflect_exponential_function(**probability_on_entering[zone_name]))(0))
                probability_off_entering[zone_name] = generate_random_exponential_probability_function_lighting(probability_lower, reflect_exponential_function(**probability_on_entering[zone_name]), interval_range = (0,0.5), constrain_point =(0, point_constrain))
            # Critical moment: additional occupant entering the room
            if occupancy_dataframe['Building'].max() > 1: 
                # Assumption: this person interacts independently. 
                probability_on_entering_occupation[zone_name] = probability_on_entering[zone_name]
                # Assumption: the occupant is expected to interact more effectively than during occupation. It is assumed that this probability is increasing with increasing indoor illuminance and is located between the off-probability for during + 15% and the off probability when entering - 15%.  
                point_constrain = min(generate_exponential_function(probability_off_during[zone_name])(0), generate_exponential_function(reflect_exponential_function(**probability_on_entering[zone_name]))(0))
                probability_off_entering_occupation[zone_name] = generate_random_exponential_probability_function_lighting(probability_off_during[zone_name], probability_off_entering[zone_name], interval_range =  (0.15,0.85), constrain_point =(0,point_constrain))
            # Critical moment: occupant leaving the room while at least one person stays in the room
            if occupancy_dataframe['Building'].max() > 1: 
                # Assumption: the probability is located between the probability when entering the room as first and interacting during occupation, with the difference that this person is now not acting for themself. Therefore, a probability function in the lowest quart of the interval between both functions is suggested.  
                probability_on_leaving_occupation[zone_name] = generate_random_exponential_probability_function_lighting(probability_on_during[zone_name], probability_on_entering[zone_name], (0,0.25), increasing = False)
                # Assumption: an occupant will occasionally switch off the lighting when leaving as this requires limited additional effort, but affects the comfort of his household members. It is assumed that this probability is increasing with increasing indoor illuminance and is located between the off-probability for during + 15% and the mirrored on-probability (around y = 0.5) - 15%.  
                point_constrain = min(generate_exponential_function(probability_off_during[zone_name])(0), generate_exponential_function(reflect_exponential_function(**probability_on_entering[zone_name]))(0))
                probability_off_leaving_occupation[zone_name] = generate_random_exponential_probability_function_lighting(probability_off_during[zone_name], reflect_exponential_function(**probability_on_entering[zone_name]), interval_range = (0.15,0.85), constrain_point = (0,point_constrain))
            # Critical moment: last occupant leaves the room when the lighting is switched off. 
            # Assumption: This person will not switch on the lighting. 
            probability_on_leaving[zone_name] = {'a': 0, 'b': 0, 'c': 0, 'd': 0}
            # Occupant is going to sleep. 
            # Assumption: lighting will be switched off between 95 and 99.99% of the occurances. 
            if 'Bedroom' in zone_name: 
                probability_off = random.uniform(0.95, 0.9999)
                probability_off_sleeping[zone_name] = {'a': 0, 'b': 0, 'c': 0, 'd': probability_off}
        self.probability_on_entering = probability_off_entering
        self.probability_on_during = probability_on_during
        self.probability_off_leaving = probability_off_leaving
        self.probability_off_during = probability_off_during
        self.probability_on_solar = probability_on_solar
        self.probability_off_solar = probability_off_solar
        self.probability_on_leaving_occupation = probability_on_leaving_occupation
        self.probability_off_leaving_occupation = probability_off_leaving_occupation
        self.probability_on_entering_occupation = probability_on_entering_occupation
        self.probability_off_entering_occupation = probability_off_entering_occupation
        self.probability_off_entering = probability_off_entering
        self.probability_on_leaving = probability_on_leaving
        self.probability_off_sleeping = probability_off_sleeping 
        # Step 3d. Set the average standard deviation 
        self.standard_deviation = np.loadtxt(directory_data_lighting+'/correction_deviation.txt')
            
        # Step 4. Define the household preferences in relation to illuminance
        # Step 4a. Generate a dictionary containing the desired illuminance per room and per timestep. 
        # rows: (1) Cooking, dishes, computer and office work, (2) Cleaning and ironing, (3) Personal hygiene and laundry and (4) Entertainment and passageways
        illuminance_ranges = np.loadtxt(directory_data_lighting +'/illuminance_ranges.txt')
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
        self.minimum_illuminance = minimum_illuminance
            
        # Step 5. Generate the habits that are used to model the practical habits
        # Stap 5a. Define 'security_lighting lighting' for the household
        # Assumption: Light will be switched on in during periods of absence and darkness
        # Rooms
        security_lighting_rooms_overview = ['Living', 'Kitchen', 'Corridor']
        self.security_lighting_rooms = random.sample(security_lighting_rooms_overview, random.randint(1, len(security_lighting_rooms_overview)))
        # Habits
        security_lighting_habits_overview = ['dark and absence', 'dark and absence/asleep', 'anticipate dark and absence', 'anticipate dark and absence/asleep']
        self.security_lighting_habits = random.choice(security_lighting_habits_overview)
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
        self.sunset = sunset
        self.sunrise = sunrise
            
        # Step 5c. Probability habits
        # Assumption: There is a 0-10% chance the inhabitants do not follow their habits (e.g. they are returning later than expected). 
        self.probability_security_lighting = random.uniform(0.90, 1.00)  
        
        # DEFINE SHADING BEHAVIOUR 
        
        def generate_random_linear_probability_function_shading(lower_expr, upper_expr, interval_range=(0, 1), increasing=True, max_iterations=5):
            """
            Generates a linear function within given bounds with variable slope and intercept.
            """
            
            def linear_function(a, b, x):
                return a * x + b

            def parse_linear_expression(expr):
                """Parses linear expressions of the form 'a*x + b'."""
                expr = expr.replace(' ', '')
                if expr == '0':
                    return {'a': 0.0, 'b': 0.0}
                elif expr == '1':
                    return {'a': 0.0, 'b': 1.0}
                if 'x' not in expr:
                    return {'a': 0.0, 'b': float(expr)}
                parts = expr.split('x')
                a = float(parts[0].replace('*', '')) if parts[0] not in ['', '+'] else 1.0
                b = float(parts[1]) if len(parts) > 1 and parts[1] not in ['', '+', '-'] else 0.0
                return {'a': a, 'b': b}

            def is_valid_line(fx, new_lower_bound, new_upper_bound, adaptive_tol=0.05, absolute_tol=1e-2):
                """
                Valid if inside bounds, with adaptive tolerance around crossings.
                Only checks points where bounds are strictly within [0,1].
                """
                delta = new_upper_bound - new_lower_bound
                cross_mask = delta <= 1e-2
                local_tol = np.full_like(fx, adaptive_tol)
                local_tol[cross_mask] *= 3
                lower_limit = new_lower_bound - np.maximum(absolute_tol, new_lower_bound*local_tol)
                upper_limit = new_upper_bound + np.maximum(absolute_tol, new_upper_bound*local_tol)
            
                for j in range(len(fx)):
                    if new_lower_bound[j] > 0 and new_upper_bound[j] < 1:
                        if not (lower_limit[j] <= fx[j] <= upper_limit[j]):
                            return False
                return True
                
            x = np.linspace(0, 1000, 100)
            lower_params = parse_linear_expression(lower_expr) if isinstance(lower_expr, str) else lower_expr
            upper_params = parse_linear_expression(upper_expr) if isinstance(upper_expr, str) else upper_expr

            def loss_function(params):
                a, b = params
                fx = linear_function(a, b, x)
                lower_bound = linear_function(lower_params['a'], lower_params['b'], x)
                upper_bound = linear_function(upper_params['a'], upper_params['b'], x)
                corrected_lower = np.minimum(lower_bound, upper_bound)
                corrected_upper = np.maximum(lower_bound, upper_bound)
                new_lower_bound = corrected_lower + (corrected_upper - corrected_lower) * interval_range[0]
                new_upper_bound = corrected_lower + (corrected_upper - corrected_lower) * interval_range[1]
                
                # Hard constraint: outside the bounds → infinite penalty
                if np.any(fx < new_lower_bound) or np.any(fx > new_upper_bound):
                    return np.inf
                
                # Soft quadratic penalty near the bounds
                penalty = np.sum(np.maximum(0, new_lower_bound - fx)**2) + np.sum(np.maximum(0, fx - new_upper_bound)**2)

                return penalty

            a_min = min(lower_params['a'], upper_params['a'])
            a_max = max(lower_params['a'], upper_params['a'])
            b_min = min(lower_params['b'], upper_params['b'])
            b_max = max(lower_params['b'], upper_params['b'])

            if increasing:
                a_min = max(a_min, 0)
            else:
                a_max = min(a_max, 0)

            bounds = [(a_min, a_max), (b_min, b_max)]

            for iteration in range(max_iterations):
                result = differential_evolution(loss_function, bounds, strategy='best1bin', tol=1e-5, maxiter=1000)
                if result.success:
                    a_opt, b_opt = result.x
                    break
                else: 
                    a_opt = None
                    b_opt = None
                
                if iteration == max_iterations -1: 
                    for i in range(max_iterations + 5):     
                        # Fallback: random interpolation within the interval
                        lower_bound = linear_function(lower_params['a'], lower_params['b'], x)
                        upper_bound = linear_function(upper_params['a'], upper_params['b'], x)
                        corrected_lower = np.minimum(lower_bound, upper_bound)
                        corrected_upper = np.maximum(lower_bound, upper_bound)
                        new_lower_bound = corrected_lower + (corrected_upper - corrected_lower) * interval_range[0]
                        new_upper_bound = corrected_lower + (corrected_upper - corrected_lower) * interval_range[1]
                
                        x0, x1 = x[0], x[-1]
                        
                        # Calculate the cutting point
                        a_lower, b_lower = lower_params['a'], lower_params['b']
                        a_upper, b_upper = upper_params['a'], upper_params['b']
                        
                        if a_lower != a_upper:  # check parallel
                            x_intersect = (b_upper - b_lower) / (a_lower - a_upper)
                            y_intersect = a_lower * x_intersect + b_lower
                        else:
                            y_intersect = None
                        
                        if y_intersect <= 0:
                            x0, y0 = x_intersect, y_intersect
                        else:
                            y0_min = new_lower_bound[0] + (new_upper_bound[0] - new_lower_bound[0]) * interval_range[0]
                            y0_max = new_lower_bound[0] + (new_upper_bound[0] - new_lower_bound[0]) * interval_range[1]
                            x0, y0 = x[0], np.random.uniform(y0_min, y0_max)
                        
                        # Assign y1 randomly within interval
                        y0_min = new_lower_bound[0] + (new_upper_bound[0] - new_lower_bound[0]) * interval_range[0]
                        y0_max = new_lower_bound[0] + (new_upper_bound[0] - new_lower_bound[0]) * interval_range[1]
                        y1_min = new_lower_bound[-1] + (new_upper_bound[-1] - new_lower_bound[-1]) * interval_range[0]
                        y1_max = new_lower_bound[-1] + (new_upper_bound[-1] - new_lower_bound[-1]) * interval_range[1]

                        y1 = np.random.uniform(y1_min, y1_max)

                        a_try = (y1 - y0) / (x1 - x0)
                        b_try = y0 - a_try * x0
                        fx_try = linear_function(a_try, b_try, x)

                        if is_valid_line(fx_try, new_lower_bound, new_upper_bound):
                            a_opt, b_opt = a_try, b_try
                            break
            
            if a_opt is None: 
                raise ValueError('Generated function is outside bounds.')

            return {'a': a_opt, 'b': b_opt}

        def generate_random_exponential_probability_function_shading(lower_expr, upper_expr, x, interval_range=(0, 1), increasing=True, max_iterations=10):
            def exponential_function(a, b, c, x):
                try:
                    fx = a + c * np.exp(b * x)
                    if not np.all(np.isfinite(fx)):
                        return np.full_like(x, np.nan)
                    return fx
                except:
                    return np.full_like(x, np.nan)

            def parse_exponential_function(expression):
                """Parses an exponential function string of the form 'a*exp(b*x+c)+d' and extracts a, b, c, d."""
                expression = expression.replace(" ", "")
                expression = expression.replace("−", "-")
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

            lower_params = parse_exponential_function(lower_expr) if isinstance(lower_expr, str) else lower_expr
            upper_params = parse_exponential_function(upper_expr) if isinstance(upper_expr, str) else upper_expr

            lower_bound = exponential_function(**lower_params, x=x)
            upper_bound = exponential_function(**upper_params, x=x)


            corrected_lower = np.minimum(lower_bound, upper_bound)
            corrected_upper = np.maximum(lower_bound, upper_bound)
            new_lower_bound = corrected_lower + (corrected_upper - corrected_lower) * interval_range[0]
            new_upper_bound = corrected_lower + (corrected_upper - corrected_lower) * interval_range[1]

            def loss_function(params):
                a, b, c = params
                fx = exponential_function(a, b, c, x)
                if np.any(np.isnan(fx)):
                    return np.inf
                if np.any(fx < new_lower_bound) or np.any(fx > new_upper_bound):
                    return np.inf
                return 0.0  # We only care about being within bounds

            def residuals(params, x_vals, y_vals):
                return exponential_function(*params, x_vals) - y_vals

            success = False
            bounds = [
                (-100, 100),   # a
                (-50, 50),       # b
                (-100, 100)     # c
            ]
            for iteration in range(max_iterations):
                result = differential_evolution(loss_function, bounds, strategy='best1bin', tol=1e-5, maxiter=2000)
                if result.success:
                    a_opt, b_opt, c_opt = result.x
                    fx = exponential_function(a_opt, b_opt, c_opt, x)
                    if np.any(fx < new_lower_bound) or np.any(fx > new_upper_bound):
                        raise ValueError("Optimised function is outside bounds.")
                    success = True
                    break

                if iteration == max_iterations -1: 
                # Fallback: interpolatie
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
                    lower_params = parse_exponential_function(lower_expr) if isinstance(lower_expr, str) else lower_expr
                    upper_params = parse_exponential_function(upper_expr) if isinstance(upper_expr, str) else upper_expr
                    
                    lower_func = exponential_function(**lower_params, x=x)
                    upper_func = exponential_function(**upper_params, x=x)
                    
                    lower_bound = np.minimum(lower_func, upper_func)
                    upper_bound = np.maximum(lower_func, upper_func)
                    delta = upper_bound - lower_bound
                    for i in range(max_iterations + 5): 
                        alpha = np.random.uniform(interval_range[0], interval_range[1])
                        f_target = lower_bound + alpha*delta
                        interpolated_a, interpolated_b, interpolated_c = estimate_initialguess(x, f_target)
                        
                        fx = interpolated_a+interpolated_c*np.exp(interpolated_b*x)

                        # Check if fx stays within bounds with tolerance
                        tolerance = 0.05
                        cross_mask = delta <= 1e-2  
                        adaptive_tol = np.ones_like(x) * tolerance
                        adaptive_tol[cross_mask] *= 3.0 
                        absolute_tol = 5e-2
                        
                        lower_limit = lower_bound - np.maximum(absolute_tol, lower_bound*adaptive_tol)
                        upper_limit = upper_bound + np.maximum(absolute_tol, upper_bound*adaptive_tol)
                    
                        if np.all((fx >= lower_limit) & (fx <= upper_limit)):
                            a_opt = interpolated_a
                            b_opt = interpolated_b
                            c_opt = interpolated_c
                            success = True
                            break
            
            if success == True: 
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
                        
                lower_limit = min_bound - np.maximum(absolute_tol, lower_bound*adaptive_tol)
                upper_limit = max_bound + np.maximum(absolute_tol, upper_bound*adaptive_tol)
                
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
                            
                return {'a': a_opt, 'b': b_opt, 'c': c_opt}
            else: 
                raise ValueError(
                        "Generated function is outside 5% tolerance bounds."
                    )

            raise ValueError("Generated function values are outside the specified bounds after all attempts.")

        def generate_linear_function(parameters): 
            '''
            Generate the linear probability function.
            '''
            def f(x):
                a, b = parameters['a'], parameters['b']
                return a*x + b
            return f
          
        def reflect_linear_function(y, a, b): 
            ''' Reflect the linear function around y.'''
            a_ref = -a 
            b_ref = 2*y-b
            return {'a': a_ref, 'b': b_ref}

        # ASSIGN ROUTINES AND VARIABLES

        # Step 1. Define the habits at family level
        # Step 1a. What is the main driver for adjusting the screens (in the living room/kitchen/offices)?
        # (1) Darkness, (2) thermal comfort, (3) privacy, (4) security and (5) visual comfort
        probs = np.loadtxt(directory_data_shading+'/driver_livingkitchen.txt', float)
        driver_living = get_probability(np.random.random(), probs)
        # Step 1b. How often do the family on average switch the state of the screens? This family characteristic is determined independently of the main habits.
        # (1) Never, (2) several times a month, (3) several times a week, (4) daily or (5) multiple times a day
        probs = np.loadtxt(directory_data_shading+'/number_interactions.txt', float)
        number_interactions = get_probability(np.random.random(), probs)
        # Step 1c. Define the activeness of the family with regard to adapting their screens.
        # Does the family anticipate on thermal discomfort by closing their shades on beforehand.
        # (1) Family does interact when experiencing thermal discomfort or (2) family anticipate on future risk of overheating.
        probs = np.loadtxt(directory_data_shading+'/thermaldiscomfort.txt', float)
        thermal_anticipate = get_probability(np.random.random(), probs)
        self.thermal_anticipate = thermal_anticipate

        # Step 2. Define the habits at room level and per season
        # Step 2a. What is the main driver in the bathroom, bedroom and other rooms? This relation depends on the main driver selected for the living room/kitchen.
        # (1) Darkness, (2) thermal comfort, (3) privacy, (4) security and (5) visual comfort
        # Bedroom
        probs = np.loadtxt(directory_data_shading+'/driver_bedroom.txt', float)
        driver_bedroom = get_probability(np.random.random(), probs[:, driver_living - 1])
        # Bathroom
        probs = np.loadtxt(directory_data_shading+'/driver_bathroom.txt', float)
        driver_bathroom = get_probability(np.random.random(), probs[:, driver_living - 1])
        # Other rooms
        probs = np.loadtxt(directory_data_shading+'/driver_other.txt', float)
        driver_other = get_probability(np.random.random(), probs[:, driver_living - 1])
        # Step 2b. What secondary drivers do also cause the occupants to interact with their shading installations?
        # (1) Darkness, (2) darkness, thermal comfort, (3) darkness, privacy, (4) darkness, security, (5) darkness, visual comfort, (6) darkness, thermal comfort, privacy, (7) darkness, thermal comfort, security, (8) darkness, thermal comfort, visual comfort, (9) darkness, security, privacy, (10) darkness, security, visual comfort, (11) darkness, privacy, visual comfort, (12) darkness, thermal comfort, privacy, security, (13) darkness, thermal comfort, privacy, visual comfort, (14) darkness, thermal comfort, security, visual comfort, (15) darkness, privacy, security, visual comfort, (16) darkness, thermal comfort, privacy, security, visual comfort, (17) thermal comfort, (18) thermal comfort, privacy, (19) thermal comfort, security, (20) thermal comfort, visual comfort, (21) thermal comfort, privacy, security, (22) thermal comfort, privacy, visual comfort, (23) thermal comfort, security, visual comfort, (24) thermal comfort, privacy, security, visual comfort, (25) privacy, (26) privacy, security, (27) privacy, visual comfort, (28) privacy, security, visual comfort, (29) security, (30) security, visual comfort, (31) visual comfort
        # The combination of drivers per room is identified and returned as a list in which 1 indicates that the driver is adopted as habit by the family.
        # (0) Darkness, (1) thermal comfort, (2) privacy, (3) security and (4) visual comfort
        # Living room/kitchen/offices
        probs = np.loadtxt(directory_data_shading+'/drivers_sec_living.txt', float)
        drivers_comb_living = get_probability(np.random.random(), probs[:, driver_living - 1])
        drivers_sec_living = np.zeros(5)
        if drivers_comb_living in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:  # Darkness
            drivers_sec_living[0] = 1
        if drivers_comb_living in [2, 6, 5, 8, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24]:  # Thermal comfort
            drivers_sec_living[1] = 1
        if drivers_comb_living in [3, 6, 9, 11, 12, 13, 15, 16, 18, 21, 22, 24, 25, 26, 27, 28]:  # Privacy
            drivers_sec_living[2] = 1
        if drivers_comb_living in [4, 7, 9, 10, 12, 14, 15, 16, 19, 21, 23, 24, 26, 28, 29, 30]:  # security
            drivers_sec_living[3] = 1
        if drivers_comb_living in [5, 8, 10, 11, 13, 14, 15, 16, 20, 22, 23, 24, 27, 28, 30, 31]:  # Visual comfort
            drivers_sec_living[4] = 1
        # Bedroom
        probs = np.loadtxt(directory_data_shading+'/drivers_sec_bedroom.txt', float)
        drivers_comb_bedroom = get_probability(np.random.random(), probs[:, driver_bedroom - 1])
        drivers_sec_bedroom = np.zeros(5)
        if drivers_comb_bedroom in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:  # Darkness
            drivers_sec_bedroom[0] = 1
        if drivers_comb_bedroom in [2, 6, 5, 8, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24]:  # Thermal comfort
            drivers_sec_bedroom[1] = 1
        if drivers_comb_bedroom in [3, 6, 9, 11, 12, 13, 15, 16, 18, 21, 22, 24, 25, 26, 27, 28]:  # Privacy
            drivers_sec_bedroom[2] = 1
        if drivers_comb_bedroom in [4, 7, 9, 10, 12, 14, 15, 16, 19, 21, 23, 24, 26, 28, 29, 30]:  # security
            drivers_sec_bedroom[3] = 1
        if drivers_comb_bedroom in [5, 8, 10, 11, 13, 14, 15, 16, 20, 22, 23, 24, 27, 28, 30, 31]:  # Visual comfort
            drivers_sec_bedroom[4] = 1
        # Bathroom
        probs = np.loadtxt(directory_data_shading+'/drivers_sec_bathroom.txt', float)
        drivers_comb_bathroom = get_probability(np.random.random(), probs[:, driver_bathroom - 1])
        drivers_sec_bathroom = np.zeros(5)
        if drivers_comb_bathroom in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:  # Darkness
            drivers_sec_bathroom[0] = 1
        if drivers_comb_bathroom in [2, 6, 5, 8, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24]:  # Thermal comfort
            drivers_sec_bathroom[1] = 1
        if drivers_comb_bathroom in [3, 6, 9, 11, 12, 13, 15, 16, 18, 21, 22, 24, 25, 26, 27, 28]:  # Privacy
            drivers_sec_bathroom[2] = 1
        if drivers_comb_bathroom in [4, 7, 9, 10, 12, 14, 15, 16, 19, 21, 23, 24, 26, 28, 29, 30]:  # security
            drivers_sec_bathroom[3] = 1
        if drivers_comb_bathroom in [5, 8, 10, 11, 13, 14, 15, 16, 20, 22, 23, 24, 27, 28, 30, 31]:  # Visual comfort
            drivers_sec_bathroom[4] = 1
        # Others
        # If a driver is present in all the previous room types (i.e. living rooms, bedrooms and bathrooms), it is considered as independent of the room type and also adopted for the rooms with short present (i.e. other rooms).
        drivers_sec_other = np.zeros(5)
        for i in range(5):
            if drivers_sec_living[i] == 1 and drivers_sec_bedroom[i] == 1 and drivers_sec_bathroom[i] == 1:
                drivers_sec_other[i] = 1
        # Add the main driver.
        drivers_sec_other[driver_other - 1] = 1
        # Step 2c. Define the trends in relation to the different seasons.
        # (1) Never, (2) several times a month, (3) several times a week, (4) daily or (5) multiple times a day
        # Spring/autumn
        probs = np.loadtxt(directory_data_shading+'/number_interactions_springautumn.txt', float)
        number_interactions_springautumn = get_probability(np.random.random(), probs[:, number_interactions - 1])
        # Winter
        probs = np.loadtxt(directory_data_shading+'/number_interactions_winter.txt', float)
        number_interactions_winter = get_probability(np.random.random(), probs[:, number_interactions_springautumn - 1])
        # Summer
        probs = np.loadtxt(directory_data_shading+'/number_interactions_summer.txt', float)
        number_interactions_summer = get_probability(np.random.random(), probs[:, number_interactions_springautumn - 1])

        # Step 3. Set the occlusion rate and the probability functions/thresholds per key moment and room
        # Step 3a. Define the occlusion rate in function of the different drivers.
        # (1) 25% occlusion rate, (2) 50% occlusion rate, (3) 75% occlusion rate and (4) fully closed.
        probs = np.loadtxt(directory_data_shading+'/occlusionrate.txt', float)
        # Evaluate first whether a specific driver is present in the family routines.
        # Darkness
        if drivers_sec_living[0] == 1 or drivers_sec_bedroom[0] == 1 or drivers_sec_bathroom[0] == 1 or \
                drivers_sec_other[0] == 1:
            occlusion_darkness = get_probability(np.random.random(), probs[:, 0])*0.25
        else:
            occlusion_darkness = 0
        # Privacy
        if drivers_sec_living[2] == 1 or drivers_sec_bedroom[2] == 1 or drivers_sec_bathroom[2] == 1 or \
                drivers_sec_other[2] == 1:
            occlusion_privacy = get_probability(np.random.random(), probs[:, 1])*0.25
        else:
            occlusion_privacy = 0
        # security
        if drivers_sec_living[3] == 1 or drivers_sec_bedroom[3] == 1 or drivers_sec_bathroom[3] == 1 or \
                drivers_sec_other[3] == 1:
            occlusion_security = get_probability(np.random.random(), probs[:, 2])*0.25
        else:
            occlusion_security = 0
        self.occlusion_darkness = occlusion_darkness
        self.occlusion_privacy = occlusion_privacy
        self.occlusion_security = occlusion_security
        # Step 3b. Characterise the family by checking the consistency of the defined habits as well as their activeness in adaptations.
        # The defined routines are checked in order to be consistent and adaptations to the darkness/security routine are made.
        drivers = [driver_living, driver_bedroom, driver_bathroom, driver_other]
        drivers_sec_dark = [drivers_sec_living[0], drivers_sec_bedroom[0], drivers_sec_bathroom[0],
                            drivers_sec_other[0]]
        drivers_sec_security = [drivers_sec_living[3], drivers_sec_bedroom[3], drivers_sec_bathroom[3],
                            drivers_sec_other[3]]
        if number_interactions == 1:  # Never
            # Darkness/Security can not be included as driver.
            # Evaluate whether Darkness/Security is included as main driver in the rooms and remove this driver.
            while drivers.count(1) > 0:
                index_one = [i for i, val in enumerate(drivers) if val == 1]
                index_zero = random.choice(index_one)
                drivers[index_zero] = 0
                drivers_sec_dark[index_zero] = 0
            [drivers_sec_living[0], drivers_sec_bedroom[0], drivers_sec_bathroom[0],
             drivers_sec_other[0]] = drivers_sec_dark
            while drivers.count(4) > 0:
                index_one = [i for i, val in enumerate(drivers) if val == 1]
                index_zero = random.choice(index_one)
                drivers[index_zero] = 0
                drivers_sec_security[index_zero] = 0
            [drivers_sec_living[3], drivers_sec_bedroom[3], drivers_sec_bathroom[3],
             drivers_sec_other[3]] = drivers_sec_security
            # Check whether Darkness/Security is included in the drivers and if so, remove them except if this is the main driver in the room.
            if drivers_sec_living[0] == 1 and driver_living != 1:
                drivers_sec_living[0] = 0
            if drivers_sec_bedroom[0] == 1 and driver_bedroom != 1:
                drivers_sec_bedroom[0] = 0
            if drivers_sec_bathroom[0] == 1 and driver_bathroom != 1:
                drivers_sec_bathroom[0] = 0
            if drivers_sec_other[0] == 1 and driver_other != 1:
                drivers_sec_other[0] = 0
            if drivers_sec_living[3] == 1 and driver_living != 4:
                drivers_sec_living[3] = 0
            if drivers_sec_bedroom[3] == 1 and driver_bedroom != 4:
                drivers_sec_bedroom[3] = 0
            if drivers_sec_bathroom[3] == 1 and driver_bathroom != 4:
                drivers_sec_bathroom[3] = 0
            if drivers_sec_other[3] == 1 and driver_other != 4:
                drivers_sec_other[3] = 0
        elif number_interactions == 2:  # Several times per month
            # Darkness and/or Security can be included in maximum one room type as driver.
            # Evaluate whether Darkness and/or Security, is included as main driver in the rooms and remove this driver.
            if drivers.count(1) + drivers.count(4) > 1:
                while drivers.count(1) + drivers.count(4) > 1:
                    index_one = [i for i, val in enumerate(drivers) if val == 1 or val == 4]
                    index_zero = random.choice(index_one)
                    drivers[index_zero] = 0
                    drivers_sec_dark[index_zero] = 0
                [drivers_sec_living[0], drivers_sec_bedroom[0], drivers_sec_bathroom[0],
                 drivers_sec_other[0]] = drivers_sec_dark
                [drivers_sec_living[3], drivers_sec_bedroom[3], drivers_sec_bathroom[3],
                drivers_sec_other[3]] = drivers_sec_security
            # Check whether Darkness/Security is included in the drivers and if so, remove them except if this is the main driver in the room.
            else:
                if drivers_sec_living[0] == 1 and driver_living != 1 and driver_living != 4:
                    drivers_sec_living[0] = 0
                if drivers_sec_bedroom[0] == 1 and driver_bedroom != 1 and driver_bedroom != 4:
                    drivers_sec_bedroom[0] = 0
                if drivers_sec_bathroom[0] == 1 and driver_bathroom != 1 and driver_bathroom != 4:
                    drivers_sec_bathroom[0] = 0
                if drivers_sec_other[0] == 1 and driver_other != 1 and driver_other != 4:
                    drivers_sec_other[0] = 0
                if drivers_sec_living[3] == 1 and driver_living != 1 and driver_living != 4:
                    drivers_sec_living[3] = 0
                if drivers_sec_bedroom[3] == 1 and driver_bedroom != 1 and driver_bedroom != 4:
                    drivers_sec_bedroom[3] = 0
                if drivers_sec_bathroom[3] == 1 and driver_bathroom != 1 and driver_bathroom != 4:
                    drivers_sec_bathroom[3] = 0
                if drivers_sec_other[3] == 1 and driver_other != 1 and driver_other != 4:
                    drivers_sec_other[3] = 0
        elif number_interactions == 3:  # Several times per week
            # Darkness/Security can be included in maximum two room type as driver.
            # Evaluate whether Darkness/Security is included as main driver in the rooms and remove this driver.
            if drivers.count(1) + drivers.count(4) > 2:
                while drivers.count(1) + drivers.count(4) > 2:
                    index_one = [i for i, val in enumerate(drivers) if val == 1 or val == 4]
                    index_zero = random.choice(index_one)
                    drivers[index_zero] = 0
                    drivers_sec_dark[index_zero] = 0
                [drivers_sec_living[0], drivers_sec_bedroom[0], drivers_sec_bathroom[0],
                 drivers_sec_other[0]] = drivers_sec_dark
                [drivers_sec_living[3], drivers_sec_bedroom[3], drivers_sec_bathroom[3],
                drivers_sec_other[3]] = drivers_sec_security
            # Check whether Darkness/Security is included in the drivers and if so, remove them except if this is the main driver in the room.
            else:
                if drivers_sec_living[0] == 1 and driver_living != 1 and driver_living != 4:
                    drivers_sec_living[0] = 0
                if drivers_sec_bedroom[0] == 1 and driver_bedroom != 1 and driver_bedroom != 4:
                    drivers_sec_bedroom[0] = 0
                if drivers_sec_bathroom[0] == 1 and driver_bathroom != 1 and driver_bathroom != 4:
                    drivers_sec_bathroom[0] = 0
                if drivers_sec_other[0] == 1 and driver_other != 1 and driver_other != 4:
                    drivers_sec_other[0] = 0
                if drivers_sec_living[3] == 1 and driver_living != 1 and driver_living != 4:
                    drivers_sec_living[3] = 0
                if drivers_sec_bedroom[3] == 1 and driver_bedroom != 1 and driver_bedroom != 4:
                    drivers_sec_bedroom[3] = 0
                if drivers_sec_bathroom[3] == 1 and driver_bathroom != 1 and driver_bathroom != 4:
                    drivers_sec_bathroom[3] = 0
                if drivers_sec_other[3] == 1 and driver_other != 1 and driver_other != 4:
                    drivers_sec_other[3] = 0
        self.drivers_sec = {'living': drivers_sec_living, 'bedroom': drivers_sec_bedroom, 'bathroom': drivers_sec_bathroom, 'other': drivers_sec_other}
        drivers_sec_darkness = [drivers_sec_living[0], drivers_sec_bedroom[0], drivers_sec_bathroom[0],
                 drivers_sec_other[0]]
        drivers_sec_security = [drivers_sec_living[3], drivers_sec_bedroom[3], drivers_sec_bathroom[3],
                drivers_sec_other[3]] 
        drivers_sec_thermal = [drivers_sec_living[1], drivers_sec_bedroom[1], drivers_sec_bathroom[1],
                drivers_sec_other[1]] 
        drivers_sec_visual = [drivers_sec_living[4], drivers_sec_bedroom[4], drivers_sec_bathroom[4],
                drivers_sec_other[4]] 
        
        # Define the activeness of the family to respont to visual and thermal discomfort.
        # Limited number of interactions related to Darkness/Security
        if number_interactions == 1: 
            if drivers_sec_visual.count(1) > 0: 
                upper_border_visual = 0.1*max(number_interactions_winter, number_interactions_springautumn)
                lower_border_visual = 0
            else: 
                upper_border_visual = 0.1
                lower_border_visual = 0
            if drivers_sec_thermal.count(1) > 0: 
                upper_border_thermal = 0.1*number_interactions_summer
                lower_border_thermal = 0
            else:
                upper_border_thermal = 0.1
                lower_border_thermal = 0
        elif number_interactions <= 3 and (drivers.count(2) + drivers.count(5)) == 0:
            if drivers_sec_visual.count(1) > 0:
                upper_border_visual = 0.2 * (max(1, 
                            max(number_interactions_winter, number_interactions_springautumn) - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2))
                lower_border_visual = 0.2 * (
                      max(1, (max(number_interactions_winter, number_interactions_springautumn) - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2)) - 1)
            else: 
                upper_border_visual = 0.1
                lower_border_visual = 0
            if drivers_sec_thermal.count(1) > 0:    
                upper_border_thermal = 0.2 * (max(1, number_interactions_summer - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2))
                lower_border_thermal = 0.2 * (max(0, number_interactions_summer - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2)-1)
            else:
                upper_border_thermal = 0.1
                lower_border_thermal = 0
        elif number_interactions <= 3 and drivers.count(2) == 0:
            upper_border_visual = 0.2 * (max(number_interactions_winter, number_interactions_springautumn))
            lower_border_visual = 0.2 * (max(number_interactions_winter, number_interactions_springautumn)-1)
            if drivers_sec_thermal.count(1) > 0: 
                upper_border_thermal = 0.2 * (max(1, number_interactions_summer - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2))
                lower_border_thermal = 0.2 * (max(0, number_interactions_summer - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2)-1)
            else: 
                upper_border_thermal = 0.1
                lower_border_thermal = 0
        elif number_interactions <= 3 and drivers.count(5) == 0:
            if drivers_sec_visual.count(1) > 0:
                upper_border_visual = 0.2 * (max(1, 
                            max(number_interactions_winter, number_interactions_springautumn) - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2))
                lower_border_visual = 0.2 * (
                           max(1, (max(number_interactions_winter, number_interactions_springautumn) - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2))-1)
            else: 
                upper_border_visual = 0.1
                lower_border_visual = 0
            upper_border_thermal = 0.2 * (number_interactions_summer)
            lower_border_thermal = 0.2 * (number_interactions_summer-1)
        elif number_interactions <= 3: 
            upper_border_visual = 0.2 * (max(number_interactions_winter, number_interactions_springautumn))
            lower_border_visual = 0.2 * (max(number_interactions_winter, number_interactions_springautumn)-1)
            upper_border_thermal = 0.2 * (number_interactions_summer)
            lower_border_thermal = 0.2 * (number_interactions_summer-1)
        elif number_interactions == 4 and drivers_sec_darkness.count(1) +drivers_sec_security.count(1) <= 3 and drivers.count(2) + drivers.count(5) == 0:
            if drivers_sec_visual.count(1) > 0: 
                upper_border_visual = 0.2 * (max(1, 
                            max(number_interactions_winter, number_interactions_springautumn) - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2))
                lower_border_visual = 0.2 * (
                          max(1, (max(number_interactions_winter, number_interactions_springautumn) - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2)) - 1)
            else: 
                upper_border_visual = 0.2
                lower_border_visual = 0
            if drivers_sec_thermal.count(1) > 0:
                upper_border_thermal = 0.2 * (max(1, number_interactions_summer - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2))
                lower_border_thermal = 0.2 * (max(0, number_interactions_summer - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2)-1)
            else: 
                upper_border_thermal = 0.2
                lower_border_thermal = 0
        elif number_interactions == 4 and drivers_sec_darkness.count(1) +drivers_sec_security.count(1) <= 3 and drivers.count(2) == 0:
            upper_border_visual = 0.2 * (max(number_interactions_winter, number_interactions_springautumn))
            lower_border_visual = 0.2 * (max(number_interactions_winter, number_interactions_springautumn)-1)
            if drivers_sec_thermal.count(1) > 0:
                upper_border_thermal = 0.2 * (max(1, number_interactions_summer - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2))
                lower_border_thermal = 0.2 * (max(0, number_interactions_summer - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2)-1)
            else: 
                upper_border_thermal = 0.2
                lower_border_thermal = 0
        elif number_interactions == 4 and drivers_sec_darkness.count(1) +drivers_sec_security.count(1) <= 3 and drivers.count(5) == 0:
            if drivers_sec_visual.count(1) > 0:
                upper_border_visual = 0.2 * (max(1, 
                            max(number_interactions_winter, number_interactions_springautumn) - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2))
                lower_border_visual = 0.2 * (
                          max(1, (max(number_interactions_winter, number_interactions_springautumn) - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2))-1)
            else: 
                upper_border_visual = 0.2
                lower_border_visual = 0        
            upper_border_thermal = 0.2 * (number_interactions_summer)
            lower_border_thermal = 0.2 * (number_interactions_summer-1)
        elif number_interactions == 4 and drivers_sec_darkness.count(1) + drivers_sec_security.count(1) <= 3: 
            upper_border_visual = 0.2 * (max(number_interactions_winter, number_interactions_springautumn))
            lower_border_visual = 0.2 * (max(number_interactions_winter, number_interactions_springautumn)-1)
            upper_border_thermal = 0.2 * (number_interactions_summer)
            lower_border_thermal = 0.2 * (number_interactions_summer-1)
        elif number_interactions == 4 and drivers.count(2) + drivers.count(5) == 0:
            if drivers_sec_visual.count(1) > 0:
                upper_border_visual = 0.1 * (max(1, 
                            max(number_interactions_winter, number_interactions_springautumn) - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2))
                lower_border_visual = 0.1 * (
                          max(1, (max(number_interactions_winter, number_interactions_springautumn) - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2)) - 1)
            else: 
                upper_border_visual = 0.2
                lower_border_visual = 0
            if drivers_sec_thermal.count(1) > 0:
                upper_border_thermal = 0.1 * (max(1, number_interactions_summer - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2))
                lower_border_thermal = 0.1 * (max(1, number_interactions_summer - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2)-1)
            else: 
                upper_border_thermal = 0.2
                lower_border_thermal = 0
        elif number_interactions == 4 and drivers.count(2) == 0:
            upper_border_visual = 0.2 * (max(number_interactions_winter, number_interactions_springautumn))
            lower_border_visual = 0.2 * (max(number_interactions_winter, number_interactions_springautumn)-1)
            if drivers_sec_thermal.count(1) > 0:
                upper_border_thermal = 0.1 * (max(1, number_interactions_summer - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2))
                lower_border_thermal = 0.1 * (max(1, number_interactions_summer - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2)-1)
            else: 
                upper_border_thermal = 0.2
                lower_border_thermal = 0
        elif number_interactions == 4 and drivers.count(5) == 0:
            if drivers_sec_visual.count(1) > 0:
                upper_border_visual = 0.1 * (max(1, 
                            max(number_interactions_winter, number_interactions_springautumn) - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2))
                lower_border_visual = 0.1 * (
                          max(1, (max(number_interactions_winter, number_interactions_springautumn) - (drivers_sec_dark.count(1) + drivers_sec_security.count(4))/2)) - 1)
            else: 
                upper_border_visual = 0.2
                lower_border_visual = 0
            upper_border_thermal = 0.2 * (number_interactions_summer)
            lower_border_thermal = 0.2 * (number_interactions_summer-1)
        elif number_interactions == 4: 
            upper_border_visual = 0.2 * (max(number_interactions_winter, number_interactions_springautumn))
            lower_border_visual = 0.2 * (max(number_interactions_winter, number_interactions_springautumn)-1)
            upper_border_thermal = 0.2 * (number_interactions_summer)
            lower_border_thermal = 0.2 * (number_interactions_summer-1)
        elif number_interactions == 5 and drivers.count(2) + drivers.count(5) == 0:
            if drivers_sec_visual.count(1) > 0:
                upper_border_visual = 0.8
                lower_border_visual = 0.4
            else: 
                upper_border_visual = 0.4
                lower_border_visual = 0
            if drivers_sec_thermal.count(1) > 0:
                upper_border_thermal = 0.8
                lower_border_thermal = 0.4
            else:
                upper_border_thermal = 0.4
                lower_border_thermal = 0
        elif number_interactions == 5 and drivers.count(2) == 0:
            upper_border_visual = 0.2 * (max(number_interactions_winter, number_interactions_springautumn))
            lower_border_visual = 0.2 * (max(number_interactions_winter, number_interactions_springautumn) - 1)
            if drivers_sec_thermal.count(1) > 0:
                upper_border_thermal = 0.8
                lower_border_thermal = 0.4
            else:
                upper_border_thermal = 0.4
                lower_border_thermal = 0
        elif number_interactions == 5 and drivers.count(5) == 0:
            if drivers_sec_visual.count(1) > 0:
                upper_border_visual = 0.8
                lower_border_visual = 0.4
            else: 
                upper_border_visual = 0.4
                lower_border_visual = 0
            upper_border_thermal = 0.2 * (number_interactions_summer)
            lower_border_thermal = 0.2 * (number_interactions_summer - 1)
        elif number_interactions == 5:
            upper_border_visual = 0.2 * (max(number_interactions_winter, number_interactions_springautumn))
            lower_border_visual = 0.2 * (max(number_interactions_winter, number_interactions_springautumn) - 1)
            upper_border_thermal = 0.2 * (number_interactions_summer)
            lower_border_thermal = 0.2 * (number_interactions_summer - 1)
        
        # Step 3c. Formulate the thresholds and probabilities for thermal and visual discomfort.          
        # Import the lower and upper probability functions
        close_during_visual_irradiance_bounds = pd.read_csv(directory_data_shading +'/close_during_visual_irradiance.txt', delimiter='\t', dtype=str, comment='#', header = None)
        close_entering_visual_irradiance_bounds = pd.read_csv(directory_data_shading +'/close_entering_visual_irradiance.txt', delimiter='\t', dtype=str, comment='#', header = None)
        open_during_visual_irradiance_bounds = pd.read_csv(directory_data_shading +'/open_during_visual_irradiance.txt', delimiter='\t', dtype=str, comment='#', header = None)
        open_leaving_visual_irradiance_bounds = pd.read_csv(directory_data_shading +'/open_leaving_visual_irradiance.txt', delimiter='\t', dtype=str, comment='#', header = None)
        close_during_visual_glare_bounds = pd.read_csv(directory_data_shading +'/close_during_visual_glare.txt', delimiter='\t', dtype=str, comment='#', header = None)
        close_entering_visual_glare_bounds = pd.read_csv(directory_data_shading +'/close_entering_visual_glare.txt', delimiter='\t', dtype=str, comment='#', header = None)
        close_during_thermal_bounds = pd.read_csv(directory_data_shading +'/close_during_thermal.txt', delimiter='\t', dtype=str, comment='#', header = None)
        close_entering_thermal_bounds = pd.read_csv(directory_data_shading +'/close_entering_thermal.txt', delimiter='\t', dtype=str, comment='#', header = None)
        probability_visual_entering_irradiance = {'open':{}, 'close':{}}
        probability_visual_leaving_irradiance = {'open':{}, 'close':{}}
        probability_visual_during_irradiance = {'open':{}, 'close':{}}
        probability_visual_entering_occupation_irradiance = {'open':{}, 'close':{}}
        probability_visual_leaving_occupation_irradiance = {'open':{}, 'close':{}}
        probability_visual_entering_glare = {'open':{}, 'close':{}}
        probability_visual_leaving_glare = {'open':{}, 'close':{}}
        probability_visual_during_glare = {'open':{}, 'close':{}}
        probability_visual_entering_occupation_glare = {'open':{}, 'close':{}}
        probability_visual_leaving_occupation_glare = {'open':{}, 'close':{}}
        probability_thermal_entering = {'open':{}, 'close':{}}
        probability_thermal_leaving = {'open':{}, 'close':{}}
        probability_thermal_during = {'open':{}, 'close':{}}
        probability_thermal_entering_occupation = {'open':{}, 'close':{}}
        probability_thermal_leaving_occupation = {'open':{}, 'close':{}}
        # Repeat the following steps for every room/zone in the simulation. 
        for zone_name in rooms_shading:
            # Define the type of room and corresponding index
            # (1) Bathroom, (2) Bedroom, (3) Main living rooms (e.g. living, kitchen, office), (4) Rooms with infrequent and irregular presence (e.g. hallway, toilet, storage)
            if 'Bathroom' in zone_name: 
                room_type = 1
                room_habits = 'bathroom'
            elif 'Bedroom' in zone_name and zone_name not in offices_list: 
                room_type = 2
                room_habits = 'bedroom'
            elif 'Corridor' in zone_name or 'Hallway' in zone_name: 
                room_type = 4
                room_habits = 'other'
            elif 'Kitchen' in zone_name and not 'Living' in zone_name: 
                room_type = 3
                room_habits = 'living'
            elif 'Living' in zone_name and not 'Kitchen' in zone_name: 
                room_type = 3
                room_habits = 'living'
            elif zone_name in offices_list: 
                room_type = 3
                room_habits = 'living'
            elif 'Toilet' in zone_name or 'Storage' in zone_name: 
                room_type = 4
                room_habits = 'other'
            elif 'LivingKitchen' in zone_name:
                # Assumption: the gathered data for Living and Kitchen are combined and the borders are set in to the extrema. 
                room_type = 3
                room_habits = 'living'
                    
            # Visual discomfort
            # In function of irradiance (linear)
            # First occupant entering the room
            close_entering_visual_irradiance_lower = close_entering_visual_irradiance_bounds.iloc[room_type - 1,0]
            close_entering_visual_irradiance_upper = close_entering_visual_irradiance_bounds.iloc[room_type - 1,1]
            probability_visual_entering_irradiance['close'][zone_name] = generate_random_linear_probability_function_shading(close_entering_visual_irradiance_lower, close_entering_visual_irradiance_upper, interval_range = (lower_border_visual, upper_border_visual), increasing = True)
            # Last occupant leaving the room 
            open_leaving_visual_irradiance_lower = open_leaving_visual_irradiance_bounds.iloc[room_type - 1,0]
            open_leaving_visual_irradiance_upper = open_leaving_visual_irradiance_bounds.iloc[room_type - 1,1]
            probability_visual_leaving_irradiance['open'][zone_name] = generate_random_linear_probability_function_shading(open_leaving_visual_irradiance_lower, open_leaving_visual_irradiance_upper, interval_range = (lower_border_visual, upper_border_visual), increasing = False)
            # During occupation
            close_during_visual_irradiance_lower = close_during_visual_irradiance_bounds.iloc[room_type - 1,0]
            close_during_visual_irradiance_upper = close_during_visual_irradiance_bounds.iloc[room_type - 1,1]
            probability_visual_during_irradiance['close'][zone_name] = generate_random_linear_probability_function_shading(close_during_visual_irradiance_lower, close_during_visual_irradiance_upper, interval_range = (lower_border_visual, upper_border_visual), increasing = True)
            open_during_visual_irradiance_lower = open_during_visual_irradiance_bounds.iloc[room_type - 1,0]
            open_during_visual_irradiance_upper = open_during_visual_irradiance_bounds.iloc[room_type - 1,1]
            probability_visual_during_irradiance['open'][zone_name] = generate_random_linear_probability_function_shading(open_during_visual_irradiance_lower, open_during_visual_irradiance_upper, interval_range = (lower_border_visual, upper_border_visual), increasing = False)
            # Additional critical moments, secondary deviated 
            # First occupant enters the room - shading closed
            # Assumption: The shading is reopened when entering: the function for closing them is mirrored around half the opening probability at irradiance 0 when leaving + 0-10%. 
            mirror_value = (generate_linear_function(probability_visual_leaving_irradiance['open'][zone_name])(0) + random.uniform(0,0.1))/2
            probability_visual_entering_irradiance['open'][zone_name] = reflect_linear_function(mirror_value, probability_visual_entering_irradiance['close'][zone_name]['a'],probability_visual_entering_irradiance['close'][zone_name]['b'] )
            # Last occupant leaves the room - shading opened
            # Assumption: The occupant does not close the shading for visual discomfort reasons. 
            probability_visual_leaving_irradiance['close'][zone_name] = {'a': 0, 'b': -2.0}
            # Additional occupant enters the room
            # Opening
            # Assumption: The occupant is interacting more effectively than during occupation - however takes into account the presence of another household member. It is assumed that the probability is located between the closing probability during + 15 and the probability when entering - 15%. 
            probability_visual_entering_occupation_irradiance['open'][zone_name] = generate_random_linear_probability_function_shading(probability_visual_during_irradiance['open'][zone_name], probability_visual_entering_irradiance['open'][zone_name], interval_range = (0.15,0.85), increasing = False)
            # Closing
            # Assumption: This occupant interacts independently
            probability_visual_entering_occupation_irradiance['close'][zone_name] = probability_visual_entering_irradiance['close'][zone_name]
            # Additional occupant leaves the room
            # Opening
            # Assumption: an occupant will occasionally open the shading when leaving as this requires limited additional effort, but affects the comfort of his household members. It is assumed that this probability is increasing with increasing irradiance and is located between the closing probability for during + 15% and the mirrored opening probability - 15%.  
            mirror_value = generate_linear_function(probability_visual_leaving_irradiance['open'][zone_name])(0)/2
            probability_visual_leaving_occupation_irradiance['open'][zone_name] = generate_random_linear_probability_function_shading(probability_visual_during_irradiance['open'][zone_name], reflect_linear_function(mirror_value, probability_visual_entering_irradiance['close'][zone_name]['a'], probability_visual_entering_irradiance['close'][zone_name]['b']), interval_range = (0.15,0.85), increasing = False)
            # Closing
            # Assumption: the probability is located between the probability when entering the room as first and interacting during occupation, with the difference that this person is now not acting for themself. Therefore, a probability function in the lowest quart of the interval between both functions is suggested. 
            probability_visual_leaving_occupation_irradiance['close'][zone_name] = generate_random_linear_probability_function_shading(probability_visual_during_irradiance['close'][zone_name], probability_visual_entering_irradiance['close'][zone_name], interval_range = (0,0.25), increasing = True)
            
            # In function of glare (exponential)
            # Assumption: glare is only evaluated to close the shading. To reopen, the irradiance is taken into account. 
            # First occupant entering the room
            close_entering_visual_glare_lower = close_entering_visual_glare_bounds.iloc[room_type - 1, 0]
            close_entering_visual_glare_upper = close_entering_visual_glare_bounds.iloc[room_type - 1, 1]
            probability_visual_entering_glare['close'][zone_name] = generate_random_exponential_probability_function_shading(close_entering_visual_glare_lower, close_entering_visual_glare_upper,  np.linspace(0, 14, 100), interval_range = (lower_border_visual, upper_border_visual))
            # During occupation
            close_during_visual_glare_lower = close_during_visual_glare_bounds.iloc[room_type - 1, 0]
            close_during_visual_glare_upper = close_during_visual_glare_bounds.iloc[room_type - 1, 1]
            probability_visual_during_glare['close'][zone_name] = generate_random_exponential_probability_function_shading(close_during_visual_glare_lower, close_during_visual_glare_upper,  np.linspace(0, 14, 100), interval_range = (lower_border_visual, upper_border_visual))
            # Additional critical moments, secondary deviated
            # Last occupant leaves the room - shading opened
            # Assumption: The occupant does not close the shading for visual discomfort reasons. 
            probability_visual_leaving_glare['close'][zone_name] = {'a':-2.0, 'b': 0, 'c': 0}
            # Additional occupant enters the room
            # Assumption: This occupant interacts independently
            probability_visual_entering_occupation_glare['close'][zone_name] = probability_visual_entering_glare['close'][zone_name]
            # Additional occupant leaves the room
            # Assumption: the probability is located between the probability when entering the room as first and interacting during occupation, with the difference that this person is now not acting for themself. Therefore, a probability function in the lowest quart of the interval between both functions is suggested. 
            probability_visual_leaving_occupation_glare['close'][zone_name] = generate_random_exponential_probability_function_shading(probability_visual_during_glare['close'][zone_name], probability_visual_entering_glare['close'][zone_name], np.linspace(0, 14, 100), interval_range = (0,0.25), increasing = True)
            
            # Thermal discomfort (exponential)
            # In function of indoor temperature
            # Assumption: temperature is only evaluated to close the shading (in combination with irradiance). To reopen, the irradiance is taken into account. 
            # First occupant entering the room
            close_entering_thermal_lower = close_entering_thermal_bounds.iloc[room_type - 1, 0]
            close_entering_thermal_upper = close_entering_thermal_bounds.iloc[room_type - 1, 1]
            probability_thermal_entering['close'][zone_name] = generate_random_exponential_probability_function_shading(close_entering_thermal_lower, close_entering_thermal_upper, np.linspace(0, 6, 100), interval_range = (lower_border_thermal, upper_border_thermal), increasing = True)
            # During occupation
            close_during_thermal_lower = close_during_thermal_bounds.iloc[room_type - 1, 0]
            close_during_thermal_upper = close_during_thermal_bounds.iloc[room_type - 1, 1]
            probability_thermal_during['close'][zone_name] = generate_random_exponential_probability_function_shading(close_during_thermal_lower, close_during_thermal_upper, np.linspace(0, 6, 100), interval_range = (lower_border_thermal, upper_border_thermal), increasing = True)
            # Additional critical moments, secondary deviated
            # Last occupant leaves the room - shading opened
            # Assumption: A similar probability as entering is proposed as occupants have adapted to the elevated temperatures, but do no longer have to take into account the daylighting entrance. 
            probability_thermal_leaving['close'][zone_name] = probability_thermal_entering['close'][zone_name]
            # Additional occupant enters the room
            # Assumption: the occupant takes into account his household members: The probability is located between entering and during, within the 25%-75% interval. 
            probability_thermal_entering_occupation['close'][zone_name] = generate_random_exponential_probability_function_shading(probability_thermal_during['close'][zone_name], probability_thermal_entering['close'][zone_name], np.linspace(0, 6, 100), interval_range =(0.25,0.75), increasing = True)
            # Additional occupant leaves the room
            # Assumption: the occupant takes into account his household members: The probability is located between entering and during, within the 0-25% interval as they have adapted to the temperatures. 
            probability_thermal_leaving_occupation['close'][zone_name] = generate_random_exponential_probability_function_shading(probability_thermal_during['close'][zone_name], probability_thermal_entering['close'][zone_name], np.linspace(0, 6, 100), interval_range =(0,0.25), increasing = True)
        self.probability_visual_entering_irradiance = probability_visual_entering_irradiance
        self.probability_visual_leaving_irradiance = probability_visual_leaving_irradiance
        self.probability_visual_during_irradiance = probability_visual_during_irradiance
        self.probability_visual_entering_occupation_irradiance = probability_visual_entering_occupation_irradiance
        self.probability_visual_leaving_occupation_irradiance = probability_visual_leaving_occupation_irradiance
        self.probability_visual_entering_glare = probability_visual_entering_glare
        self.probability_visual_leaving_glare = probability_visual_leaving_glare
        self.probability_visual_during_glare = probability_visual_during_glare
        self.probability_visual_entering_occupation_glare = probability_visual_entering_occupation_glare
        self.probability_visual_leaving_occupation_glare = probability_visual_leaving_occupation_glare
        self.probability_thermal_entering = probability_thermal_entering
        self.probability_thermal_leaving = probability_thermal_leaving
        self.probability_thermal_during = probability_thermal_during
        self.probability_thermal_entering_occupation = probability_thermal_entering_occupation
        self.probability_thermal_leaving_occupation = probability_thermal_leaving_occupation
            
        # Step 3d. Set the average standard deviation 
        self.standard_deviation = np.loadtxt(directory_data_shading+'/correction_deviation.txt')
        
        # Step 4. Generate the habits that are used to model the practical habits
        # Step 4a. Define 'darkness interactions' for the household
        # Assumption: shading will be operated as part of the morning and evening routine related to going to sleep/wakeing up and/or sunset/sunrise
        darkness_habits_overview = ['dark', 'anticipate dark', 'evening and morning routine']
        self.darkness_habits = random.choice(darkness_habits_overview)    
        # Step 4b. Define 'privacy interactions' for the household
        # Assumption: the shading is permanently closed or closed during presence. 
        if number_interactions == 5: 
            self.privacy_habits = 'operating by presence'
        else:
            self.privacy_habits = 'no interaction'
        # Step 4c. Define 'security interactions' for the household
        # Assumption: 
        security_shading_habits_overview = ['absence/asleep', 'dark and absence/asleep', 'anticipate dark and absence/asleep']
        self.security_shading_habits = random.choice(security_shading_habits_overview) 
        # Step 4d. Probability habits
        # Assumption: There is a 0-10% chance the inhabitants do not follow their habits (e.g. they are returning later than expected). 
        self.probability_darkness = random.uniform(0.90, 1.00)
        self.probability_privacy = random.uniform(0.90, 1.00)
        self.probability_security_shading = random.uniform(0.90, 1.00)
        
        # Step 5. Import probabilities markov chains occlusion rates
        # Assumption: the solar shading is always reopened fully. 
        occlusion_thermal_open = np.loadtxt(directory_data_shading+'\occlusion_thermal_open.txt', float)
        occlusion_thermal_close = np.loadtxt(directory_data_shading+'\occlusion_thermal_close.txt', float)
        occlusion_thermal_anticipate_open = np.loadtxt(directory_data_shading+'\occlusion_thermalanticipate_open.txt', float)
        occlusion_thermal_anticipate_close = np.loadtxt(directory_data_shading+'\occlusion_thermalanticipate_close.txt', float)
        occlusion_visual_open = np.loadtxt(directory_data_shading+'\occlusion_visual_open.txt', float)
        occlusion_visual_close = np.loadtxt(directory_data_shading+'\occlusion_visual_close.txt', float)
        self.probabilities_occlusion = {'thermal': {'open': occlusion_thermal_open, 'close': occlusion_thermal_close}, 'thermal_anticipate': {'open': occlusion_thermal_anticipate_open, 'close': occlusion_thermal_anticipate_close},'visual': {'open': occlusion_visual_open, 'close': occlusion_visual_close}}
            
        # Step 6. Define comfortable glare and indoor temperature levels per timestep. 
        # Step 6a. Define the extent of glare discomfort in relation to the activities.  
        # Assumption: Variation of acceptable glare in function of activities from EN 12464 is adopted. 
        glare_evaluation = {'Living':{}, 'Kitchen':{}, 'LivingKitchen': {}, 'Bedroom1':{}, 'Bedroom2':{}, 'Bedroom3':{}, 'Corridor':{}, 'Bathroom':{}, 'Storage': {}, 'ToiletGround': {}, 'ToiletFirst': {}}
        for i in range(task_activities_dataframe.shape[0]): 
            # Kitchen
            if task_activities['cook'][i] > 0 or task_activities['dishes'][i] > 0:
                glare_evaluation['Kitchen'].update({i:22})
            else:
                glare_evaluation['Kitchen'].update({i:25})
            # Living
            if task_activities['pcDayzone'][i] > 0 or task_activities['admDayzone'][i] > 0 or task_activities['tv'][i]: 
                glare_evaluation['Living'].update({i:19})
            else: 
                glare_evaluation['Living'].update({i:25})
            # LivingKitchen
            glare_evaluation['LivingKitchen'].update({i: max(glare_evaluation['Living'].get(i, 0), glare_evaluation['Kitchen'].get(i,0))})
            # Bedroom1
            if task_activities['pcBedroom1'][i] > 0 or task_activities['admBedroom1'][i] > 0: 
                glare_evaluation['Bedroom1'].update({i:19})
            else: 
                glare_evaluation['Bedroom1'].update({i:25})
            # Bedroom2
            if task_activities['pcBedroom2'][i] > 0 or task_activities['admBedroom2'][i] > 0: 
                glare_evaluation['Bedroom2'].update({i:19})
            else: 
                glare_evaluation['Bedroom2'].update({i:25})
            # Bedroom3
            if task_activities['pcBedroom3'][i] > 0 or task_activities['admBedroom3'][i] > 0: 
                glare_evaluation['Bedroom3'].update({i:19})
            else: 
                glare_evaluation['Bedroom3'].update({i:25})
            # Corridor
            glare_evaluation['Corridor'].update({i:25})
            # Bathroom
            glare_evaluation['Bathroom'].update({i:25})
            # Storage
            glare_evaluation['Storage'].update({i:25})
            #ToiletGround
            glare_evaluation['ToiletGround'].update({i:25})
            # ToiletFirst
            glare_evaluation['ToiletFirst'].update({i:25})
        self.glare_evaluation = glare_evaluation
            
        # Step 6b. Calculate the reference outdoor temperature per timestep. 
        with open(weather_file, "r") as f: 
            weather = f.readlines()
        outdoor_temperature = pd.concat([pd.DataFrame((line.split(",")[1:4] for line in weather[8:]), columns =["Month", "Date", "Time"]), pd.DataFrame((line.split(",")[6:7] for line in weather[8:]), columns = ["Temperature"])], axis = 1)
        outdoor_temperature["Datetime"] = outdoor_temperature.apply(lambda row: parse_datetime(row["Month"], row["Date"], row["Time"]), axis=1)
        outdoor_temperature["Temperature"] = outdoor_temperature["Temperature"].astype(float)
        daily_min_max_temperatures = outdoor_temperature.groupby(outdoor_temperature["Datetime"].dt.date)["Temperature"].agg(["min", "max"])
        daily_average_temperature = daily_min_max_temperatures.sum(axis=1)/2
        reference_external_temperature = np.zeros(len(daily_average_temperature))
        for i in range(len(daily_average_temperature)):
            if i > 2: 
                reference_external_temperature[i] = (daily_average_temperature[i] + 0.8*daily_average_temperature[i-1]+0.4*daily_average_temperature[i-2]+0.2*daily_average_temperature[i-3])/2.4
            elif i == 2: 
                reference_external_temperature[i] = (daily_average_temperature[i] + 0.8*daily_average_temperature[i-1]+0.4*daily_average_temperature[i-2]+0.2*daily_average_temperature[i-3+365])/2.4
            elif i == 1: 
                reference_external_temperature[i] = (daily_average_temperature[i] + 0.8*daily_average_temperature[i-1]+0.4*daily_average_temperature[i-2+365]+0.2*daily_average_temperature[i-3+365])/2.4
            elif i == 0: 
                reference_external_temperature[i] = (daily_average_temperature[i] + 0.8*daily_average_temperature[364]+0.4*daily_average_temperature[363]+0.2*daily_average_temperature[362])/2.4
        self.reference_external_temperature = reference_external_temperature
        
        # Step 6c. Calculate the differences in maximum outdoor temperature over three days, and give the corresponding maximum temperature. 
        # Future maximum temperatures (next 3 days)
        shifts = {
            1: daily_min_max_temperatures["max"].shift(-1),
            2: daily_min_max_temperatures["max"].shift(-2),
            3: daily_min_max_temperatures["max"].shift(-3),
        } 
        diffs = {}
        for i in [1,2,3]:
            diffs[i] = (shifts[i] - daily_min_max_temperatures["max"]).abs()
        differences = pd.DataFrame(diffs)
        # Calculate maximum difference
        max_diff_period = differences.idxmax(axis=1)
        daily_max_temperatures = {}
        daily_max_temperatures["max_day"] = daily_min_max_temperatures['max']
        daily_max_temperatures["max_diff_next_1_3_days"] = differences.max(axis=1)
        shift_df = pd.DataFrame(shifts)
        row_indices = np.arange(len(daily_min_max_temperatures))
        col_indices = max_diff_period.values[:-1].astype(int) - 1
        next_day_max_for_max_diff = shift_df.values[row_indices[:-1], col_indices]
        next_day_max_for_max_diff = np.append(next_day_max_for_max_diff,0)
        daily_max_temperatures["next_day_max_for_max_diff"] = pd.Series(next_day_max_for_max_diff, index = daily_min_max_temperatures.index)
        self.daily_max_temperatures = daily_max_temperatures

        # Set EnergyPlus installation
        IDF.setiddname(idd_path)
        self.idf = IDF(idf_path, weather_path)
        
        self.handles_set = False
        self.handle_illuminance = {}
        self.handle_previouslighting = {}
        self.handle_schedulelighting = {}
        self.handle_previousstateshading = {}
        self.handle_previoustimestepshading = {}
        self.handle_irradiance = {}
        self.handle_indooroperativetemperature = {}
        self.handle_glare = {}
        self.handle_darkness = {}
        self.handle_privacy = {}
        self.handle_security = {}
        self.handle_scheduleshading = {}
        self.handle_timestepinteraction = {}
        self.handle_timesteppreviousevaluation = {}
    
    # CONTROL PROGRAM
    
    def on_begin_timestep_before_predictor(self,state) -> int: 
        
        def simulate_manual_lighting(self, zone_name, hour, day, timestep, sun_up, previousstate_lighting, illuminance, shading_interaction): 
        
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
                
            state_lighting = 0
            
            # EVALUATION PRACTICAL REASONS (I.E. security_lighting)
            
            # Household shows a habit to switch on lighting for security_lighting reasons
            if self.reasons_not_off[4] == 1: 
                # security_lighting habits are only maintained in a limited selection of the rooms. 
                if zone_name in self.security_lighting_rooms or (zone_name == 'LivingKitchen' and ('Living' in self.security_lighting_rooms or 'Kitchen' in self.security_lighting_rooms)):
                    if timestep-1 >= 0:
                        # Switching lighting on
                        if previousstate_lighting == 0: 
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
                                    previousstate_lighting = state_lighting
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
                                        previousstate_lighting = state_lighting

            # EVALUATION PSYCHOLOGICAL REASONS

            # Evaluation only runs when lighting is not switched on due to security_lighting reasons
            if not state_lighting == 1:
                # Light ON during the previous timestep
                if previousstate_lighting == 1:
                    
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
                                k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting)
                                probability_timestep = generate_logarithmic_function(self.probability_off_leaving[zone_name])(time_next_occupancy) + k*self.standard_deviation_lighting
                                state_lighting = stochastic_switching(probability_timestep, previousstate_lighting)
                            # Nobody present in the room, the lighting remains on  
                            else: 
                                state_lighting = previousstate_lighting
                        else: 
                            state_lighting = previousstate_lighting
            
                    #Occupants present in the room
                    else:         
                    
                        # Check for critical moments
                        if timestep-1 >= 0:
                            # The occupants have interacted with the solar shading installation
                            probability_timestep = 0
                            if shading_interaction == 1: 
                                k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting)
                                illuminance_timestep = illuminance
                                probability_timestep = horizontal_dilation((self.probability_off_solar[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation_lighting
                            # The first occupant has entered the room 
                            if self.lighting_requirements[zone_name][timestep-1] == 0:
                                # Rooms without sufficient daylight entrance 
                                if zone_name in self.rooms_without_daylighting:
                                    illuminance_timestep = 0
                                    probability_timestep = max(probability_timestep, horizontal_dilation((self.probability_off_entering[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep))
                                # Rooms with sufficient daylight entrance and presence pattern defined in Occupancy
                                else: 
                                    k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting)
                                    illuminance_timestep = illuminance
                                    probability_timestep = max(probability_timestep, horizontal_dilation((self.probability_off_entering[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation_lighting)
                            # An additional occupant has entered the room
                            elif zone_name in self.occupancy_dataframe.columns and self.occupancy[zone_name][timestep] > self.occupancy[zone_name][timestep-1]: 
                                if zone_name in self.rooms_without_daylighting: 
                                    illuminance_timestep = 0
                                    probability_timestep = max(probability_timestep,horizontal_dilation((self.probability_off_entering_occupation[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep))
                                else:
                                    k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting)
                                    illuminance_timestep = illuminance
                                    probability_timestep = max(probability_timestep,horizontal_dilation((self.probability_off_entering_occupation[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation_lighting)
                            # An occupant has left the room
                            elif zone_name in self.occupancy_dataframe.columns and self.occupancy[zone_name][timestep] < self.occupancy[zone_name][timestep-1]: 
                                if zone_name in self.rooms_without_daylighting: 
                                    illuminance_timestep = 0
                                    probability_timestep = max(probability_timestep,horizontal_dilation((self.probability_off_leaving_occupation[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep))
                                else:
                                    k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting)
                                    illuminance_timestep = illuminance
                                    probability_timestep = max(probability_timestep,horizontal_dilation((self.probability_off_leaving_occupation[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation_lighting)
                            # No external critical moment detected 
                            else: 
                                # Rooms without sufficient daylight entrance
                                if zone_name in self.rooms_without_daylighting: 
                                    illuminance_timestep = 0
                                    probability_timestep = max(probability_timestep,horizontal_dilation((self.probability_off_during[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep))
                                # Rooms, except bedrooms, with sufficient daylight entrance and presence pattern defined in Occupancy
                                elif (zone_name in self.rooms_daylighting or zone_name in self.rooms_daylighting_2) and zone_name not in self.bedrooms_list: 
                                    k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting)
                                    illuminance_timestep = illuminance
                                    probability_timestep = max(probability_timestep,horizontal_dilation((self.probability_off_during[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation_lighting)
                                # Bedrooms with sufficient daylight entrance and presence pattern defined in Occupancy
                                elif zone_name in self.bedrooms_list: 
                                    illuminance_timestep = illuminance 
                                    # Person has entered the previous timestep and went immediately to sleep
                                    if timestep-2 >=0  and self.occupancy[zone_name][timestep - 2] == 0 and self.asleep_bedroom[zone_name][timestep-1] == self.occupancy[zone_name][timestep - 1] and self.asleep_bedroom[zone_name][timestep] == self.occupancy[zone_name][timestep]: 
                                        probability_timestep = max(probability_timestep, horizontal_dilation((self.probability_off_sleeping[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep))                          
                                    # Person awake
                                    elif self.asleep_bedroom[zone_name][timestep] != self.occupancy[zone_name][timestep]: 
                                        k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting)
                                        probability_timestep = max(probability_timestep, horizontal_dilation((self.probability_off_during[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) +k*self.standard_deviation_lighting)
                                    # Person is going to sleep 
                                    elif self.asleep_bedroom[zone_name][timestep] > self.asleep_bedroom[zone_name][timestep-1]: 
                                        probability_timestep = max(probability_timestep, horizontal_dilation((self.probability_off_sleeping[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep))
                                    # All occupants in the bedroom are asleep, making interactions impossible
                                    else: 
                                        probability_timestep = 0
                            state_lighting = stochastic_switching(probability_timestep, previousstate_lighting)
                        else: 
                            state_lighting = previousstate_lighting     
                        
                        
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
                                k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting)
                                illuminance_timestep = illuminance
                                probability_timestep = horizontal_dilation((self.probability_on_leaving[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation_lighting
                            state_lighting = stochastic_switching(probability_timestep, previousstate_lighting)
                        # Nobody present in the room, the lighting remains off
                        else: 
                            state_lighting = previousstate_lighting
                        
                    # Occupants present in the room
                    else: 
                        
                        # Check for critical moments
                        if timestep - 1 >= 0: 
                            # The occupants have interacted with the solar shading installation
                            probability_timestep = 0
                            if shading_interaction == 1: 
                                illuminance_timestep = illuminance
                                probability_timestep = horizontal_dilation((self.probability_on_solar[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep)      
                            # First occupant has entered the room
                            if self.lighting_requirements[zone_name][timestep-1] == 0: 
                                # Rooms without sufficient daylight entrance
                                if zone_name in self.rooms_without_daylighting: 
                                    illuminance_timestep = 0
                                    probability_timestep = max(probability_timestep, horizontal_dilation((self.probability_on_entering[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep))
                                # Rooms, except bedrooms, with sufficient daylight entrance and presence pattern defined in Occupancy
                                elif (zone_name in self.rooms_daylighting or zone_name in self.rooms_daylighting_2) and zone_name not in self.bedrooms_list: 
                                    k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting)
                                    illuminance_timestep = illuminance
                                    probability_timestep = max(probability_timestep, horizontal_dilation((self.probability_on_entering[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation_lighting)
                                # Bedrooms with sufficient daylight entrance and presence pattern defined in Occupancy
                                elif zone_name in self.bedrooms_list:
                                    illuminance_timestep = illuminance 
                                    # Person awake
                                    if self.asleep_bedroom[zone_name][timestep] != self.occupancy[zone_name][timestep]:
                                        k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting)
                                        probability_timestep = max(probability_timestep, horizontal_dilation((self.probability_on_entering[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation_lighting)
                                    # All occupants in the bedroom are asleep, making interactions impossible
                                    else: 
                                        probability_timestep = 0
                            # An additional occupant has entered the room
                            elif zone_name not in ['ToiletGround', 'ToiletFirst', 'Toilet', 'Corridor'] and self.occupancy[zone_name][timestep] > self.occupancy[zone_name][timestep-1]: 
                                if zone_name in self.rooms_without_daylighting: 
                                    illuminance_timestep = 0
                                    probability_timestep = max(probability_timestep, horizontal_dilation((self.probability_on_entering_occupation[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep))
                                else: 
                                    k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting)
                                    illuminance_timestep = illuminance
                                    probability_timestep = max(probability_timestep, horizontal_dilation((self.probability_on_entering_occupation[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation_lighting)
                            # An occupant has left the room
                            elif zone_name not in ['ToiletGround', 'ToiletFirst', 'Toilet', 'Corridor'] and self.occupancy[zone_name][timestep] < self.occupancy[zone_name][timestep-1]: 
                                if zone_name in self.rooms_without_daylighting: 
                                    illuminance_timestep = 0
                                    probability_timestep = max(probability_timestep, horizontal_dilation((self.probability_on_leaving_occupation[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep))
                                else:
                                    k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting)
                                    illuminance_timestep = illuminance
                                    probability_timestep = max(probability_timestep, horizontal_dilation((self.probability_on_leaving_occupation[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation_lighting)
                            # No external critical moment detected
                            else: 
                                # Rooms without sufficient daylight entrance
                                if zone_name in self.rooms_without_daylighting: 
                                    illuminance_timestep = 0
                                    probability_timestep = max(probability_timestep,horizontal_dilation((self.probability_on_during[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep))
                                # Rooms, except bedrooms, with sufficient daylight entrance and presence pattern defined in Occupancy
                                elif (zone_name in self.rooms_daylighting or zone_name in self.rooms_daylighting_2) and zone_name not in self.bedrooms_list: 
                                    k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting)
                                    illuminance_timestep = illuminance
                                    probability_timestep = max(probability_timestep,horizontal_dilation((self.probability_on_during[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation_lighting)
                                # Bedrooms with sufficient daylight entrance and presence pattern defined in Occupancy
                                elif zone_name in self.bedrooms_list:  
                                    illuminance_timestep = illuminance 
                                    # Person awake
                                    if self.asleep_bedroom[zone_name][timestep] != self.occupancy[zone_name][timestep]: 
                                        k = correcting_lighting(zone_name, timestep, hour, day, self.sunrise, self.sunset, self.occupancy, self.lighting_requirements, self.asleep_bedroom, self.minimum_illuminance, previousstate_lighting)
                                        probability_timestep = max(probability_timestep,horizontal_dilation((self.probability_on_during[zone_name]), self.minimum_illuminance[zone_name][timestep])(illuminance_timestep) + k*self.standard_deviation_lighting)
                                    # All occupants in the bedroom are asleep, making interactions impossible
                                    else: 
                                        probability_timestep = 0
                            state_lighting = stochastic_switching(probability_timestep, previousstate_lighting)
                        else: 
                            state_lighting = previousstate_lighting            
                    
            
            return state_lighting
        
        def simulate_manual_shading(self, zone_name, hour, day, timestep, sun_up, indoor_operative_temperature, glare, action_practical_darkness, action_practical_privacy, action_practical_security, timestep_previous_shading): 
            def stochastic_interaction(probability_situation: float) -> int:
                ''' 
                Simulates a stochastic state switch based on a given probability.
                '''
                if not (0 <= probability_situation <= 1): 
                    if probability_situation < 0: 
                        probability_situation = 0
                    elif probability_situation > 1: 
                        probability_situation = 1
                chance = np.random.random()
                if chance <= probability_situation: # switching the shading
                    return 1
                else: 
                    return 0
             
            def correcting_shading(zone, timestep, timestep_previous_shading, occupancy, asleep) -> float: 
                # Correcting factor subsequent interactions
                # Assumption: The chance of interacting is lowered within the first 4 hours after another interaction.
                if timestep - timestep_previous_shading <= 120: 
                    # Assumption: the probability is interpolated between -1 and 0 according to the time since the previous interaction with a maximum of 20 minutes.
                    k_int = (timestep - timestep_previous_shading)/120 - 1
                else: 
                    k_int = 0
                    
                # Correcting factor during - occupancy
                if occupancy[zone][timestep -1] > 0 and occupancy[zone][timestep + 1] > 0:
                    # Assumption: The probability of interacting increases with the period of expected occupation
                    i = 1
                    while occupancy[zone][timestep + i] >= 1 and timestep + i <= 262800 and i <= 10: 
                        i += 1
                    # Assumption: the probability is interpolated between -1 and 0 according to the time until leaving with a maximum of 20 minutes.  
                    k_occpros = i/10 - 1
                    # Assumption: The occupant is less likely to interact right after entering the room, followed by a fast increase in probability and a gradual decrease as the eyes adapt to the environment. 
                    i = 1
                    while occupancy[zone][timestep - i] >= 1 and timestep - i > 0 and i <= 10: 
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
                
                k =  0.5*k_int+0.5*k_occ
                
                return k
                    
            def correcting_anticipate(max_temp, temp_difference) -> float: 
                # Correcting factor maximum temperatuur 3 days
                if 25 <= max_temp < 30:
                    k_max = (max_temp-25)/5
                elif max_temp >= 30: 
                    k_max = 1
                else: 
                    k_max = 0
                    
                # Correcting factor temperature increase
                if temp_difference < 1: 
                    k_diff = temp_difference/10
                elif 1 <= temp_difference < 3:
                    k_diff = (temp_difference-1)/5 + 1/10
                elif 3 <= temp_difference < 5: 
                    k_diff = 1/2 + (temp_difference-3)/2
                else: 
                    k_diff = 1
                
                k = 0.5*k_max+0.5*k_diff
                return k
                
            def correcting_irradiance(irradiance) -> float: 
                # Correcting factor vertical irradiance on window
                if irradiance < 50: 
                    k_irr = -4 # Drastically reduce the probability of closing interactions under cloudy sky or twilight
                elif irradiance < 300: 
                    k_irr = ((irradiance-50)/250)*3 - 4 
                elif 300 <= irradiance < 600: 
                    k_irr = irradiance/300 - 1
                else: 
                    k_irr = 1
                
                return k_irr
                    
            def generate_linear_function(parameters): 
                '''
                Generate the linear probability function.
                '''
                def f(x):
                    a, b = parameters['a'], parameters['b']
                    return a*x + b
                return f
                
            def horizontal_shift(parameters, shift_distance): 
                '''
                Shifts the probability function in relation to the level of acceptable glare/indoor temperature during the timestep.
                
                '''
                def f(x):
                    a, b, c = parameters['a'], parameters['b'], parameters['c']
                    return a + c * np.exp(b * (x - shift_distance))
                return f
                
            def neutral_indoor_temperature(zone, timestep, offices_list, bedrooms_list, temperature_ref): 
                if zone in ['Living', 'Kitchen', 'LivingKitchen'] or zone in offices_list: 
                    if temperature_ref < 12.5: 
                        temperature_n = 20.4 + 0.06*temperature_ref
                    else: 
                        temperature_n = 16.63 + 0.36*temperature_ref
                elif zone in ['Bathroom']: 
                    if temperature_ref < 11: 
                        temperature_n = 0.112*temperature_ref + 22.65
                    else:
                        temperature_n = 0.306*temperature_ref + 20.32
                elif zone in bedrooms_list: 
                    if temperature_ref < 0: 
                        temperature_n = 16
                    elif 0 <= temperature_ref < 12.6: 
                        temperature_n = 0.23*temperature_ref + 16
                    elif 12.6 <= temperature_ref < 21.8: 
                        temperature_n = 0.77*temperature_ref + 9.18
                    else: 
                        temperature_n = 26
                else: 
                    temperature_n = 26
                return temperature_n
            
            if 'Bathroom' in zone_name: 
                room_habits = 'bathroom'
            elif 'Bedroom' in zone_name and zone_name not in self.offices_list: 
                room_habits = 'bedroom'
            elif 'Corridor' in zone_name or 'Hallway' in zone_name: 
                room_habits = 'other'
            elif 'Kitchen' in zone_name or 'Living' in zone_name: 
                room_habits = 'living'
            elif zone_name in self.offices_list: 
                room_habits = 'living'
            elif 'Toilet' in zone_name or 'Storage' in zone_name: 
                room_habits = 'other'            
			
            # EVALUATION PRACTICAL REASONS (E.G. DARKNESS, PRIVACY, SECURITY)
            # Assumption: practical-driven shading interactions are executed for all window orientations simultaneously. 
            # Assumption: occlusion rate is expected to be constant for practical habits. 
            occlusion_timestep_practical = 0
            interaction_practical = 0
            occlusion = {}
            # Darkness
            if self.drivers_sec[room_habits][0] == 1: 
                if timestep-1 >= 0: 
                    if action_practical_darkness == 0: 
                        # Closing the solar shading
                        if self.darkness_habits == 'dark': 
                        # Assumption: shading is closed when it is dark and occupant in the room. 
                            if sun_up == False and self.occupancy[zone_name][timestep] > 0 and zone_name not in self.bedrooms_list: 
                                if stochastic_interaction(self.probability_darkness) == 1: 
                                    occlusion_timestep_practical = self.occlusion_darkness
                                    interaction_practical = 1
                                    action_practical_darkness = 1
                            elif sun_up == False and zone_name in self.bedrooms_list and self.occupancy[zone_name][timestep] > self.asleep_bedroom[zone_name][timestep]: 
                                if stochastic_interaction(self.probability_darkness) == 1: 
                                    occlusion_timestep_practical = self.occlusion_darkness
                                    interaction_practical = 1
                                    action_practical_darkness = 1
                        elif self.darkness_habits == 'anticipate dark': 
                        # Assumption: shading is closed when it is dark or will become dark during absence/asleep    
                            if sun_up == False and self.occupancy[zone_name][timestep] > 0 and zone_name not in self.bedrooms_list: 
                                if stochastic_interaction(self.probability_darkness) == 1: 
                                    occlusion_timestep_practical = self.occlusion_darkness
                                    interaction_practical = 1
                                    action_practical_darkness = 1
                            elif sun_up == False and zone_name in self.bedrooms_list and self.occupancy[zone_name][timestep] > self.asleep_bedroom[zone_name][timestep]:  
                                if stochastic_interaction(self.probability_darkness) == 1: 
                                    occlusion_timestep_practical = self.occlusion_darkness
                                    interaction_practical = 1
                                    action_practical_darkness = 1
                            # Person leaves the building/is going to sleep before it is getting dark
                            elif self.occupancy['Building'][timestep] == 0 and self.occupancy['Building'][timestep-1] > 0:
                                # Check whether they return before darkness
                                i = timestep + 1
                                while self.occupancy['Building'][i] == 0 and (i - (day - 1)*24*30)/30 < self.sunset[day]: 
                                    i += 1
                                    if (i - (day - 1)*24*30)/30 >= self.sunset[day]: 
                                        if stochastic_interaction(self.probability_darkness) == 1: 
                                            occlusion_timestep_practical = self.occlusion_darkness
                                            interaction_practical = 1
                                            action_practical_darkness = 1
                            elif self.occupancy['Building'][timestep] == self.asleep[timestep] and self.occupancy['Building'][timestep-1] > self.asleep[timestep-1]:
                                # Check whether someone wakes up before darkness
                                i = timestep + 1
                                while self.occupancy['Building'][i] == self.asleep[i] and (i - (day - 1)*24*30)/30 < self.sunset[day]: 
                                    i += 1
                                    if (i - (day - 1)*24*30)/30 >= self.sunset[day]: 
                                        if stochastic_interaction(self.probability_darkness) == 1: 
                                            occlusion_timestep_practical = self.occlusion_darkness
                                            interaction_practical = 1
                                            action_practical_darkness = 1
                        elif self.darkness_habits == 'evening and morning routine': 
                        # Assumption: shading is closed when last person leaves the room in the evening/first person is going to sleep in the bedrooms
                            if zone_name not in self.bedrooms_list and self.occupancy[zone_name][timestep-1] > 0 and self.occupancy[zone_name][timestep] == 0: 
                                # Check whether someone returns before going to sleep
                                i = timestep + 1
                                j = 0
                                while self.occupancy[zone_name][i] == 0 and i < len(self.asleep) and interaction_practical == 0: 
                                    i += 1
                                    if self.occupancy['Building'][i] == self.asleep[i] and self.asleep[i] > 0: 
                                        j += 1
                                        if j > 5: 
                                            if stochastic_interaction(self.probability_darkness) == 1: 
                                                occlusion_timestep_practical = self.occlusion_darkness
                                                interaction_practical = 1
                                                action_practical_darkness = 1
                            elif zone_name in self.bedrooms_list and self.occupancy[zone_name][timestep] > 0 and self.occupancy[zone_name][timestep-1] == 0: 
                                # Check whether they are leaving the room before going asleep
                                i = timestep + 1
                                while self.occupancy[zone_name][i] > 0 and self.asleep_bedroom[zone_name][i] != 0 and i > 0: 
                                    i -= 1
                                    if self.asleep_bedroom[zone_name][i] == 0: 
                                        if stochastic_interaction(self.probability_darkness) == 1: 
                                            occlusion_timestep_practical = self.occlusion_darkness
                                            interaction_practical = 1
                                            action_practical_darkness = 1
                    elif action_practical_darkness == 1: 
                    # Opening the solar shading
                        if self.darkness_habits == 'dark':
                        # Assumption: shading is opened when the sun is up and at least one inhabitant in the room.     
                            if sun_up == True and self.occupancy[zone_name][timestep] > 0 and zone_name not in self.bedrooms_list:
                                # Check whether they return to bed within 20 minutes
                                return_to_bed = 0
                                i = timestep
                                while i <= timestep + 10 and self.asleep[i] != self.occupancy['Building'][i]: 
                                    i+= 1
                                    if self.asleep[i] == self.occupancy['Building'][i]: 
                                        return_to_bed = 1
                                if return_to_bed == 0: 
                                    if stochastic_interaction(self.probability_darkness) == 0:
                                        occlusion_timestep_practical = self.occlusion_darkness
                                    else:
                                        interaction_practical = 1
                                        action_practical_darkness = 0
                            elif sun_up == True and zone_name in self.bedrooms_list and self.occupancy[zone_name][timestep] > self.asleep_bedroom[zone_name][timestep] and self.asleep_bedroom[zone_name][timestep] == 0:
                                # Check whether someone return to bed within 20 minutes
                                return_to_bed = 0
                                i = timestep
                                while i <= timestep + 10 and self.asleep_bedroom[zone_name][i] == 0: 
                                    i+= 1
                                    if self.asleep_bedroom[zone_name][i] > 0: 
                                        return_to_bed = 1
                                if return_to_bed == 0: 
                                    if stochastic_interaction(self.probability_darkness) == 0:
                                        occlusion_timestep_practical = self.occlusion_darkness
                                    else:
                                        interaction_practical = 1
                                        action_practical_darkness = 0
                        elif self.darkness_habits == 'anticipate dark':
                        # Assumption: shading is reopened when the sun is up and at least one inhabitant is the room or when they leave and not return before sunrise. 
                            if sun_up == True and self.occupancy[zone_name][timestep] > 0 and self.occupancy[zone_name][timestep-1]==0 and zone_name not in self.bedrooms_list:
                                # Check whether they return to bed within 20 minutes
                                return_to_bed = 0
                                i = timestep
                                while i <= timestep + 10 and self.asleep[i] != self.occupancy['Building'][i]: 
                                    i+= 1
                                    if self.asleep[i] == self.occupancy['Building'][i]: 
                                        return_to_bed = 1
                                if return_to_bed == 0: 
                                    if stochastic_interaction(self.probability_darkness) == 0:
                                        occlusion_timestep_practical = self.occlusion_darkness
                                    else:
                                        interaction_practical = 1
                                        action_practical_darkness = 0
                            elif sun_up == True and zone_name in self.bedrooms_list and self.occupancy[zone_name][timestep] > 0 and self.asleep_bedroom[zone_name][timestep] == 0:
                                # Check whether someone return to bed within 20 minutes
                                return_to_bed = 0
                                i = timestep
                                while i <= timestep + 10 and self.asleep_bedroom[zone_name][i] == 0: 
                                    i+= 1
                                    if self.asleep_bedroom[zone_name][i] > 0: 
                                        return_to_bed = 1
                                if return_to_bed == 0: 
                                    if stochastic_interaction(self.probability_darkness) == 0:
                                        occlusion_timestep_practical = self.occlusion_darkness
                                    else:
                                        interaction_practical = 1
                                        action_practical_darkness = 0
                            elif self.occupancy['Building'][timestep] == 0 and self.occupancy['Building'][timestep-1] > 0:
                                # Check whether they return before sunrise
                                i = timestep + 1
                                while self.occupancy['Building'][i] == 0 and (i - (day - 1)*24*30)/30 < self.sunrise[day]: 
                                    i += 1
                                    if (i - (day - 1)*24*30)/30 >= self.sunrise[day]: 
                                        if stochastic_interaction(self.probability_darkness) == 0: 
                                            occlusion_timestep_practical = self.occlusion_darkness
                                        else:
                                            interaction_practical = 1
                                            action_practical_darkness = 0
                        elif self.darkness_habits == 'evening and morning routine': 
                        # Assumption: the solar shading is opened as part of the wakeing up routine: when the first person enters the room in the morning and the last person wakes up in the bedrooms. 
                            if zone_name not in self.bedrooms_list and self.occupancy[zone_name][timestep-1] > 0 and self.occupancy[zone_name][timestep] == 0: 
                                # Check whether they return to bed within 20 minutes
                                return_to_bed = 0
                                i = timestep
                                while i <= timestep + 10 and self.asleep[i] != self.occupancy['Building'][i]: 
                                    i+= 1
                                    if self.asleep[i] == self.occupancy['Building'][i]: 
                                        return_to_bed = 1
                                if return_to_bed == 0: 
                                    if stochastic_interaction(self.probability_darkness) == 0:
                                        occlusion_timestep_practical = self.occlusion_darkness
                                    else:
                                        interaction_practical = 1
                                        action_practical_darkness = 0
                            elif zone_name in self.bedrooms_list and self.asleep_bedroom[zone_name][timestep] == 0 and self.occupancy[zone_name][timestep-1] > 0 and self.occupancy[zone_name][timestep] == 0:     
                                # Check whether someone return to bed within 20 minutes
                                return_to_bed = 0
                                i = timestep
                                while i <= timestep + 10 and self.asleep_bedroom[zone_name][i] == 0: 
                                    i+= 1
                                    if self.asleep_bedroom[zone_name][i] > 0: 
                                        return_to_bed = 1
                                if return_to_bed == 0: 
                                    if stochastic_interaction(self.probability_darkness) == 0:
                                        occlusion_timestep_practical = self.occlusion_darkness
                                    else:
                                        interaction_practical = 1
                                        action_practical_darkness = 0
            # Privacy
            if self.drivers_sec[room_habits][2] == 1: 
                if timestep-1 >= 0: 
                    if action_practical_privacy == 0: 
                        # Closing the solar shading
                        if self.privacy_habits == 'no interaction': 
                            # Assumption: the shading is permanently closed. 
                            if self.occlusion_privacy > occlusion_timestep_practical: 
                                occlusion_timestep_practical = self.occlusion_privacy
                            interaction_practical = 1
                            action_practical_privacy = 1
                        elif self.privacy_habits == 'operating by presence': 
                        # Assumption: solar shading is closed when entering
                            if self.occupancy[zone_name][timestep] > 0 and self.occupancy[zone_name][timestep-1] == 0: 
                                if stochastic_interaction(self.probability_privacy) == 1: 
                                    if self.occlusion_privacy > occlusion_timestep_practical:
                                        occlusion_timestep_practical = self.occlusion_privacy
                                    interaction_practical = 1
                                    action_practical_privacy = 1
                    elif action_practical_privacy == 1: 
                        # Opening the solar shading
                        if self.privacy_habits == 'no interaction': 
                        # Assumption: the shading is permanently closed. 
                            pass
                        elif self.privacy_habits == 'operating by presence': 
                        # Assumption: the shading is reopened when the last person leaves the room and is absent for more than 20 minutes. 
                            if self.occupancy[zone_name][timestep-1] > 0 and self.occupancy[zone_name][timestep] == 0: 
                                i = timestep + 1
                                # absent for more than 20 minutes
                                while self.occupancy[zone_name][i] == 0 and i <= timestep + 10: 
                                    i += 1
                                    if i == timestep + 10:
                                        if stochastic_interaction(self.probability_privacy) == 0 and occlusion_timestep_practical < self.occlusion_privacy: 
                                            occlusion_timestep_practical = self.occlusion_privacy
                                        else: 
                                            interaction_practical = 1
                                            action_practical_privacy = 0
            # Security
            if self.drivers_sec[room_habits][3] == 1:  
                if timestep-1 >= 0: 
                    if action_practical_security == 0: 
                        # Closing the solar shading
                        if self.security_shading_habits == 'absence/asleep': 
                        # Assumption: shading will be closed during periods of absence and when all occupants are in their bedrooms. 
                            occupancy_bedrooms = 0
                            occupancy_bedrooms_previous = 0
                            for bedroom in self.bedrooms_list:
                                occupancy_bedrooms += self.occupancy[bedroom][timestep]
                                occupancy_bedrooms_previous += self.occupancy[bedroom][timestep-1]
                            if self.occupancy['Building'][timestep] == 0 and self.occupancy['Building'][timestep-1] > 0: 
                                i = timestep + 1
                                # absent for more than 20 minutes
                                while self.occupancy['Building'][i] == 0 and i <= timestep + 10: 
                                    i += 1
                                    if i == timestep + 10:
                                        if stochastic_interaction(self.probability_security_shading) == 1:
                                            if self.occlusion_security > occlusion_timestep_practical:
                                                occlusion_timestep_practical = self.occlusion_security
                                            interaction_practical = 1
                                            action_practical_security = 1
                            elif self.occupancy['Building'][timestep] == occupancy_bedrooms and self.occupancy['Building'][timestep-1] > occupancy_bedrooms_previous: 
                                if stochastic_interaction(self.probability_security_shading) == 1:
                                    if self.occlusion_security > occlusion_timestep_practical:
                                        occlusion_timestep_practical = self.occlusion_security
                                    interaction_practical = 1
                                    action_practical_security = 1
                        elif self.security_shading_habits == 'dark and absence/asleep':
                            # Assumption: shading will be closed on during periods of absence and when all occupants are in their bedrooms as it is dark. 
                            occupancy_bedrooms = 0
                            occupancy_bedrooms_previous = 0
                            for bedroom in self.bedrooms_list:
                                occupancy_bedrooms += self.occupancy[bedroom][timestep]
                                occupancy_bedrooms_previous += self.occupancy[bedroom][timestep-1]
                            if sun_up == False and self.occupancy['Building'][timestep] == 0 and self.occupancy['Building'][timestep-1] > 0: 
                                # absent for more than 20 minutes
                                while self.occupancy['Building'][i] == 0 and i <= timestep + 10: 
                                    i += 1
                                    if i == timestep + 10:
                                        if stochastic_interaction(self.probability_security_shading) == 1:
                                            if self.occlusion_security > occlusion_timestep_practical:
                                                occlusion_timestep_practical = self.occlusion_security
                                            interaction_practical = 1
                                            action_practical_security = 1
                            elif sun_up == False and self.occupancy['Building'][timestep] == occupancy_bedrooms and self.occupancy['Building'][timestep-1] > occupancy_bedrooms_previous: 
                                if stochastic_interaction(self.probability_security_shading) == 1:
                                    if self.occlusion_security > occlusion_timestep_practical:
                                        occlusion_timestep_practical = self.occlusion_security
                                    interaction_practical = 1
                                    action_practical_security = 1
                        elif self.security_shading_habits == 'anticipate dark and absence/asleep': 
                        # Assumption: shading will be closed on during periods of absence and when all occupants are in their bedrooms. The occupants also anticipate when leaving the house while the sun is still up.
                            occupancy_bedrooms = 0
                            occupancy_bedrooms_previous = 0
                            for bedroom in self.bedrooms_list:
                                occupancy_bedrooms += self.occupancy[bedroom][timestep]
                                occupancy_bedrooms_previous += self.occupancy[bedroom][timestep-1]                                    
                            # Last person leaves the building
                            if self.occupancy['Building'][timestep] == 0 and self.occupancy['Building'][timestep-1] > 0: 
                                i = timestep + 1
                                # absent for more than 20 minutes
                                while self.occupancy['Building'][i] == 0 and i <= timestep + 10: 
                                    i += 1
                                    if i == timestep + 10: 
                                        # Dark when leaving
                                        if sun_up == False: 
                                            if stochastic_interaction(self.probability_security_shading) == 1:
                                                if self.occlusion_security > occlusion_timestep_practical:
                                                    occlusion_timestep_practical = self.occlusion_security
                                                interaction_practical = 1
                                                action_practical_security = 1
                                        else: 
                                            # Check whether they return before darkness
                                            j = timestep + 1
                                            while self.occupancy['Building'][j] == 0 and (j - (day - 1)*24*30)/30 < self.sunset[day]: 
                                                j += 1 
                                                if (j - (day - 1)*24*30)/30 >= self.sunset[day]: 
                                                    if stochastic_interaction(self.probability_security_shading) == 1:
                                                        if self.occlusion_security > occlusion_timestep_practical:
                                                            occlusion_timestep_practical = self.occlusion_security
                                                        interaction_practical = 1
                                                        action_practical_security = 1
                            # Last occupant going to sleep
                            elif self.occupancy['Building'][timestep] == occupancy_bedrooms and self.occupancy['Building'][timestep-1] > occupancy_bedrooms_previous: 
                                if sun_up == False: 
                                    if stochastic_interaction(self.probability_security_shading) == 1:
                                        if self.occlusion_security > occlusion_timestep_practical:
                                            occlusion_timestep_practical = self.occlusion_security
                                        interaction_practical = 1
                                        action_practical_security = 1
                                else: 
                                    i = timestep + 1
                                    while self.occupancy['Building'][i] == occupancy_bedrooms and (i - (day - 1)*24*30)/30 < self.sunset[day]: 
                                        i += 1
                                        if (i - (day - 1)*24*30)/30 >= self.sunset[day]: 
                                            if stochastic_interaction(self.probability_security_shading) == 1:
                                                if self.occlusion_security > occlusion_timestep_practical:
                                                    occlusion_timestep_practical = self.occlusion_security
                                                interaction_practical = 1
                                                action_practical_security = 1
                    elif action_practical_security == 1:
                        # Opening the solar shading
                        if self.security_shading_habits == 'absence/asleep': 
                        # Assumption: shading is opened when first occupant coming home or wakeing up.     
                            if self.occupancy['Building'][timestep] > 0 and self.occupancy['Building'][timestep - 1] == 0:
                                if stochastic_interaction(self.probability_security_shading) == 0 and occlusion_timestep_practical < self.occlusion_security: 
                                    occlusion_timestep_practical == self.occlusion_security
                                else: 
                                    interaction_practical = 1
                                    action_practical_security = 0
                            elif zone_name in self.bedrooms_list and self.occupancy[zone_name][timestep] > self.asleep_bedroom[zone_name][timestep] and self.asleep_bedroom[zone_name][timestep] == 0:
                                # Check whether someone return to bed within 20 minutes
                                return_to_bed = 0
                                i = timestep
                                while i <= timestep + 10 and self.asleep_bedroom[zone_name][i] == 0: 
                                    i+= 1
                                    if self.asleep_bedroom[zone_name][i] > 0: 
                                        return_to_bed = 1
                                if return_to_bed == 0: 
                                    if stochastic_interaction(self.probability_security_shading) == 0 and occlusion_timestep_practical < self.occlusion_security: 
                                        occlusion_timestep_practical == self.occlusion_security
                                    else: 
                                        interaction_practical = 1
                                        action_practical_security = 0
                            elif self.occupancy[zone_name][timestep] > 0 and zone_name not in self.bedrooms_list: 
                                # Check whether they return to bed within 20 minutes/leave the building while others sleeping
                                return_to_bed = 0
                                i = timestep
                                while i <= timestep + 10 and self.asleep[i] != self.occupancy['Building'][i]: 
                                    i+= 1
                                    if self.asleep[i] == self.occupancy['Building'][i]: 
                                        return_to_bed = 1
                                if return_to_bed == 0: 
                                    if stochastic_interaction(self.probability_security_shading) == 0:
                                        occlusion_timestep_practical = self.occlusion_security
                                    else:
                                        interaction_practical = 1
                                        action_practical_security = 0
                        elif self.security_shading_habits == 'dark and absence/asleep':
                        # Assumption: shading is reopened when first occupant coming home or wakeing up when the sun is up. 
                            if sun_up == True and self.occupancy['Building'][timestep] > self.asleep[timestep] and zone_name not in self.bedrooms_list:
                                if stochastic_interaction(self.probability_security_shading) == 0 and occlusion_timestep_practical < self.occlusion_security: 
                                    occlusion_timestep_practical == self.occlusion_security
                                else: 
                                    interaction_practical = 1
                                    action_practical_security = 0
                            elif sun_up == True and zone_name in self.bedrooms_list and self.occupancy['Building'][timestep] > 0 and self.asleep_bedroom[zone_name][timestep] == 0:
                                # Check whether someone return to bed within 20 minutes
                                return_to_bed = 0
                                i = timestep
                                while i <= timestep + 10 and self.asleep_bedroom[zone_name][i] == 0: 
                                    i+= 1
                                    if self.asleep_bedroom[zone_name][i] > 0: 
                                       return_to_bed = 1
                                if return_to_bed == 0: 
                                    if stochastic_interaction(self.probability_security_shading) == 0 and occlusion_timestep_practical < self.occlusion_security: 
                                        occlusion_timestep_practical == self.occlusion_security
                                    else: 
                                        interaction_practical = 1
                                        action_practical_security = 0
                        elif self.security_shading_habits == 'anticipate dark and absence/asleep': 
                        # Assumption: shading is reopened when first occupant coming home or wakeing up when the sun is up. 
                            if sun_up == True and self.occupancy['Building'][timestep] > self.asleep[timestep] and zone_name not in self.bedrooms_list:
                                if stochastic_interaction(self.probability_security_shading) == 0 and occlusion_timestep_practical < self.occlusion_security: 
                                    occlusion_timestep_practical == self.occlusion_security
                                else: 
                                    interaction_practical = 1
                                    action_practical_security = 0
                            elif sun_up == True and zone_name in self.bedrooms_list and self.occupancy['Building'][timestep] > 0 and self.asleep_bedroom[zone_name][timestep] == 0:
                                # Check whether someone return to bed within 20 minutes
                                return_to_bed = 0
                                i = timestep
                                while i <= timestep + 10 and self.asleep_bedroom[zone_name][i] == 0: 
                                    i+= 1
                                    if self.asleep_bedroom[zone_name][i] > 0: 
                                       return_to_bed = 1
                                if return_to_bed == 0: 
                                    if stochastic_interaction(self.probability_security_shading) == 0 and occlusion_timestep_practical < self.occlusion_security: 
                                        occlusion_timestep_practical == self.occlusion_security
                                    else: 
                                        interaction_practical = 1
                                        action_practical_security = 0
            # Shading opened from practical perspective
            if action_practical_darkness == 0 and action_practical_privacy == 0 and action_practical_security == 0:
                occlusion_timestep_practical = 0
            else: 
                # Set occlusion to matching occlusion rate in case no interaction takes place during this timestep
                if action_practical_darkness == 1: 
                    occlusion_timestep_practical = max(occlusion_timestep_practical, self.occlusion_darkness)
                if action_practical_privacy == 1: 
                    occlusion_timestep_practical = max(occlusion_timestep_practical, self.occlusion_privacy)
                if action_practical_security == 1: 
                    occlusion_timestep_practical = max(occlusion_timestep_practical, self.occlusion_security)

            # EVALUATION PSYCHOLOGICAL REASONS 
            
            # Evaluate per separate orientation
            for orientation in range(1,number_orientations+1):
                irradiance = self.api.exchange.get_variable_value(state, self.handle_irradiance[zone_name+'.'+str(orientation)])
                previousstate_shading = self.api.exchange.get_global_value(state, self.handle_previousstateshading[zone_name + '.' + str(orientation)])
                interaction_psychological = 0
                action_psychological = None
                driver = None
                
                # Evalation whether interaction takes place
                if previousstate_shading == 0: 
                    evaluation_actions = ['close']
                elif previousstate_shading == 1: 
                    evaluation_actions = ['open']
                else: 
                    evaluation_actions = ['open','close']
                    random.shuffle(evaluation_actions)
                
                # Nobody present in the room
                if self.occupancy[zone_name][timestep] == 0:  
                        
                    # Last occupant has left the room since last timestep
                    if timestep-1 >= 0 and self.occupancy[zone_name][timestep-1] > 0: 
                        for action in evaluation_actions: 
                            probability_timestep = 0
                            # Thermal comfort
                            if self.drivers_sec[room_habits][1] == 1:
                                if action == 'close': 
                                    k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                    k_irr = correcting_irradiance(irradiance)
                                    temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                    probability_timestep_thermal = horizontal_shift(self.probability_thermal_leaving[action][zone_name], temperature_n)(indoor_operative_temperature) + ((k + k_irr)*self.standard_deviation_shading)
                                # Opening is based on irradiance
                                elif action == 'open': 
                                    k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                    probability_timestep_thermal = generate_linear_function(self.probability_visual_leaving_irradiance[action][zone_name])(irradiance) + (2*k*self.standard_deviation_shading)
                                probability_timestep = max(probability_timestep, probability_timestep_thermal)
                                driver = 'thermal'
                            # Thermal comfort anticipate
                            if self.drivers_sec[room_habits][1] == 1 and self.thermal_anticipate == 2:
                                # Assumption: the probability of thermal comfort is adjusted to future outdoor temperatures and expected temperature increases.
                                # Only when maximum temperature is expected to increase to more than 25°C and there is an increase in temperature expected the next 3 days. 
                                if self.daily_max_temperatures['next_day_max_for_max_diff'][day-1] > 25 and self.daily_max_temperatures['max_diff_next_1_3_days'][day-1] > 0: 
                                    if action == 'close': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        k_temp = correcting_anticipate(self.daily_max_temperatures['next_day_max_for_max_diff'][day-1], self.daily_max_temperatures['next_day_max_for_max_diff'][day-1])
                                        k_irr = correcting_irradiance(irradiance)
                                        temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                        probability_timestep_thermalanticipate = horizontal_shift(self.probability_thermal_leaving[action][zone_name], temperature_n)(indoor_operative_temperature) + (2/3*(k + k_irr + k_temp)*self.standard_deviation_shading)
                                    elif action == 'open': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        probability_timestep_thermalanticipate = generate_linear_function(self.probability_visual_leaving_irradiance[action][zone_name])(irradiance) + (2*k*self.standard_deviation_shading)
                                    probability_timestep = max(probability_timestep, probability_timestep_thermalanticipate)
                                    if probability_timestep == probability_timestep_thermalanticipate: 
                                        driver = 'thermal_anticipate'
                            # Visual comfort 
                            if self.drivers_sec[room_habits][4] == 1:
                                # Irradiance
                                k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                probability_timestep_irradiance = generate_linear_function(self.probability_visual_leaving_irradiance[action][zone_name])(irradiance) + (2*k*self.standard_deviation_shading)
                                # Glare
                                # Assumption: glare is only evaluated to close the shading. 
                                if action == 'close': 
                                    k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                    k_irr = correcting_irradiance(irradiance)
                                    probability_timestep_glare = horizontal_shift(self.probability_visual_leaving_glare[action][zone_name], self.glare_evaluation[zone_name][timestep])(glare) + ((k+k_irr)*self.standard_deviation_shading)
                                else: 
                                    probability_timestep_glare = 0
                                probability_timestep_visual = max(probability_timestep_irradiance, probability_timestep_glare)
                                probability_timestep = max(probability_timestep, probability_timestep_visual)
                                if probability_timestep == probability_timestep_visual: 
                                    if probability_timestep == probability_timestep_irradiance: 
                                        driver = 'irradiance'
                                    else: 
                                        driver = 'glare'
                            interaction_psychological = stochastic_interaction(probability_timestep)
                            if interaction_psychological == 1: 
                                action_psychological = action
                                break
                                
                    # No interaction
                    else:
                        interaction_psychological = 0
                            
                # Occupants are present and can interact
                else: 
                    
                    # Check for critical moments
                    if timestep - 1 >= 0: 
                            
                        # The first occupant has entered the room 
                        if self.occupancy[zone_name][timestep-1] == 0:
                            for action in evaluation_actions: 
                                probability_timestep = 0
                                # Thermal comfort
                                if self.drivers_sec[room_habits][1] == 1:
                                    if action == 'close': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        k_irr = correcting_irradiance(irradiance)
                                        temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                        probability_timestep_thermal = horizontal_shift(self.probability_thermal_entering[action][zone_name], temperature_n)(indoor_operative_temperature) + (k + k_irr)*self.standard_deviation_shading
                                    # Opening is based on irradiance
                                    elif action == 'open': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        probability_timestep_thermal = generate_linear_function(self.probability_visual_entering_irradiance[action][zone_name])(irradiance) +( 2*k*self.standard_deviation_shading)
                                    probability_timestep = max(probability_timestep, probability_timestep_thermal)
                                    driver = 'thermal'
                                # Thermal comfort anticipate
                                if self.drivers_sec[room_habits][1] == 1 and self.thermal_anticipate == 2:
                                    # Assumption: the probability of thermal comfort is adjusted to future outdoor temperatures and expected temperature increases.
                                    # Only when maximum temperature is expected to increase to more than 25°C and there is an increase in temperature expected the next 3 days. 
                                    if self.daily_max_temperatures['next_day_max_for_max_diff'][day-1] > 25 and self.daily_max_temperatures['max_diff_next_1_3_days'][day-1] > 0: 
                                        if action == 'close': 
                                            k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                            k_temp = correcting_anticipate(self.daily_max_temperatures['next_day_max_for_max_diff'][day-1], self.daily_max_temperatures['next_day_max_for_max_diff'][day-1])
                                            k_irr = correcting_irradiance(irradiance)
                                            temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                            probability_timestep_thermalanticipate = horizontal_shift(self.probability_thermal_entering[action][zone_name], temperature_n)(indoor_operative_temperature) +( 2/3*(k + k_irr + k_temp)*self.standard_deviation_shading)
                                        # Opening is based on irradiance
                                        elif action == 'open': 
                                            k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                            probability_timestep_thermalanticipate = generate_linear_function(self.probability_visual_entering_irradiance[action][zone_name])(irradiance) +( 2*k*self.standard_deviation_shading)
                                        probability_timestep = max(probability_timestep, probability_timestep_thermalanticipate)
                                        if probability_timestep == probability_timestep_thermalanticipate: 
                                            driver = 'thermal_anticipate'
                                # Visual comfort 
                                if self.drivers_sec[room_habits][4] == 1:
                                    # Irradiance
                                    k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                    probability_timestep_irradiance = generate_linear_function(self.probability_visual_entering_irradiance[action][zone_name])(irradiance) +( 2*k*self.standard_deviation_shading)
                                    # Glare
                                    # Assumption: glare is only evaluated to close the shading. 
                                    if action == 'close': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        k_irr = correcting_irradiance(irradiance)
                                        probability_timestep_glare = horizontal_shift(self.probability_visual_entering_glare[action][zone_name], self.glare_evaluation[zone_name][timestep])(glare) +( (k+k_irr)*self.standard_deviation_shading)
                                    else: 
                                        probability_timestep_glare = 0
                                    probability_timestep_visual = max(probability_timestep_irradiance, probability_timestep_glare)
                                    probability_timestep = max(probability_timestep, probability_timestep_visual)
                                    if probability_timestep == probability_timestep_visual: 
                                        if probability_timestep == probability_timestep_irradiance: 
                                            driver = 'irradiance'
                                        else: 
                                            driver = 'glare'
                                interaction_psychological = stochastic_interaction(probability_timestep)
                                if interaction_psychological == 1: 
                                    action_psychological = action
                                    break
                        
                        # An additional occupant has entered the room
                        elif zone_name in self.occupancy_dataframe.columns and self.occupancy[zone_name][timestep] > self.occupancy[zone_name][timestep-1]: 
                            for action in evaluation_actions: 
                                probability_timestep = 0
                                # Thermal comfort
                                if self.drivers_sec[room_habits][1] == 1:
                                    if action == 'close': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        k_irr = correcting_irradiance(irradiance)
                                        temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                        probability_timestep_thermal = horizontal_shift(self.probability_thermal_entering_occupation[action][zone_name], temperature_n)(indoor_operative_temperature) +( (k + k_irr)*self.standard_deviation_shading)
                                    # Opening is based on irradiance
                                    elif action == 'open': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        probability_timestep_thermal = generate_linear_function(self.probability_visual_entering_occupation_irradiance[action][zone_name])(irradiance) +( 2*k*self.standard_deviation_shading)
                                    probability_timestep = max(probability_timestep, probability_timestep_thermal)
                                    driver = 'thermal'
                                # Thermal comfort anticipate
                                if self.drivers_sec[room_habits][1] == 1 and self.thermal_anticipate == 2:
                                    # Assumption: the probability of thermal comfort is adjusted to future outdoor temperatures and expected temperature increases.
                                    # Only when maximum temperature is expected to increase to more than 25°C and there is an increase in temperature expected the next 3 days. 
                                    if self.daily_max_temperatures['next_day_max_for_max_diff'][day-1] > 25 and self.daily_max_temperatures['max_diff_next_1_3_days'][day-1] > 0: 
                                        if action == 'close': 
                                            k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                            k_temp = correcting_anticipate(self.daily_max_temperatures['next_day_max_for_max_diff'][day-1], self.daily_max_temperatures['next_day_max_for_max_diff'][day-1])
                                            k_irr = correcting_irradiance(irradiance)
                                            temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                            probability_timestep_thermalanticipate = horizontal_shift(self.probability_thermal_entering_occupation[action][zone_name], temperature_n)(indoor_operative_temperature) +( 2/3*(k + k_irr + k_temp)*self.standard_deviation_shading)
                                        # Opening is based on irradiance
                                        elif action == 'open': 
                                            k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                            probability_timestep_thermalanticipate = generate_linear_function(self.probability_visual_entering_occupation_irradiance[action][zone_name])(irradiance) +( 2*k*self.standard_deviation_shading)
                                        probability_timestep = max(probability_timestep, probability_timestep_thermalanticipate)
                                        if probability_timestep == probability_timestep_thermalanticipate: 
                                            driver = 'thermal_anticipate'
                                # Visual comfort 
                                if self.drivers_sec[room_habits][4] == 1:
                                    # Irradiance
                                    k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                    probability_timestep_irradiance = generate_linear_function(self.probability_visual_entering_occupation_irradiance[action][zone_name])(irradiance) +( 2*k*self.standard_deviation_shading)
                                    # Glare
                                    # Assumption: glare is only evaluated to close the shading. 
                                    if action == 'close': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        k_irr = correcting_irradiance(irradiance)
                                        probability_timestep_glare = horizontal_shift(self.probability_visual_entering_occupation_glare[action][zone_name], self.glare_evaluation[zone_name][timestep])(glare) +( (k+k_irr)*self.standard_deviation_shading)
                                    else: 
                                        probability_timestep_glare = 0
                                    probability_timestep_visual = max(probability_timestep_irradiance, probability_timestep_glare)
                                    probability_timestep = max(probability_timestep, probability_timestep_visual)
                                    if probability_timestep == probability_timestep_visual: 
                                        if probability_timestep == probability_timestep_irradiance: 
                                            driver = 'irradiance'
                                        else: 
                                            driver = 'glare'
                                interaction_psychological = stochastic_interaction(probability_timestep)
                                if interaction_psychological == 1: 
                                    action_psychological = action
                                    break
                        
                        # An occupant has left the room
                        elif zone_name in self.occupancy_dataframe.columns and self.occupancy[zone_name][timestep] < self.occupancy[zone_name][timestep-1]: 
                           for action in evaluation_actions: 
                                probability_timestep = 0
                                # Thermal comfort
                                if self.drivers_sec[room_habits][1] == 1:
                                    if action == 'close': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        k_irr = correcting_irradiance(irradiance)
                                        temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                        probability_timestep_thermal = horizontal_shift(self.probability_thermal_leaving_occupation[action][zone_name], temperature_n)(indoor_operative_temperature) +( (k + k_irr)*self.standard_deviation_shading)
                                    # Opening is based on irradiance
                                    elif action == 'open': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        probability_timestep_thermal = generate_linear_function(self.probability_visual_leaving_occupation_irradiance[action][zone_name])(irradiance) +( 2*k*self.standard_deviation_shading)
                                    probability_timestep = max(probability_timestep, probability_timestep_thermal)
                                    driver = 'thermal'
                                # Thermal comfort anticipate
                                if self.drivers_sec[room_habits][1] == 1 and self.thermal_anticipate == 2:
                                    # Assumption: the probability of thermal comfort is adjusted to future outdoor temperatures and expected temperature increases.
                                    # Only when maximum temperature is expected to increase to more than 25°C and there is an increase in temperature expected the next 3 days. 
                                    if self.daily_max_temperatures['next_day_max_for_max_diff'][day-1] > 25 and self.daily_max_temperatures['max_diff_next_1_3_days'][day-1] > 0: 
                                        if action == 'close': 
                                            k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                            k_temp = correcting_anticipate(self.daily_max_temperatures['next_day_max_for_max_diff'][day-1], self.daily_max_temperatures['next_day_max_for_max_diff'][day-1])
                                            k_irr = correcting_irradiance(irradiance)
                                            temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                            probability_timestep_thermalanticipate = horizontal_shift(self.probability_thermal_leaving_occupation[action][zone_name], temperature_n)(indoor_operative_temperature) +( 2/3*(k + k_irr + k_temp)*self.standard_deviation_shading)
                                        # Opening is based on irradiance
                                        elif action == 'open': 
                                            k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                            probability_timestep_thermalanticipate = generate_linear_function(self.probability_visual_leaving_occupation_irradiance[action][zone_name])(irradiance) +( 2*k*self.standard_deviation_shading)
                                        probability_timestep = max(probability_timestep, probability_timestep_thermalanticipate)
                                        if probability_timestep == probability_timestep_thermalanticipate: 
                                            driver = 'thermal_anticipate'
                                # Visual comfort 
                                if self.drivers_sec[room_habits][4] == 1:
                                    # Irradiance
                                    k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                    probability_timestep_irradiance = generate_linear_function(self.probability_visual_leaving_occupation_irradiance[action][zone_name])(irradiance) +( 2*k*self.standard_deviation_shading)
                                    # Glare
                                    # Assumption: glare is only evaluated to close the shading. 
                                    if action == 'close': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        k_irr = correcting_irradiance(irradiance)
                                        probability_timestep_glare = horizontal_shift(self.probability_visual_leaving_occupation_glare[action][zone_name], self.glare_evaluation[zone_name][timestep])(glare) +( (k+k_irr)*self.standard_deviation_shading)
                                    else: 
                                        probability_timestep_glare = 0
                                    probability_timestep_visual = max(probability_timestep_irradiance, probability_timestep_glare)
                                    probability_timestep = max(probability_timestep, probability_timestep_visual)
                                    if probability_timestep == probability_timestep_visual: 
                                        if probability_timestep == probability_timestep_irradiance: 
                                            driver = 'irradiance'
                                        else: 
                                            driver = 'glare'  
                                interaction_psychological = stochastic_interaction(probability_timestep)
                                if interaction_psychological == 1: 
                                    action_psychological = action
                                    break
                                   
                        # Bedrooms: scan on additional critical moments related to sleeping.
                        elif 'Bedroom' in zone_name and zone_name not in self.offices_list:   
                                        
                            # Person awake
                            if self.asleep_bedroom[zone_name][timestep] != self.occupancy[zone_name][timestep]: 
                                for action in evaluation_actions: 
                                    probability_timestep = 0
                                    # Thermal comfort
                                    if self.drivers_sec[room_habits][1] == 1:
                                        if action == 'close': 
                                            k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                            k_irr = correcting_irradiance(irradiance)
                                            temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                            probability_timestep_thermal = horizontal_shift(self.probability_thermal_during[action][zone_name], temperature_n)(indoor_operative_temperature) +( (k + k_irr)*self.standard_deviation_shading)
                                        # Opening is based on irradiance
                                        elif action == 'open': 
                                            k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                            probability_timestep_thermal = generate_linear_function(self.probability_visual_during_irradiance[action][zone_name])(irradiance) +( 2*k*self.standard_deviation_shading)
                                        probability_timestep = max(probability_timestep, probability_timestep_thermal)
                                        driver = 'thermal'
                                    # Thermal comfort anticipate
                                    if self.drivers_sec[room_habits][1] == 1 and self.thermal_anticipate == 2:
                                        # Assumption: the probability of thermal comfort is adjusted to future outdoor temperatures and expected temperature increases.
                                        # Only when maximum temperature is expected to increase to more than 25°C and there is an increase in temperature expected the next 3 days. 
                                        if self.daily_max_temperatures['next_day_max_for_max_diff'][day-1] > 25 and self.daily_max_temperatures['max_diff_next_1_3_days'][day-1] > 0: 
                                            if action == 'close': 
                                                k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                                k_temp = correcting_anticipate(self.daily_max_temperatures['next_day_max_for_max_diff'][day-1], self.daily_max_temperatures['next_day_max_for_max_diff'][day-1])
                                                k_irr = correcting_irradiance(irradiance)
                                                temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                                probability_timestep_thermalanticipate = horizontal_shift(self.probability_thermal_during[action][zone_name], temperature_n)(indoor_operative_temperature) +( 2/3*(k + k_irr + k_temp)*self.standard_deviation_shading)
                                            # Opening is based on irradiance
                                            elif action == 'open': 
                                                k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                                probability_timestep_thermalanticipate = generate_linear_function(self.probability_visual_during_irradiance[action][zone_name])(irradiance) +( 2*k*self.standard_deviation_shading)
                                            probability_timestep = max(probability_timestep, probability_timestep_thermalanticipate)
                                            if probability_timestep == probability_timestep_thermalanticipate: 
                                                driver = 'thermal_anticipate'
                                    # Visual comfort 
                                    if self.drivers_sec[room_habits][4] == 1:
                                        # Irradiance
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        probability_timestep_irradiance = generate_linear_function(self.probability_visual_during_irradiance[action][zone_name])(irradiance) +( 2*k*self.standard_deviation_shading)
                                        # Glare
                                        # Assumption: glare is only evaluated to close the shading. 
                                        if action == 'close': 
                                            k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                            k_irr = correcting_irradiance(irradiance)
                                            probability_timestep_glare = horizontal_shift(self.probability_visual_during_glare[action][zone_name], self.glare_evaluation[zone_name][timestep])(glare) +( (k+k_irr)*self.standard_deviation_shading)
                                        else: 
                                            probability_timestep_glare = 0
                                        probability_timestep_visual = max(probability_timestep_irradiance, probability_timestep_glare)
                                        probability_timestep = max(probability_timestep, probability_timestep_visual)
                                        if probability_timestep == probability_timestep_visual: 
                                            if probability_timestep == probability_timestep_irradiance: 
                                                driver = 'irradiance'
                                            else: 
                                                driver = 'glare'  
                                    interaction_psychological = stochastic_interaction(probability_timestep)
                                    if interaction_psychological == 1: 
                                        action_psychological = action
                                        break
                        
                                        
                            # All occupants in the bedroom are asleep, making interactions impossible
                            else: 
                                interaction_psychological = 0
                                            
                        # No external critical moment detected 
                        else:
                            for action in evaluation_actions: 
                                probability_timestep = 0
                                # Thermal comfort
                                if self.drivers_sec[room_habits][1] == 1:
                                    if action == 'close': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        k_irr = correcting_irradiance(irradiance)
                                        temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                        probability_timestep_thermal = horizontal_shift(self.probability_thermal_during[action][zone_name], temperature_n)(indoor_operative_temperature) +( (k + k_irr)*self.standard_deviation_shading)
                                    # Opening is based on irradiance
                                    elif action == 'open': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        probability_timestep_thermal = generate_linear_function(self.probability_visual_during_irradiance[action][zone_name])(irradiance) +( 2*k*self.standard_deviation_shading)
                                    probability_timestep = max(probability_timestep, probability_timestep_thermal)
                                    driver = 'thermal'
                                # Thermal comfort anticipate
                                if self.drivers_sec[room_habits][1] == 1 and self.thermal_anticipate == 2:
                                    # Assumption: the probability of thermal comfort is adjusted to future outdoor temperatures and expected temperature increases.
                                    # Only when maximum temperature is expected to increase to more than 25°C and there is an increase in temperature expected the next 3 days. 
                                    if self.daily_max_temperatures['next_day_max_for_max_diff'][day-1] > 25 and self.daily_max_temperatures['max_diff_next_1_3_days'][day-1] > 0: 
                                        if action == 'close': 
                                            k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                            k_temp = correcting_anticipate(self.daily_max_temperatures['next_day_max_for_max_diff'][day-1], self.daily_max_temperatures['next_day_max_for_max_diff'][day-1])
                                            k_irr = correcting_irradiance(irradiance)
                                            temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                            probability_timestep_thermalanticipate = horizontal_shift(self.probability_thermal_during[action][zone_name], temperature_n)(indoor_operative_temperature) +( 2/3*(k + k_irr + k_temp)*self.standard_deviation_shading)
                                        # Opening is based on irradiance
                                        elif action == 'open': 
                                            k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                            probability_timestep_thermalanticipate = generate_linear_function(self.probability_visual_during_irradiance[action][zone_name])(irradiance) +( 2*k*self.standard_deviation_shading)
                                        probability_timestep = max(probability_timestep, probability_timestep_thermalanticipate)
                                        if probability_timestep == probability_timestep_thermalanticipate: 
                                            driver = 'thermal_anticipate'
                                # Visual comfort 
                                if self.drivers_sec[room_habits][4] == 1:
                                    # Irradiance
                                    k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                    probability_timestep_irradiance = generate_linear_function(self.probability_visual_during_irradiance[action][zone_name])(irradiance) +( 2*k*self.standard_deviation_shading)
                                    # Glare
                                    # Assumption: glare is only evaluated to close the shading. 
                                    if action == 'close': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        k_irr = correcting_irradiance(irradiance)
                                        probability_timestep_glare = horizontal_shift(self.probability_visual_during_glare[action][zone_name], self.glare_evaluation[zone_name][timestep])(glare) +( (k+k_irr)*self.standard_deviation_shading)
                                    else: 
                                        probability_timestep_glare = 0
                                    probability_timestep_visual = max(probability_timestep_irradiance, probability_timestep_glare)
                                    probability_timestep = max(probability_timestep, probability_timestep_visual)
                                    if probability_timestep == probability_timestep_visual: 
                                        if probability_timestep == probability_timestep_irradiance: 
                                            driver = 'irradiance'
                                        else: 
                                            driver = 'glare' 
                                interaction_psychological = stochastic_interaction(probability_timestep)
                                if interaction_psychological == 1: 
                                    action_psychological = action
                                    break

                    else: 
                       interaction_psychological = 0
                
                
                # Evaluation of shading 
                if interaction_psychological == 0 and interaction_practical == 0: 
                    occlusion_timestep_psychological = previousstate_shading
                elif interaction_psychological == 0 and interaction_practical == 1: 
                    occlusion_timestep_psychological = 0               
                # Stochastic evaluation of the change in occlusion rate
                else: 
                    previous_occlusion = int(previousstate_shading/0.25)
                    if driver == 'irradiance' or driver == 'glare': 
                         probs = self.probabilities_occlusion['visual'][action_psychological]
                    else: 
                        probs = self.probabilities_occlusion[driver][action_psychological]
                    prob = probs[:,previous_occlusion]
                    rnd = np.random.random()
                    idx = 1
                    while rnd >= prob[idx - 1]:
                        idx += 1
                    if action_psychological == 'open': 
                        occlusion_timestep_psychological = (idx-1) * 0.25
                    else: 
                        occlusion_timestep_psychological = idx * 0.25
                        
                # Compare occlusion rate 
                occlusion_timestep = max(occlusion_timestep_practical, occlusion_timestep_psychological)
            
                occlusion[orientation] = occlusion_timestep    
            action_practical = [action_practical_darkness, action_practical_privacy, action_practical_security]
       
            return occlusion, action_practical, interaction_practical, interaction_psychological, timestep_previous_shading
            
        def calculate_illuminance_glare_adjusted_occlusion_rate(zone_name, overview_occlusion_rates): 
            def make_eplaunch_options(idf):
                """Make options for run, so that it runs like EPLaunch on Windows"""
                idfversion = idf.idfobjects['version'][0].Version_Identifier.split('.')
                idfversion.extend([0] * (3 - len(idfversion)))
                fname = idf.idfname
                options = {
                    'output_prefix': os.path.basename(fname).split('.')[0],
                    'output_suffix': 'C',
                    'output_directory': os.path.dirname(fname),
                    'readvars': True,
                    'expandobjects': True
                    }
                return options
                
            month = self.api.exchange.month(state)
            day_of_month = self.api.exchange.day_of_month(state)
            time_index = round((self.api.exchange.current_time(state) - 2/60)/(2/60)) # Timesteps of two minutes
            
            idf = self.idf
            
            # Adjust run period to the corresponding day
            run_period = idf.getobject('RUNPERIOD', 'RunPeriodDay')
            run_period.Begin_Month = month
            run_period.Begin_Day_of_Month = day_of_month
            run_period.End_Month = month
            run_period.End_Day_of_Month = day_of_month
            
            # Adjust the occlusion rate for the different windoworientations of the room
            for orientation, occlusion_rate in overview_occlusion_rates.items(): 
                if occlusion_rate == 0.25: 
                    schedule_name = "ScheduleShadingControl" + zone_name+'.'+str(orientation)+ '_0.25'
                    schedule = idf.getobject('SCHEDULE:CONSTANT', schedule_name)
                    schedule.Hourly_Value = 1
                    for occlusion in ['_0.5', '_0.75', '_1']: 
                        schedule_name = "ScheduleShadingControl" + zone_name+'.'+str(orientation) + occlusion
                        schedule = idf.getobject('SCHEDULE:CONSTANT', schedule_name)
                        schedule.Hourly_Value = 0
                if occlusion_rate == 0.5: 
                    for occlusion in ['_0.25', '_0.5']:
                        schedule_name = "ScheduleShadingControl" + zone_name+'.'+str(orientation)+ occlusion
                        schedule = idf.getobject('SCHEDULE:CONSTANT', schedule_name)
                        schedule.Hourly_Value = 1
                    for occlusion in ['_0.75', '_1']:
                        schedule_name = "ScheduleShadingControl" + zone_name+'.'+str(orientation)+ occlusion
                        schedule = idf.getobject('SCHEDULE:CONSTANT', schedule_name)
                        schedule.Hourly_Value = 0
                if occlusion_rate == 0.75: 
                    for occlusion in ['_0.25', '_0.5', '_0.75']:
                        schedule_name = "ScheduleShadingControl" + zone_name+'.'+str(orientation)+ occlusion
                        schedule = idf.getobject('SCHEDULE:CONSTANT', schedule_name)
                        schedule.Hourly_Value = 1
                    schedule_name = "ScheduleShadingControl" + zone_name+'.'+str(orientation)+ '_1'
                    schedule = idf.getobject('SCHEDULE:CONSTANT', schedule_name)
                    schedule.Hourly_Value = 0
                if occlusion_rate == 1: 
                    for occlusion in ['_0.25', '_0.5', '_0.75', '_1']:
                        schedule_name = "ScheduleShadingControl" + zone_name+'.'+str(orientation)+ occlusion
                        schedule = idf.getobject('SCHEDULE:CONSTANT', schedule_name)
                        schedule.Hourly_Value = 1
            
            idf.saveas(self.output_path + '/calculations_' + zone_name + '_' + str(time_index) + '.idf')
            options = make_eplaunch_options(idf)
            
            # Run simulation
            idf.run(**options)
            
            # Process simulation output: extract the daylighting illuminance and glare index
            if os.path.exists(self.output_path + '/calculations_' + zone_name + '_' + str(time_index) + '.csv'): 
                simulation_output = pd.read_csv(self.output_path + '/calculations_' + zone_name + '_' + str(time_index) +'.csv')
                daylighting_ref_point_columns = [col for col in simulation_output.columns if 'Daylighting Reference Point' in col]
                daylighting_ref_point = simulation_output[daylighting_ref_point_columns]
                daylighting_ref_point_room_columns = [col for col in daylighting_ref_point.columns if zone_name.lower() in col.lower()]
                daylighting_ref_point_room = daylighting_ref_point[daylighting_ref_point_room_columns]
                illuminance_col = [col for col in daylighting_ref_point_room.columns if 'Illuminance' in col]
                illuminance_adjustment_shading = daylighting_ref_point_room[illuminance_col].iloc[time_index].mean()
                glare_index_col = [col for col in daylighting_ref_point_room.columns if 'Glare Index' in col and '1' in col]
                glare_index_adjustment_shading = daylighting_ref_point_room[glare_index_col].iloc[time_index].mean()
            
            output_path_prefix = self.output_path + '/calculations_' + zone_name + '_' + str(time_index)
            for file_path in glob.glob(output_path_prefix + "*"):
                os.remove(file_path)
            
            return illuminance_adjustment_shading, glare_index_adjustment_shading
        
        def neutral_indoor_temperature(zone, timestep, offices_list, bedrooms_list, temperature_ref): 
            if zone in ['Living', 'Kitchen', 'LivingKitchen'] or zone in offices_list: 
                if temperature_ref < 12.5: 
                    temperature_n = 20.4 + 0.06*temperature_ref
                else: 
                    temperature_n = 16.63 + 0.36*temperature_ref
            elif zone in ['Bathroom']: 
                if temperature_ref < 11: 
                    temperature_n = 0.112*temperature_ref + 22.65
                else:
                    temperature_n = 0.306*temperature_ref + 20.32
            elif zone in bedrooms_list: 
                if temperature_ref < 0: 
                    temperature_n = 16
                elif 0 <= temperature_ref < 12.6: 
                    temperature_n = 0.23*temperature_ref + 16
                elif 12.6 <= temperature_ref < 21.8: 
                    temperature_n = 0.77*temperature_ref + 9.18
                else: 
                    temperature_n = 26
            else: 
                temperature_n = 26
            return temperature_n
            
        def compare_occlusion_rates(occlusion_interaction, occlusion_previous): 
            for key in occlusion_interaction:
                if key not in occlusion_previous:
                    raise KeyError(f"Orientation '{key}' not in occlusion_previous")
                if occlusion_interaction[key] > occlusion_previous[key]:
                    return False
            return True
        
        if not self.handles_set: 
            for zone_name in self.rooms_lighting: 
                self.handle_previouslighting[zone_name] = self.api.exchange.get_global_handle(state, 'PreviousStateLighting'+zone_name)
                self.handle_timestepinteraction[zone_name] = self.api.exchange.get_global_handle(state, 'TimestepInteraction'+zone_name)
                if zone_name in self.rooms_daylighting:
                    self.handle_illuminance[zone_name] = self.api.exchange.get_variable_handle(state, 'Daylighting Reference Point 1 Illuminance', 'DaylightingControl'+zone_name)
                elif zone_name in self.rooms_daylighting_2:
                    self.handle_illuminance[zone_name+"_1"] = self.api.exchange.get_variable_handle(state, 'Daylighting Reference Point 1 Illuminance', 'DaylightingControl'+zone_name)
                    self.handle_illuminance[zone_name+"_2"] = self.api.exchange.get_variable_handle(state, 'Daylighting Reference Point 2 Illuminance', 'DaylightingControl'+zone_name)
                self.handle_schedulelighting[zone_name] = self.api.exchange.get_actuator_handle(state, "Schedule:Constant", "Schedule Value", "ScheduleLight"+zone_name)
            for zone_name in self.rooms_shading:
                if zone_name in self.rooms_3_orientations:
                    number_orientations = 3
                elif zone_name in self.rooms_2_orientations: 
                    number_orientations = 2
                else: 
                    number_orientations = 1
                if 'Bathroom' in zone_name: 
                    room_habits = 'bathroom'
                elif 'Bedroom' in zone_name and zone_name not in self.offices_list: 
                    room_habits = 'bedroom'
                elif 'Corridor' in zone_name or 'Hallway' in zone_name: 
                    room_habits = 'other'
                elif 'Kitchen' in zone_name or 'Living' in zone_name: 
                    room_habits = 'living'
                elif zone_name in self.offices_list: 
                    room_habits = 'living'
                elif 'Toilet' in zone_name or 'Storage' in zone_name: 
                    room_habits = 'other'
                self.handle_indooroperativetemperature[zone_name] = self.api.exchange.get_variable_handle(state, 'Zone Operative Temperature', zone_name)
                self.handle_glare[zone_name] = self.api.exchange.get_variable_handle(state, 'Daylighting Reference Point 1 Glare Index', 'DaylightingControl'+zone_name)
                self.handle_darkness[zone_name] = self.api.exchange.get_global_handle(state, 'ActionPracticalDarkness' + zone_name)
                self.handle_privacy[zone_name] = self.api.exchange.get_global_handle(state, 'ActionPracticalPrivacy' + zone_name)
                self.handle_security[zone_name] = self.api.exchange.get_global_handle(state, 'ActionPracticalSecurity' + zone_name)
                self.handle_previoustimestepshading[zone_name] = self.api.exchange.get_global_handle(state, 'TimestepPreviousShading' + zone_name)
                self.handle_timesteppreviousevaluation[zone_name] = self.api.exchange.get_global_handle(state, 'TimestepPreviousEvaluation' + zone_name)
                for orientation in range(1,number_orientations+1): 
                    window_orientation = self.overview_windows[zone_name][orientation]
                    self.handle_irradiance[zone_name+'.'+str(orientation)] = self.api.exchange.get_variable_handle(state, "Surface Outside Face Incident Solar Radiation Rate per Area", window_orientation)
                    self.handle_previousstateshading[zone_name+'.'+str(orientation)] = self.api.exchange.get_global_handle(state,'PreviousStateShading' + zone_name + '.' + str(orientation))
                    for schedule in [0.25, 0.5, 0.75, 1]:
                        self.handle_scheduleshading[zone_name+'.'+str(orientation)+'_'+str(schedule)] = self.api.exchange.get_actuator_handle(state, "Schedule:Constant","Schedule Value","ScheduleShadingControl" + zone_name+'.'+str(orientation)+'_'+str(schedule))
            self.handles_set = True
        
        hour = round(self.api.exchange.current_time(state),2)
        day = self.api.exchange.day_of_year(state)
        timestep_simulation = self.api.exchange.zone_time_step(state)
        timestep = round((day-1)*720+(hour/0.033333)-1) # simulation runs at two minute timesteps
        sun_up = self.api.exchange.sun_is_up(state)

        # Simulation per timestep
        for zone_name in self.rooms_lighting: 
                
            # Exchange input information with EnergyPlus
            timestep_interaction = self.api.exchange.get_global_value(state, self.handle_timestepinteraction[zone_name])
            previousstate_lighting = self.api.exchange.get_global_value(state, self.handle_previouslighting[zone_name])
            if zone_name in self.rooms_daylighting:
                illuminance = self.api.exchange.get_variable_value(state, self.handle_illuminance[zone_name])
            elif zone_name == self.rooms_daylighting_2: # Average of two Daylighting Reference Points
                illuminance = (self.api.exchange.get_variable_value(state, self.handle_illuminance[zone_name+"_1"])+self.api.exchange.get_variable_value(state, self.handle_illuminance[zone_name+"_2"]))/2
            
            # Simulate the shading state        
            if zone_name in self.rooms_shading:  
                # Set the number of orienations: 
                if zone_name in self.rooms_3_orientations:
                    number_orientations = 3
                elif zone_name in self.rooms_2_orientations: 
                    number_orientations = 2
                else: 
                    number_orientations = 1
                indoor_operative_temperature = self.api.exchange.get_variable_value(state, self.handle_indooroperativetemperature[zone_name])
                glare = self.api.exchange.get_variable_value(state, self.handle_glare[zone_name])
            
                action_practical_darkness = self.api.exchange.get_global_value(state, self.handle_darkness[zone_name])
                action_practical_privacy = self.api.exchange.get_global_value(state, self.handle_privacy[zone_name])
                action_practical_security = self.api.exchange.get_global_value(state, self.handle_security[zone_name])
                timestep_previous_shading = self.api.exchange.get_global_value(state, self.handle_previoustimestepshading[zone_name])
                timestep_previous_evaluation = self.api.exchange.get_global_value(state, self.handle_timesteppreviousevaluation[zone_name])
                overview_occlusion_rates, action_practical, interaction_practical, interaction_psychological, timestep_previous_shading_interaction = simulate_manual_shading(self, zone_name, hour, day, timestep, sun_up, indoor_operative_temperature, glare, action_practical_darkness, action_practical_privacy, action_practical_security, timestep_previous_shading)
                occlusion_previous = {}
                for orientation in range(1, number_orientations+1): 
                    occlusion_previous[orientation] = self.api.exchange.get_global_value(state, self.handle_previousstateshading[zone_name + '.' + str(orientation)])   
            
            # Analyse shading behaviour
            if zone_name not in self.rooms_shading: 
                # No solar shading.
                # Simulate lighting interactions for the simulated indoor illuminance.
                state_lighting = simulate_manual_lighting(self, zone_name, hour, day, timestep, sun_up, previousstate_lighting, illuminance, 0)
            elif interaction_practical == 0 and interaction_psychological == 0:  
                # No interaction with solar shading detected.
                # Simulate lighting interactions for the simulated indoor illuminance.
                state_lighting = simulate_manual_lighting(self, zone_name, hour, day, timestep, sun_up, previousstate_lighting, illuminance, 0)
                occlusion_timestep = occlusion_previous         
            elif illuminance == 0 and overview_occlusion_rates != {i: 1 for i in range(1, number_orientations + 1)}: 
                # Dark outside and shading was not fully closed, lighting interaction is independent of shading. 
                illuminance_shading_interaction = 0
                state_lighting = simulate_manual_lighting(self, zone_name, hour, day, timestep, sun_up, previousstate_lighting, illuminance_shading_interaction, 1)
                occlusion_timestep = overview_occlusion_rates
                timestep_previous_shading = timestep_previous_shading_interaction
            elif interaction_practical == 1:     
                # Practical interaction with solar shading. 
                # Calculate the indoor illuminance after shading adjustment
                illuminance_shading_interaction, glare_index_interaction = calculate_illuminance_glare_adjusted_occlusion_rate(zone_name, overview_occlusion_rates)
                # And simulate the lighting interactions after shading adjustment
                state_lighting = simulate_manual_lighting(self, zone_name, hour, day, timestep, sun_up, previousstate_lighting, illuminance_shading_interaction, 1)
                occlusion_timestep = overview_occlusion_rates
                timestep_previous_shading = timestep_previous_shading_interaction
            elif compare_occlusion_rates(overview_occlusion_rates, occlusion_previous) == True: 
                # Reopen shading, allows more daylighting to come in and shading will consequently always be adjusted. 
                # Calculate the indoor illuminance after shading adjustment
                illuminance_shading_interaction, glare_index_interaction = calculate_illuminance_glare_adjusted_occlusion_rate(zone_name, overview_occlusion_rates)
                # And simulate the lighting interactions after shading adjustment
                state_lighting = simulate_manual_lighting(self, zone_name, hour, day, timestep, sun_up, previousstate_lighting, illuminance_shading_interaction, 1)
                occlusion_timestep = overview_occlusion_rates
                timestep_previous_shading = timestep_previous_shading_interaction
            else: 
                # Shading closing interactions driven by discomfort                  
                # Simulate lighting interactions for the simulated indoor illuminance.
                state_lighting_without_interaction = simulate_manual_lighting(self, zone_name, hour, day, timestep, sun_up, previousstate_lighting, illuminance, 0)
                if state_lighting_without_interaction == 1: 
                    state_lighting_interaction = 1
                else: 
                    # Calculate the indoor illuminance after shading adjustment
                    illuminance_shading_interaction, glare_index_interaction = calculate_illuminance_glare_adjusted_occlusion_rate(zone_name, overview_occlusion_rates)
                    # And resimulate the lighting interactions after shading adjustment
                    state_lighting_interaction = simulate_manual_lighting(self, zone_name, hour, day, timestep, sun_up, previousstate_lighting, illuminance_shading_interaction, 1)
                 
                # Evaluate lighting state and shading state
                if state_lighting_interaction == state_lighting_without_interaction: 
                    # Adjusting shading doesn't affect lighting state
                    state_lighting = state_lighting_interaction
                    occlusion_timestep = occlusion_previous
         
                # Evaluation dependending on the indoor parameters. 
                else: 
                    # Indoor illuminance, indoor operative temperature and glare are taken into account to set the probability, as well as time time since entering and previous interaction for moments during occupancy. 
                    # Correction temperature
                    temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])                                           
                    if temperature_n - indoor_operative_temperature > 3: 
                        k_temp = -1
                    elif indoor_operative_temperature - temperature_n > 3: 
                        k_temp = 1
                    else: 
                        k_temp = (indoor_operative_temperature - temperature_n)/3
                    # Correction glare
                    if self.glare_evaluation[zone_name][timestep] - glare > 3: 
                        k_glare = -1
                    elif glare - self.glare_evaluation[zone_name][timestep] > 3: 
                        k_glare = 1
                    else: 
                        k_glare = (glare - self.glare_evaluation[zone_name][timestep])/3
                    if self.occupancy[zone_name][timestep] == self.occupancy[zone_name][timestep-1] and self.occupancy[zone_name][timestep] == self.occupancy[zone_name][timestep+1]: 
                        # During occupation
                        i_occpros = 1
                        while self.occupancy[zone_name][timestep + i_occpros] >= 1 and timestep + i_occpros <= 262800 and i_occpros < 10: 
                            i_occpros += 1
                        t_occprev = 1
                        while self.occupancy[zone_name][timestep - t_occprev] >= 1 and timestep - t_occprev > 0: 
                            t_occprev += 1
                        # Time since previous interaction
                        t_interaction = abs(timestep - timestep_interaction)
                        if t_interaction < 60:
                            i_interaction = 1
                        else: 
                            i_interaction = 1 + 0.15 *(min(120, t_interaction)-60)    
                        # Time since enterance
                        if t_occprev <= 10: 
                            i_occprev = 10
                        elif t_occprev < 30: 
                            i_occprev = 10 - 0.45*(t_occprev - 10)
                        else: 
                            i_occprev = 1    
                        # Consistency in decision: time since previous decision evaluation
                        if timestep - timestep_previous_evaluation > 10: 
                            i_eval = 10
                        else: 
                            i_eval = 1
                        k_occ = 11 - min(i_occpros, i_occprev, i_interaction, i_eval)
                        k = k_temp + k_glare
                        probability_decision = (illuminance_shading_interaction/self.minimum_illuminance[zone_name][timestep] + k*self.standard_deviation_lighting)/k_occ
                    elif self.occupancy[zone_name][timestep-1] != 0 and self.occupancy[zone_name][timestep+1] != 0: 
                        # Reluctant as other occupants present
                        # Time since previous interaction
                        t_interaction = abs(timestep - timestep_interaction)
                        if t_interaction < 60:
                            i_interaction = 1
                        else: 
                            i_interaction = 1 + 0.15 *(min(120, t_interaction)-60)
                        # Consistency in decision: time since previous decision evaluation
                        if timestep - timestep_previous_evaluation > 10: 
                            i_eval = 10
                        else: 
                            i_eval = 1    
                        k_occ = (11 - min(i_interaction, i_eval))/2
                        k = k_temp + k_glare
                        probability_decision = (illuminance_shading_interaction/self.minimum_illuminance[zone_name][timestep] + k*self.standard_deviation_lighting)/k_occ
                    else: 
                        # Time since previous interaction
                        t_interaction = abs(timestep - timestep_interaction)
                        if t_interaction < 60:
                            i_interaction = 1
                        else: 
                            i_interaction = 1 + 0.15 *(min(120, t_interaction)-60)    
                        k_occ = (11 - i_interaction)/3
                        k = k_temp + k_glare
                        probability_decision = (illuminance_shading_interaction/self.minimum_illuminance[zone_name][timestep] + k*self.standard_deviation_lighting)/k_occ
                    
                    rnd = np.random.random()
                    if rnd >= probability_decision:
                        # Prioritising daylighting
                        state_lighting = state_lighting_without_interaction
                        occlusion_timestep = occlusion_previous 
                    else: 
                        # Prioritising shading
                        state_lighting = state_lighting_interaction
                        occlusion_timestep = overview_occlusion_rates
                        timestep_previous_shading = timestep_previous_shading_interaction
                    timestep_previous_evaluation = timestep

            # Return lighting state to EnergyPlus
            self.api.exchange.set_actuator_value(state, self.handle_schedulelighting[zone_name], state_lighting)
            self.api.exchange.set_global_value(state, self.handle_previouslighting[zone_name], state_lighting)

            if state_lighting != previousstate_lighting: 
                timestep_interaction = timestep
            elif zone_name in self.rooms_shading and occlusion_timestep != occlusion_previous: 
                timestep_interaction = timestep           
            self.api.exchange.set_global_value(state, self.handle_timestepinteraction[zone_name],timestep_interaction)
            
            # Return shading state to EnergyPlus
            if zone_name in self.rooms_shading: 
                for orientation in range(1, number_orientations+1): 
                    # Set the occlusion rate of the solar shading in the simulation and write it to a temporary variable. 
                    if occlusion_timestep[orientation] == 0: 
                        schedules_0 = [0.25, 0.5, 0.75, 1]
                        schedules_1 = []
                    elif occlusion_timestep[orientation] == 0.25: 
                        schedules_0 = [0.5, 0.75, 1]
                        schedules_1 = [0.25]
                    elif occlusion_timestep[orientation] == 0.5: 
                        schedules_0 = [0.75, 1]
                        schedules_1 = [0.25, 0.5]
                    elif occlusion_timestep[orientation] == 0.75: 
                        schedules_0 = [1]
                        schedules_1 = [0.25, 0.5, 0.75]
                    elif occlusion_timestep[orientation] == 1: 
                        schedules_0 = []
                        schedules_1 = [0.25, 0.5, 0.75, 1]
                    
                    # Return the values to EnergyPlus, the windows are splitted and shading is separetely controlled per part. 
                    for schedule in schedules_0: 
                        self.api.exchange.set_actuator_value(state, self.handle_scheduleshading[zone_name+'.'+str(orientation)+'_'+str(schedule)], 0)
                    for schedule in schedules_1: 
                        self.api.exchange.set_actuator_value(state, self.handle_scheduleshading[zone_name+'.'+str(orientation)+'_'+str(schedule)], 1)
                    self.api.exchange.set_global_value(state, self.handle_previousstateshading[zone_name+'.'+str(orientation)], occlusion_timestep[orientation])
                                                               
                self.api.exchange.set_global_value(state,  self.handle_darkness[zone_name], action_practical[0])
                self.api.exchange.set_global_value(state, self.handle_privacy[zone_name], action_practical[1])
                self.api.exchange.set_global_value(state, self.handle_security[zone_name], action_practical[2])
                self.api.exchange.set_global_value(state, self.handle_previoustimestepshading[zone_name], timestep_previous_shading)
                self.api.exchange.set_global_value(state, self.handle_timesteppreviousevaluation[zone_name], timestep_previous_evaluation)                                                                   

        return 0        
       