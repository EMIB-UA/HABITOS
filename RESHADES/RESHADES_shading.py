# ******************************************
# ENERGYPLUS PLUGIN: MANUAL SHADING CONTROL
# ******************************************

"""
Created on July 19 2024

@author: LVanThillo
"""

# ----------------------
# INPUT ENERGYPLUS FILE
# ----------------------

# Required input EnergyPlus per zone: 

#   PythonPlugin:Variables,
#     PythonPluginGlobalVariables,   !- Name
#     ActionPracticalDarknessZone,   !- Variable Name 1
#     ActionPracticalPrivacyZone,    !- Variable Name 2
#     ActionPracticalSecurityZone,   !- Variable Name 3
#     TimestepPreviousShadingZone,   !- Variable Name 4

# Required input EnergyPlus per zone and per orientation: 

#   PythonPlugin:Variables,
#     PythonPluginGlobalVariables,   !- Name
#     PreviousStateShadingZone.Orientation,  !- Variable Name 1

# Required input EnergyPlus per zone, per orientation and per occlusion rate: 
    
#   Schedule:Constant,ScheduleManualShadingControlZone.Orientation_OcclusionRate,Fraction,0;

#   WindowShadingControl,
#     ControlScreensZoneOrientationOcclusionRate,  !- Name
#     Zone,                    !- Zone Name
#     ,                        !- Shading Control Sequence Number
#     ExteriorScreen,          !- Shading Type
#     ,                        !- Construction with Shading Name
#     OnIfScheduleAllows,      !- Shading Control Type
#     ScheduleManualShadingControlZone.Orientation_OcclusionRate,  !- Schedule Name
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

# -----------------------------------
# PYTHON CODE MANUAL SHADING CONTROL
# -----------------------------------

import os
import re
import random
import math
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import differential_evolution
from scipy.optimize import least_squares
from pyenergyplus.plugin import EnergyPlusPlugin

def parse_datetime(month, date, time): 
        date_format = "%m-%d %H:%M"
        return datetime.strptime(str(month) + "-" + str(date) +" "+ str(int(time)-1) + ":00", date_format)
        

class ShadingControl(EnergyPlusPlugin):
        
    def __init__(self): 
        super().__init__()    
        
        # GIVE THE WEATHER FILE
        weather_file = r'' # Refer to the weather file. 

        # PROVIDE INFORMATION ABOUT THE BUILDING GEOMETRY IN ENERGYPLUS
        rooms_shading = [] # List all zone names with solar shading.
        rooms_3_orientations = [] # List the zone names with windows in three orientations.
        rooms_2_orientations = [] # List the zone names with windows in two orientations
        overview_windows = {} # Define a specific window name for each orientation of the rooms
        self.rooms_shading = rooms_shading
        self.rooms_3_orientations = rooms_3_orientations
        self.rooms_2_orientations = rooms_2_orientations
        self.overview_windows = overview_windows
        
        # Automatically generated
        bedrooms = [room for room in rooms_lighting if 'Bedroom' in room] # Sum the bedrooms
           
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
        probs = np.loadtxt(directory_data+'/driver_livingkitchen.txt', float)
        driver_living = get_probability(np.random.random(), probs)
        # Step 1b. How often do the family on average switch the state of the screens? This family characteristic is determined independently of the main habits.
        # (1) Never, (2) several times a month, (3) several times a week, (4) daily or (5) multiple times a day
        probs = np.loadtxt(directory_data+'/number_interactions.txt', float)
        number_interactions = get_probability(np.random.random(), probs)
        # Step 1c. Define the activeness of the family with regard to adapting their screens.
        # Does the family anticipate on thermal discomfort by closing their shades on beforehand.
        # (1) Family does interact when experiencing thermal discomfort or (2) family anticipate on future risk of overheating.
        probs = np.loadtxt(directory_data+'/thermaldiscomfort.txt', float)
        thermal_anticipate = get_probability(np.random.random(), probs)
        self.thermal_anticipate = thermal_anticipate
        
        # Step 2. Define the habits at room level and per season
        # Step 2a. What is the main driver in the bathroom, bedroom and other rooms? This relation depends on the main driver selected for the living room/kitchen.
        # (1) Darkness, (2) thermal comfort, (3) privacy, (4) security and (5) visual comfort
        # Bedroom
        probs = np.loadtxt(directory_data+'/driver_bedroom.txt', float)
        driver_bedroom = get_probability(np.random.random(), probs[:, driver_living - 1])
        # Bathroom
        probs = np.loadtxt(directory_data+'/driver_bathroom.txt', float)
        driver_bathroom = get_probability(np.random.random(), probs[:, driver_living - 1])
        # Other rooms
        probs = np.loadtxt(directory_data+'/driver_other.txt', float)
        driver_other = get_probability(np.random.random(), probs[:, driver_living - 1])
        # Step 2b. What secondary drivers do also cause the occupants to interact with their shading installations?
        # (1) Darkness, (2) darkness, thermal comfort, (3) darkness, privacy, (4) darkness, security, (5) darkness, visual comfort, (6) darkness, thermal comfort, privacy, (7) darkness, thermal comfort, security, (8) darkness, thermal comfort, visual comfort, (9) darkness, security, privacy, (10) darkness, security, visual comfort, (11) darkness, privacy, visual comfort, (12) darkness, thermal comfort, privacy, security, (13) darkness, thermal comfort, privacy, visual comfort, (14) darkness, thermal comfort, security, visual comfort, (15) darkness, privacy, security, visual comfort, (16) darkness, thermal comfort, privacy, security, visual comfort, (17) thermal comfort, (18) thermal comfort, privacy, (19) thermal comfort, security, (20) thermal comfort, visual comfort, (21) thermal comfort, privacy, security, (22) thermal comfort, privacy, visual comfort, (23) thermal comfort, security, visual comfort, (24) thermal comfort, privacy, security, visual comfort, (25) privacy, (26) privacy, security, (27) privacy, visual comfort, (28) privacy, security, visual comfort, (29) security, (30) security, visual comfort, (31) visual comfort
        # The combination of drivers per room is identified and returned as a list in which 1 indicates that the driver is adopted as habit by the family.
        # (0) Darkness, (1) thermal comfort, (2) privacy, (3) security and (4) visual comfort
        # Living room/kitchen/offices
        probs = np.loadtxt(directory_data+'/drivers_sec_living.txt', float)
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
        probs = np.loadtxt(directory_data+'/drivers_sec_bedroom.txt', float)
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
        probs = np.loadtxt(directory_data+'/drivers_sec_bathroom.txt', float)
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
        probs = np.loadtxt(directory_data+'/number_interactions_springautumn.txt', float)
        number_interactions_springautumn = get_probability(np.random.random(), probs[:, number_interactions - 1])
        # Winter
        probs = np.loadtxt(directory_data+'/number_interactions_winter.txt', float)
        number_interactions_winter = get_probability(np.random.random(), probs[:, number_interactions_springautumn - 1])
        # Summer
        probs = np.loadtxt(directory_data+'/number_interactions_summer.txt', float)
        number_interactions_summer = get_probability(np.random.random(), probs[:, number_interactions_springautumn - 1])

        # Step 3. Set the occlusion rate and the probability functions/thresholds per key moment and room
        # Step 3a. Define the occlusion rate in function of the different drivers.
        # (1) 25% occlusion rate, (2) 50% occlusion rate, (3) 75% occlusion rate and (4) fully closed.
        probs = np.loadtxt(directory_data+'/occlusionrate.txt', float)
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
                
        # Import the lower and upper probability functions
        close_during_visual_irradiance_bounds = pd.read_csv(directory_data +'/close_during_visual_irradiance.txt', delimiter='\t', dtype=str, comment='#', header = None)
        close_entering_visual_irradiance_bounds = pd.read_csv(directory_data +'/close_entering_visual_irradiance.txt', delimiter='\t', dtype=str, comment='#', header = None)
        open_during_visual_irradiance_bounds = pd.read_csv(directory_data +'/open_during_visual_irradiance.txt', delimiter='\t', dtype=str, comment='#', header = None)
        open_leaving_visual_irradiance_bounds = pd.read_csv(directory_data +'/open_leaving_visual_irradiance.txt', delimiter='\t', dtype=str, comment='#', header = None)
        close_during_visual_glare_bounds = pd.read_csv(directory_data +'/close_during_visual_glare.txt', delimiter='\t', dtype=str, comment='#', header = None)
        close_entering_visual_glare_bounds = pd.read_csv(directory_data +'/close_entering_visual_glare.txt', delimiter='\t', dtype=str, comment='#', header = None)
        close_during_thermal_bounds = pd.read_csv(directory_data +'/close_during_thermal.txt', delimiter='\t', dtype=str, comment='#', header = None)
        close_entering_thermal_bounds = pd.read_csv(directory_data +'/close_entering_thermal.txt', delimiter='\t', dtype=str, comment='#', header = None)
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
        self.standard_deviation = np.loadtxt(directory_data+'/correction_deviation.txt')
        
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
        # Step 4e. Calculate timestep sunrise and sunset per day. 
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
        
        # Step 5. Import probabilities markov chains occlusion rates
        # Assumption: the solar shading is always reopened fully. 
        occlusion_thermal_open = np.loadtxt(directory_data+'\occlusion_thermal_open.txt', float)
        occlusion_thermal_close = np.loadtxt(directory_data+'\occlusion_thermal_close.txt', float)
        occlusion_thermal_anticipate_open = np.loadtxt(directory_data+'\occlusion_thermalanticipate_open.txt', float)
        occlusion_thermal_anticipate_close = np.loadtxt(directory_data+'\occlusion_thermalanticipate_close.txt', float)
        occlusion_visual_open = np.loadtxt(directory_data+'\occlusion_visual_open.txt', float)
        occlusion_visual_close = np.loadtxt(directory_data+'\occlusion_visual_close.txt', float)
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
        
        self.handles_set = False
        self.handle_previousstateshading = {}
        self.handle_previoustimestepshading = {}
        self.handle_irradiance = {}
        self.handle_indooroperativetemperature = {}
        self.handle_glare = {}
        self.handle_darkness = {}
        self.handle_privacy = {}
        self.handle_security = {}
        self.handle_scheduleshading = {}
        
    # CONTROL PROGRAM
    def on_begin_timestep_before_predictor(self, state) -> int:
        
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

        if not self.handles_set: 
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
                for orientation in range(1,number_orientations+1): 
                    window_orientation = self.overview_windows[zone_name][orientation]
                    self.handle_irradiance[zone_name+'.'+str(orientation)] = self.api.exchange.get_variable_handle(state, "Surface Outside Face Incident Solar Radiation Rate per Area", window_orientation)
                    self.handle_previousstateshading[zone_name+'.'+str(orientation)] = self.api.exchange.get_global_handle(state,'PreviousStateShading' + zone_name + '.' + str(orientation))
                    for schedule in [0.25, 0.5, 0.75, 1]:
                        self.handle_scheduleshading[zone_name+'.'+str(orientation)+'_'+str(schedule)] = self.api.exchange.get_actuator_handle(state, "Schedule:Constant","Schedule Value","ScheduleShadingControl" + zone_name+'.'+str(orientation)+'_'+str(schedule))
            self.handles_set = True

        hour = round(self.api.exchange.current_time(state), 2)
        day = self.api.exchange.day_of_year(state)
        timestep_simulation = self.api.exchange.zone_time_step(state)
        timestep = round((day - 1) * 720 + (hour / 0.033333) - 1)  # calculation for 2 minute timesteps
        sun_up = self.api.exchange.sun_is_up(state)
        
        previousstate_shading = {}
        irradiance = {}
        indoor_operative_temperature = {}
        glare = {}
        for zone_name in self.rooms_shading:
        
            # set the number of orientations of the windows in each room
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
                
            # Exchange input information with EnergyPlus
            indoor_operative_temperature['IndoorOperativeTemperature{}'.format(zone_name)] = self.api.exchange.get_variable_value(state, self.handle_indooroperativetemperature[zone_name])
            glare['Glare{}'.format(zone_name)] = self.api.exchange.get_variable_value(state, self.handle_glare[zone_name])
            
            action_practical_darkness = self.api.exchange.get_global_value(state, self.handle_darkness[zone_name])
            action_practical_privacy = self.api.exchange.get_global_value(state, self.handle_privacy[zone_name])
            action_practical_security = self.api.exchange.get_global_value(state, self.handle_security[zone_name])
            timestep_previous_shading = self.api.exchange.get_global_value(state, self.handle_previoustimestepshading[zone_name])
            for orientation in range(1,number_orientations+1): 
                irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))] = self.api.exchange.get_variable_value(state, self.handle_irradiance[zone_name+'.'+str(orientation)])
                previousstate_shading['PreviousStateShading{}'.format(zone_name+'.'+str(orientation))] = self.api.exchange.get_global_value(state, self.handle_previousstateshading[zone_name+'.'+str(orientation)])
                                                                                               
            # EVALUATION PRACTICAL REASONS (E.G. DARKNESS, PRIVACY, SECURITY)
            # Assumption: practical-driven shading interactions are executed for all window orientations simultaneously. 
            # Assumption: occlusion rate is expected to be constant for practical habits. 
            occlusion_timestep_practical = 0
            interaction_practical = 0
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
                interaction_psychological = 0
                action_psychological = None
                driver = None
                
                # Evalation whether interaction takes place
                if previousstate_shading['PreviousStateShading{}'.format(zone_name+'.'+str(orientation))] == 0: 
                    evaluation_actions = ['close']
                elif previousstate_shading['PreviousStateShading{}'.format(zone_name+'.'+str(orientation))] == 1: 
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
                                    k_irr = correcting_irradiance(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))])
                                    temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                    probability_timestep_thermal = horizontal_shift(self.probability_thermal_leaving[action][zone_name], temperature_n)(indoor_operative_temperature['IndoorOperativeTemperature{}'.format(zone_name)]) + ((k + k_irr)*self.standard_deviation)
                                # Opening is based on irradiance
                                elif action == 'open': 
                                    k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                    probability_timestep_thermal = generate_linear_function(self.probability_visual_leaving_irradiance[action][zone_name])(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))]) + (2*k*self.standard_deviation)
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
                                        k_irr = correcting_irradiance(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))])
                                        temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                        probability_timestep_thermalanticipate = horizontal_shift(self.probability_thermal_leaving[action][zone_name], temperature_n)(indoor_operative_temperature['IndoorOperativeTemperature{}'.format(zone_name)]) + (2/3*(k + k_irr + k_temp)*self.standard_deviation)
                                    elif action == 'open': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        probability_timestep_thermalanticipate = generate_linear_function(self.probability_visual_leaving_irradiance[action][zone_name])(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))]) + (2*k*self.standard_deviation)
                                    probability_timestep = max(probability_timestep, probability_timestep_thermalanticipate)
                                    if probability_timestep == probability_timestep_thermalanticipate: 
                                        driver = 'thermal_anticipate'
                            # Visual comfort 
                            if self.drivers_sec[room_habits][4] == 1:
                                # Irradiance
                                k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                probability_timestep_irradiance = generate_linear_function(self.probability_visual_leaving_irradiance[action][zone_name])(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))]) + (2*k*self.standard_deviation)
                                # Glare
                                # Assumption: glare is only evaluated to close the shading. 
                                if action == 'close': 
                                    k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                    k_irr = correcting_irradiance(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))])
                                    probability_timestep_glare = horizontal_shift(self.probability_visual_leaving_glare[action][zone_name], self.glare_evaluation[zone_name][timestep])(glare['Glare{}'.format(zone_name)]) + ((k+k_irr)*self.standard_deviation)
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
                                        k_irr = correcting_irradiance(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))])
                                        temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                        probability_timestep_thermal = horizontal_shift(self.probability_thermal_entering[action][zone_name], temperature_n)(indoor_operative_temperature['IndoorOperativeTemperature{}'.format(zone_name)]) + (k + k_irr)*self.standard_deviation
                                    # Opening is based on irradiance
                                    elif action == 'open': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        probability_timestep_thermal = generate_linear_function(self.probability_visual_entering_irradiance[action][zone_name])(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))]) +( 2*k*self.standard_deviation)
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
                                            k_irr = correcting_irradiance(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))])
                                            temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                            probability_timestep_thermalanticipate = horizontal_shift(self.probability_thermal_entering[action][zone_name], temperature_n)(indoor_operative_temperature['IndoorOperativeTemperature{}'.format(zone_name)]) +( 2/3*(k + k_irr + k_temp)*self.standard_deviation)
                                        # Opening is based on irradiance
                                        elif action == 'open': 
                                            k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                            probability_timestep_thermalanticipate = generate_linear_function(self.probability_visual_entering_irradiance[action][zone_name])(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))]) +( 2*k*self.standard_deviation)
                                        probability_timestep = max(probability_timestep, probability_timestep_thermalanticipate)
                                        if probability_timestep == probability_timestep_thermalanticipate: 
                                            driver = 'thermal_anticipate'
                                # Visual comfort 
                                if self.drivers_sec[room_habits][4] == 1:
                                    # Irradiance
                                    k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                    probability_timestep_irradiance = generate_linear_function(self.probability_visual_entering_irradiance[action][zone_name])(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))]) +( 2*k*self.standard_deviation)
                                    # Glare
                                    # Assumption: glare is only evaluated to close the shading. 
                                    if action == 'close': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        k_irr = correcting_irradiance(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))])
                                        probability_timestep_glare = horizontal_shift(self.probability_visual_entering_glare[action][zone_name], self.glare_evaluation[zone_name][timestep])(glare['Glare{}'.format(zone_name)]) +( (k+k_irr)*self.standard_deviation)
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
                                        k_irr = correcting_irradiance(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))])
                                        temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                        probability_timestep_thermal = horizontal_shift(self.probability_thermal_entering_occupation[action][zone_name], temperature_n)(indoor_operative_temperature['IndoorOperativeTemperature{}'.format(zone_name)]) +( (k + k_irr)*self.standard_deviation)
                                    # Opening is based on irradiance
                                    elif action == 'open': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        probability_timestep_thermal = generate_linear_function(self.probability_visual_entering_occupation_irradiance[action][zone_name])(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))]) +( 2*k*self.standard_deviation)
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
                                            k_irr = correcting_irradiance(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))])
                                            temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                            probability_timestep_thermalanticipate = horizontal_shift(self.probability_thermal_entering_occupation[action][zone_name], temperature_n)(indoor_operative_temperature['IndoorOperativeTemperature{}'.format(zone_name)]) +( 2/3*(k + k_irr + k_temp)*self.standard_deviation)
                                        # Opening is based on irradiance
                                        elif action == 'open': 
                                            k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                            probability_timestep_thermalanticipate = generate_linear_function(self.probability_visual_entering_occupation_irradiance[action][zone_name])(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))]) +( 2*k*self.standard_deviation)
                                        probability_timestep = max(probability_timestep, probability_timestep_thermalanticipate)
                                        if probability_timestep == probability_timestep_thermalanticipate: 
                                            driver = 'thermal_anticipate'
                                # Visual comfort 
                                if self.drivers_sec[room_habits][4] == 1:
                                    # Irradiance
                                    k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                    probability_timestep_irradiance = generate_linear_function(self.probability_visual_entering_occupation_irradiance[action][zone_name])(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))]) +( 2*k*self.standard_deviation)
                                    # Glare
                                    # Assumption: glare is only evaluated to close the shading. 
                                    if action == 'close': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        k_irr = correcting_irradiance(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))])
                                        probability_timestep_glare = horizontal_shift(self.probability_visual_entering_occupation_glare[action][zone_name], self.glare_evaluation[zone_name][timestep])(glare['Glare{}'.format(zone_name)]) +( (k+k_irr)*self.standard_deviation)
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
                                        k_irr = correcting_irradiance(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))])
                                        temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                        probability_timestep_thermal = horizontal_shift(self.probability_thermal_leaving_occupation[action][zone_name], temperature_n)(indoor_operative_temperature['IndoorOperativeTemperature{}'.format(zone_name)]) +( (k + k_irr)*self.standard_deviation)
                                    # Opening is based on irradiance
                                    elif action == 'open': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        probability_timestep_thermal = generate_linear_function(self.probability_visual_leaving_occupation_irradiance[action][zone_name])(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))]) +( 2*k*self.standard_deviation)
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
                                            k_irr = correcting_irradiance(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))])
                                            temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                            probability_timestep_thermalanticipate = horizontal_shift(self.probability_thermal_leaving_occupation[action][zone_name], temperature_n)(indoor_operative_temperature['IndoorOperativeTemperature{}'.format(zone_name)]) +( 2/3*(k + k_irr + k_temp)*self.standard_deviation)
                                        # Opening is based on irradiance
                                        elif action == 'open': 
                                            k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                            probability_timestep_thermalanticipate = generate_linear_function(self.probability_visual_leaving_occupation_irradiance[action][zone_name])(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))]) +( 2*k*self.standard_deviation)
                                        probability_timestep = max(probability_timestep, probability_timestep_thermalanticipate)
                                        if probability_timestep == probability_timestep_thermalanticipate: 
                                            driver = 'thermal_anticipate'
                                # Visual comfort 
                                if self.drivers_sec[room_habits][4] == 1:
                                    # Irradiance
                                    k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                    probability_timestep_irradiance = generate_linear_function(self.probability_visual_leaving_occupation_irradiance[action][zone_name])(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))]) +( 2*k*self.standard_deviation)
                                    # Glare
                                    # Assumption: glare is only evaluated to close the shading. 
                                    if action == 'close': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        k_irr = correcting_irradiance(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))])
                                        probability_timestep_glare = horizontal_shift(self.probability_visual_leaving_occupation_glare[action][zone_name], self.glare_evaluation[zone_name][timestep])(glare['Glare{}'.format(zone_name)]) +( (k+k_irr)*self.standard_deviation)
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
                                            k_irr = correcting_irradiance(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))])
                                            temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                            probability_timestep_thermal = horizontal_shift(self.probability_thermal_during[action][zone_name], temperature_n)(indoor_operative_temperature['IndoorOperativeTemperature{}'.format(zone_name)]) +( (k + k_irr)*self.standard_deviation)
                                        # Opening is based on irradiance
                                        elif action == 'open': 
                                            k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                            probability_timestep_thermal = generate_linear_function(self.probability_visual_during_irradiance[action][zone_name])(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))]) +( 2*k*self.standard_deviation)
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
                                                k_irr = correcting_irradiance(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))])
                                                temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                                probability_timestep_thermalanticipate = horizontal_shift(self.probability_thermal_during[action][zone_name], temperature_n)(indoor_operative_temperature['IndoorOperativeTemperature{}'.format(zone_name)]) +( 2/3*(k + k_irr + k_temp)*self.standard_deviation)
                                            # Opening is based on irradiance
                                            elif action == 'open': 
                                                k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                                probability_timestep_thermalanticipate = generate_linear_function(self.probability_visual_during_irradiance[action][zone_name])(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))]) +( 2*k*self.standard_deviation)
                                            probability_timestep = max(probability_timestep, probability_timestep_thermalanticipate)
                                            if probability_timestep == probability_timestep_thermalanticipate: 
                                                driver = 'thermal_anticipate'
                                    # Visual comfort 
                                    if self.drivers_sec[room_habits][4] == 1:
                                        # Irradiance
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        probability_timestep_irradiance = generate_linear_function(self.probability_visual_during_irradiance[action][zone_name])(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))]) +( 2*k*self.standard_deviation)
                                        # Glare
                                        # Assumption: glare is only evaluated to close the shading. 
                                        if action == 'close': 
                                            k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                            k_irr = correcting_irradiance(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))])
                                            probability_timestep_glare = horizontal_shift(self.probability_visual_during_glare[action][zone_name], self.glare_evaluation[zone_name][timestep])(glare['Glare{}'.format(zone_name)]) +( (k+k_irr)*self.standard_deviation)
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
                                        k_irr = correcting_irradiance(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))])
                                        temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                        probability_timestep_thermal = horizontal_shift(self.probability_thermal_during[action][zone_name], temperature_n)(indoor_operative_temperature['IndoorOperativeTemperature{}'.format(zone_name)]) +( (k + k_irr)*self.standard_deviation)
                                    # Opening is based on irradiance
                                    elif action == 'open': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        probability_timestep_thermal = generate_linear_function(self.probability_visual_during_irradiance[action][zone_name])(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))]) +( 2*k*self.standard_deviation)
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
                                            k_irr = correcting_irradiance(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))])
                                            temperature_n = neutral_indoor_temperature(zone_name, timestep, self.offices_list, self.bedrooms_list, self.reference_external_temperature[day-1])
                                            probability_timestep_thermalanticipate = horizontal_shift(self.probability_thermal_during[action][zone_name], temperature_n)(indoor_operative_temperature['IndoorOperativeTemperature{}'.format(zone_name)]) +( 2/3*(k + k_irr + k_temp)*self.standard_deviation)
                                        # Opening is based on irradiance
                                        elif action == 'open': 
                                            k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                            probability_timestep_thermalanticipate = generate_linear_function(self.probability_visual_during_irradiance[action][zone_name])(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))]) +( 2*k*self.standard_deviation)
                                        probability_timestep = max(probability_timestep, probability_timestep_thermalanticipate)
                                        if probability_timestep == probability_timestep_thermalanticipate: 
                                            driver = 'thermal_anticipate'
                                # Visual comfort 
                                if self.drivers_sec[room_habits][4] == 1:
                                    # Irradiance
                                    k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                    probability_timestep_irradiance = generate_linear_function(self.probability_visual_during_irradiance[action][zone_name])(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))]) +( 2*k*self.standard_deviation)
                                    # Glare
                                    # Assumption: glare is only evaluated to close the shading. 
                                    if action == 'close': 
                                        k = correcting_shading(zone_name, timestep, timestep_previous_shading, self.occupancy, self.asleep_bedroom)
                                        k_irr = correcting_irradiance(irradiance['Irradiance{}'.format(zone_name+'.'+str(orientation))])
                                        probability_timestep_glare = horizontal_shift(self.probability_visual_during_glare[action][zone_name], self.glare_evaluation[zone_name][timestep])(glare['Glare{}'.format(zone_name)]) +( (k+k_irr)*self.standard_deviation)
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
                    occlusion_timestep_psychological = previousstate_shading['PreviousStateShading{}'.format(zone_name+'.'+str(orientation))]
                elif interaction_psychological == 0 and interaction_practical == 1: 
                    occlusion_timestep_psychological = 0               
                # Stochastic evaluation of the change in occlusion rate
                else: 
                    previous_occlusion = int(previousstate_shading['PreviousStateShading{}'.format(zone_name+'.'+str(orientation))]/0.25)
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
                
                # Set the occlusion rate of the solar shading in the simulation and write it to a temporary variable. 
                if occlusion_timestep == 0: 
                    schedules_0 = [0.25, 0.5, 0.75, 1]
                    schedules_1 = []
                elif occlusion_timestep == 0.25: 
                    schedules_0 = [0.5, 0.75, 1]
                    schedules_1 = [0.25]
                elif occlusion_timestep == 0.5: 
                    schedules_0 = [0.75, 1]
                    schedules_1 = [0.25, 0.5]
                elif occlusion_timestep == 0.75: 
                    schedules_0 = [1]
                    schedules_1 = [0.25, 0.5, 0.75]
                elif occlusion_timestep == 1: 
                    schedules_0 = []
                    schedules_1 = [0.25, 0.5, 0.75, 1]
                    
               # Return the values to EnergyPlus, the windows are splitted and shading is separetely controlled per part. 
                for schedule in schedules_0: 
                    self.api.exchange.set_actuator_value(state, self.handle_scheduleshading[zone_name+'.'+str(orientation)+'_'+str(schedule)], 0)
                for schedule in schedules_1: 
                    self.api.exchange.set_actuator_value(state, self.handle_scheduleshading[zone_name+'.'+str(orientation)+'_'+str(schedule)], 1)
                self.api.exchange.set_global_value(state, self.handle_previousstateshading[zone_name+'.'+str(orientation)], occlusion_timestep)
                if interaction_practical == 1 or interaction_psychological == 1: 
                    timestep_previous_shading = timestep
                    
                    
            self.api.exchange.set_global_value(state, self.handle_darkness[zone_name], action_practical_darkness)
            self.api.exchange.set_global_value(state, self.handle_privacy[zone_name], action_practical_privacy)
            self.api.exchange.set_global_value(state, self.handle_security[zone_name], action_practical_security)
            self.api.exchange.set_global_value(state, self.handle_previoustimestepshading[zone_name], timestep_previous_shading)
        
        return 0