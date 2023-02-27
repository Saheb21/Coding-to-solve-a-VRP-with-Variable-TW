# Coding-to-solve-a-VRP-with-Variable-TW
Coding to solve a VRP with Variable TW

Current Code (Using Colab)
------------------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Read input data from Excel sheet
df = pd.read_excel('Input Sheet for Code.xlsx') #sheet_name='Sheet1')

# Extract coordinates, product requirements, and facility capacities
coords = df[['Latitude', 'Longitude']].values.tolist()
product_reqs = df['Daily Product Requirement'].values
facility_caps = df['Facility Capacity'].values

# Extract delivery time windows
fixed_delivery_times = []
adjustable_delivery_times = []
for i in range(len(df)):
    if np.isnan(df.loc[i, 'Earliest Delivery Time']):
        adjustable_delivery_times.append((df.loc[i, 'Latest Delivery Time'], i))
    else:
        fixed_delivery_times.append((df.loc[i, 'Earliest Delivery Time'], df.loc[i, 'Latest Delivery Time'], i))

# Calculate distance matrix
dist_matrix = cdist(coords, coords, metric='euclidean')

# Check the data type of the coords array
print(type(coords))

# Convert any non-float values in the coords array to float
for i in range(len(coords)):
    for j in range(len(coords[i])):
        if not isinstance(coords[i][j], float):
            coords[i][j] = float(coords[i][j])

def distance_callback(from_node, to_node):
    return int(dist_matrix[from_node][to_node])

# Set up the OR-Tools routing model
routing_index_manager = pywrapcp.RoutingIndexManager(len(coords), 1, 0)
routing_model_parameters = pywrapcp.DefaultRoutingModelParameters()
routing_model_parameters.use_light_propagation = True
routing_model_parameters.reduce_vehicle_cost_model = True
routing = pywrapcp.RoutingModel(routing_index_manager, routing_model_parameters)
routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)

# Set up capacity constraints
demand_callback = lambda from_node, to_node: int(product_reqs[from_node])
routing.AddDimensionWithVehicleCapacity(demand_callback, 0, facility_caps, True, 'Capacity')

# Set up time window constraints
time_callback = lambda from_node, to_node, departure_time: (int(fixed_delivery_times[to_node][0]), int(fixed_delivery_times[to_node][1]))
routing.AddDimensionWithVehicleTransits(time_callback, 0, 1440, True, 'Time')
for i in range(len(adjustable_delivery_times)):
    routing.AddDisjunction([len(coords) + i], int(adjustable_delivery_times[i][0]))

# Solve the VRPTW with OR-Tools
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
solution = routing.SolveWithParameters(search_parameters)

# Extract the delivery groups and times
groups = []
for vehicle_id in range(routing.vehicles()):
    group = []
    index = routing.Start(vehicle_id)
    while not routing.IsEnd(index):
        node = routing.IndexToNode(index)
        if node < len(coords):
            group.append(node)
        index = solution.Value(routing.NextVar(index))
    groups.append(group)
group_delivery_times = []
for group in groups:
    delivery_times = []
    for i in range(len(group)):
        node = group[i]
        if node in [x[2] for x in fixed_delivery_times]:
            delivery_time = (fixed_delivery_times[[x[2] for x in fixed_delivery_times].index(node)][0],)
        else:
            delivery_time = (solution.Value(routing.CumulVar(routing.NodeToIndex(node), 'Time')),)
        delivery_times.append(delivery_time)
    group_delivery_times.append(delivery_times)

# Create a new Excel sheet for output data
output_df = pd.DataFrame(columns=['Group', 'Stores', 'Delivery Times'])
for i in range(len(groups)):
    group_num = i+1
    store_nums = ', '.join([str(x+1) for x in groups[i]])
    delivery_times = ', '.join([str(x[0]) for x in group_delivery_times[i]])
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
Here is the error I face while running it
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-3-46c7adc66e23> in <module>
     40 routing_index_manager = pywrapcp.RoutingIndexManager(len(coords), 1, 0)
     41 routing_model_parameters = pywrapcp.DefaultRoutingModelParameters()
---> 42 routing_model_parameters.use_light_propagation = True
     43 routing_model_parameters.reduce_vehicle_cost_model = True
     44 routing = pywrapcp.RoutingModel(routing_index_manager, routing_model_parameters)

AttributeError: Protocol message RoutingModelParameters has no "use_light_propagation" field.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
