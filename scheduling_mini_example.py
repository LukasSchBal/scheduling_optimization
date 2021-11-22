""" disclaimer
Model is based on the paper of Ierapetritou and Floudas (1998): "Effective Continuous-Time Formulation for Short-Term 
Scheduling. 1. Multipurpose Batch Processes" (Doi: 10.1021/ie970927g)

Paper is from paper series. Further constraints are explained in Part 2 and 3 but not implemented here:
Part 2: includes continuous processes
Part 3: includes intermediate due dates

Syntax is oriented towards Mendez (2006): "State-of-the-art review of optimization methods for short-term scheduling 
of batch processes" (Doi: 10.1016/j.compchemeng.2006.02.008). 

All parameters and variables are written as "name_index".
"""

from pyomo.environ import *

# states and their specification in matrix form: state, min storage, max storage, initial amount, price, market demand
#               S           STmin   STmax           ST0     P   D
resources = [   ['state 1', 0,      float('inf'),   1e4,    0,  0],
                ['state 2', 0,      100,            0,      0,  0],
                ['state 3', 0,      100,            0,      0,  0],
                ['state 4', 0,      float('inf'),   0,      1,  0]]

# production recipe in matrix form: state, task, rate of consumption, rate of prodction
#           S           I           Rhoc    Rhop 
recipe = [  ['state 1', 'task 1',   -1,     0],
            ['state 2', 'task 1',   0,      1],
            ['state 2', 'task 2',   -1,     0],
            ['state 3', 'task 2',   0,      1],
            ['state 3', 'task 3',   -1,     0],
            ['state 4', 'task 3',   0,      1]]

# tasks and their specification in matrix form:
#           J           I           Vmin    Vmax    alpha   beta    Mass flow
tasks = [   ['unit 1',  'task 1',   0,      100,    3,      0.03,   3],
            ['unit 2',  'task 2',   0,      75,     2,      0.0267, 3],
            ['unit 3',  'task 3',   0,      50,     1,      0.02,   3]]

## sets
N = {}                 # set of event points
S = set()              # set of states
I = set()              # set of tasks
I_s = {}               # set of tasks processing state s
I_j = {}               # set of tasks performed by unit j
J = set()              # set of units
J_i = {}               # set of units capable of task i

## parameters
H = {}                 # time horizon for production
ST_min_s = {}          # minimum storage capacity of state s
ST_max_s = {}          # maximum storage capacity of state s
ST_0_s = {}            # initial amount of state s
D_s = {}               # market demand for state s at end of time horizon
P_s = {}               # price (revenue) of state s
Rho_c_si = {}          # proportion of state s consumed from task i
Rho_p_si = {}          # proportion of state s produced in task i
V_min_ij = {}          # minimum capacity of unit j
V_max_ij = {}          # maximum capacity of unit j
Alpha_ij = {}          # constant term of processing time of task i at unit j
Beta_ij = {}           # variable term of processing time of task i at unit j

# parameter for STN input
param = {'number_event_points_STN':5, 'time_horizon':12}

N = [t for t in range(param['number_event_points_STN'])]     #set list of event points (should be defined by iterative procedure)
H = param['time_horizon']          # set time horizon

## read input matrix for resources
for row in resources:                       # for each state from the input...
    state = row[0]
    S.add(state)                       # add state s to set of states
    ST_min_s[state] = row[1]           # set minimum storage amount of state s
    ST_max_s[state] = row[2]           # set maximum storage amount of state s
    ST_0_s[state] = row[3]             # set initial (storage) amount of state s
    P_s[state] = row[4]                # set price of state s
    D_s[state] = row[5]                # set market demand of state s
    I_s[state] = set()                 # create set of tasks using the state s

## read input matrix for recipe
for row in recipe:                          # for each step of the recipe...
    state = row[0]
    task = row[1]
    if task not in I:                  # if task i not yet in set of tasks...
        I.add(task)                    # add task i to set of tasks
        J_i[task] = set()              # create set of units performing task i
    Rho_c_si[state, task] = row[2]     # set Rho_c 
    Rho_p_si[state, task] = row[3]     # set Rho_p 
    I_s[state].add(task)               # add task to set of tasks using the state s

# read input matrix for units and tasks
for row in tasks:                           # for each unit/task combination from the input...
    unit = row[0]
    task = row[1]
    if unit not in J:                  # if unit is not yet in set of units...
        J.add(unit)                    # add unit to set of units
        I_j[unit] = set()              # create set of tasks performed by unit j
    V_min_ij[task, unit] = row[2]      # set V_min
    V_max_ij[task, unit] = row[3]      # set V_max
    Alpha_ij[task, unit] = row[4]      # set alpha
    Beta_ij[task, unit] = row[5]       # set beta
    I_j[unit].add(task)                # add task i to set of tasks performed by unit j
    J_i[task].add(unit)                # add unit j to set of units performing task i




model = ConcreteModel()     # create model instance
model.cons = ConstraintList()       # create list of constraints

# set up variables

## w_in[i,n] 1 if task i is starting at event point n, else 0
model.w_in = Var(I, N, domain=Boolean, initialize=0 ) #, 

## w_jn[j,n] 1 if unit j is utilized at event point n, else 0
model.w_jn = Var(J, N, domain=Boolean, initialize=0 ) #, 

## b_ijn[i,j,n] batch/capacity of task i in unit j at event point n
model.b_ijn = Var(I, J, N, domain=NonNegativeReals, initialize=0 ) #, 

## st_sn[s,n] storage capacity of state s at event point n
model.st_sn = Var(S, N,  domain=NonNegativeReals) #,

## d_sn[s,n] amount of state s being delivered to the market at event point n (demand)
model.d_sn = Var(S, N, domain=NonNegativeReals, initialize=0 ) #, 

## tauf_ijn[i,j,n] finish time of task i at unit j at event point n
model.tauf_ijn = Var(I, J, N, domain=NonNegativeReals, initialize=0 ) #, 

## taus_ijn[i,j,n] start time of task i at unit j at event point n
model.taus_ijn = Var(I, J, N, domain=NonNegativeReals, initialize=0 ) #,

# set up constraints

## allocation constraints
# at each unit j max one task i can be performed at same event point n
for n in N:
    for j in J:
        model.cons.add(sum(model.w_in[i,n] for i in I_j[j]) == model.w_jn[j,n]) 

## capacity constraints
# requirement for the minimum amount V_min of material to start task i on unit j 
# and max amount V_max of material when performing task i on unit j
for n in N:
    for i in I:
        for j in J_i[i]:     
            model.cons.add((1+V_min_ij[i,j])*model.w_in[i,n] <= model.b_ijn[i,j,n])
            model.cons.add(model.b_ijn[i,j,n] <= V_max_ij[i,j]*model.w_in[i,n])
            # model.cons.add(1*model.w_in[i,n] <= model.b_ijn[i,j,n])
            # model.cons.add(model.b_ijn[i,j,n] <= V_max_ij[i,j]*model.w_in[i,n])

# storage constraints
# max available storage capacity for each state s at each event point n
for n in N:
    for s in S:
        model.cons.add(model.st_sn[s,n] <= ST_max_s[s])
        model.cons.add(ST_min_s[s] <= model.st_sn[s,n])

# mass balance
# stored mass of current event point st_sn[s,n] equals stored mass of previous time step st_sn[s,n-1] adjusted by market demand d_sn[s,n], 
# material produced in previous event point Rho_p_si*b_ijn[n-1] and material consumed in this event point Rho_c_si*b_ijn[n]
for n in N[1:]:
    for s in S:
        model.cons.add(model.st_sn[s,n] == model.st_sn[s,n-1]-model.d_sn[s,n]+sum(Rho_p_si[s,i]*sum(model.b_ijn[i,j,n-1] for j in J_i[i]) for i in I_s[s])\
            +sum(Rho_c_si[s,i]*sum(model.b_ijn[i,j,n] for j in J_i[i]) for i in I_s[s]))

# state mass balance for initial time step
for s in S:
    model.cons.add(model.st_sn[s,0] == ST_0_s[s]-model.d_sn[s,0]+sum(Rho_c_si[(s,i)]*sum(model.b_ijn[i,j,0] for j in J_i[i]) for i in I_s[s]))
    
# demand constraints
# requirement to produce at least as much as required by the market
for s in S:
    model.cons.add(sum(model.d_sn[s,n] for n in N) >= D_s[s])

# duration constraints
# duration constraints express the dependence of the time duratin of task i at unit j at event point n on the material being processed
for n in N:
    for i in I:
        for j in J_i[i]:     #set of units able to do task i
            model.cons.add(model.tauf_ijn[i,j,n] == model.taus_ijn[i,j,n]+Alpha_ij[i,j]*model.w_in[i,n]+Beta_ij[i,j]*model.b_ijn[i,j,n])

# sequence constraints
# same task in the same unit: task i starting at n+1 should start after task i ended at n
for n in N[0:len(N)-1]:       
    for i in I:
        for j in J_i[i]:
            model.cons.add(model.taus_ijn[i,j,n+1] >= model.tauf_ijn[i,j,n]-H*(2-model.w_in[i,n]-model.w_jn[j,n]))
            model.cons.add(model.taus_ijn[i,j,n+1] >= model.taus_ijn[i,j,n])
            model.cons.add(model.tauf_ijn[i,j,n+1] >= model.tauf_ijn[i,j,n])

# different tasks in the same unit
for n in N[0:len(N)-1]:       
    for j in J:
        for i in I_j[j]:
            for i_ in I_j[j]:
                if i != i_:
                    model.cons.add(model.taus_ijn[i,j,n+1] >= model.tauf_ijn[i_,j,n]-H*(2-model.w_in[i_,n]-model.w_jn[j,n]))

# different tasks in different units
for n in N[0:len(N)-1]:       
    for j in J:
        for j_ in J:
            for i in I_j[j]:
                for i_ in I_j[j_]:
                    if i != i_:
                        model.cons.add(model.taus_ijn[i,j,n+1] >= model.tauf_ijn[i_,j_,n]-H*(2-model.w_in[i_,n]-model.w_jn[j_,n]))

# completion of previous tasks
for n in N[0:len(N)-1]:       
    for i in I:
        for j in J_i[i]:
            model.cons.add(model.taus_ijn[i,j,n+1] >= sum(sum(model.tauf_ijn[i_,j,n_]-model.taus_ijn[i_,j,n_] for i_ in I_j[j]) for n_ in N[0:n+1]))                   

# time horizon constraints
# every task i starts and end within the time horizon H
for n in N:
    for i in I:
        for j in J_i[i]:     #set of units able to do task i
            model.cons.add(model.tauf_ijn[i,j,n] <= H)
            model.cons.add(model.taus_ijn[i,j,n] <= H)


# set up objective function
model.obj_fun = Objective(expr = sum(P_s[s]*(sum(model.d_sn[s,n] for n in N)+model.st_sn[s,N[-1]]) for s in S), sense = maximize)     # set the objective of the problem

## solve model and get objective value
solver = SolverFactory('gurobi')             # choose solver
solver.solve(model, tee=True).write() # solve problem