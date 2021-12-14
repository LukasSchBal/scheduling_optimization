""" disclaimer
Model is based on the paper of Castro, Barbosa-Póvoa and Matos (2001): "An Improved RTN Continuous-Time Formulation 
for the Short-term Scheduling of Multipurpose Batch Plants" (DOI: 10.1021/ie000683r) 

Syntax is oriented towards Mendez (2006): "State-of-the-art review of optimization methods for short-term scheduling 
of batch processes" (Doi: 10.1016/j.compchemeng.2006.02.008) 
"""


import pyomo.environ as pyo
import matplotlib.pyplot as plt

## input data

# resources and their specification in matrix form: resource, Rmin, Rmax, R0, price (for unit resources R/Rmin/Rmax/R0 have to be integer)
#               R           RminRmax            R0              P
resources = [   ['state 1', 0,  float('inf'),   1e4,            0],
                ['state 2', 0,  100,            0,              0],
                ['state 3', 0,  100,            0,              0],
                ['product', 0,  float('inf'),   0,              1],
                ['unit 1',  0,  1,              1,              0],
                ['unit 2',  0,  1,              1,              0],
                ['unit 3',  0,  1,              1,              0]]

# production recipe in matrix form: resource, task, Mu_c (=0 for states), Mu_p (=0 for states), Nu_c (=0 for units), Nu_p(=0 for units)
#           R           I           Mu_cMu_pNu_cNu_p     
recipe = [  ['state 1', 'task 1',   0,  0,  -1, 0],
            ['unit 1',  'task 1',   -1, 1,  0,  0],   
            ['state 2', 'task 1',   0,  0,  0,  1],
            ['state 2', 'task 2',   0,  0,  -1, 0],
            ['state 3', 'task 2',   0,  0,  0,  1],
            ['unit 2',  'task 2',   -1, 1,  0,  0],
            ['state 3', 'task 3',   0,  0,  -1, 0],
            ['product', 'task 3',   0,  0,  0,  1],
            ['unit 3',  'task 3',   -1, 1,  0,  0]]

# tasks and their specification in matrix form: 
#           I           Vmin    Vmax    alpha   beta
tasks = [   ['task 1',  5,      100,    3,      0.03],
            ['task 2',  5,      75,     2,      0.0267],
            ['task 3',  5,      50,     1,      0.02]]

## general input parameters
param_scheduling={'number_event_points':5, 'time_horizon':12, 'scale_st':1, 'scale_b':1, 'scale_valueRTN':1}

# sets
N = {}             # set of event points
R = set()          # set of resources
I = set()          # set of tasks
I_r = {}           # set of tasks performed at resource r

# parameters
H = {}             # time horizon
ST_min_r = {}      # Minimum availability of resource 𝑟 (states: storage, units: availability)
ST_max_r = {}      # Maximum availability of resource 𝑟 (states: storage, units: availability)
ST_0_r = {}        # initial amount of resource r
P_r = {}         # Value of resource 𝑟 (negative for costs)
Mu_c_ri = {}        # Coefficient for the binary extent of resource 𝑟 at the start of task 𝑖
Mu_p_ri ={}        # Coefficient for the binary extent of resource 𝑟 at the end of task 𝑖
Nu_c_ri = {}        # Coefficient for the continuous extent of resource 𝑟 at the start of task 𝑖
Nu_p_ri = {}       # Coefficient for the continuous extent of resource 𝑟 at the end of task 𝑖
V_min_i = {}      # Minimum amount of material processed by task 𝑖
V_max_i = {}      # Maximum amount of material processed by task 𝑖
Alpha_i = {}     # constant term of processing time of task i
Beta_i = {}      # variable term of processing time of task i (time requiered to process one unit of material)

H = param_scheduling['time_horizon']      # set time horizon
N = [t for t in range(param_scheduling['number_event_points'])]    # set list with event points

## read input

# read input matrix for resources
for row in resources:                       # for each resource from the input...
    resource = row[0]
    R.add(resource)                    # add resource r to set of resources
    ST_min_r[resource] = row[1]        # set minimum amount of resource r
    ST_max_r[resource] = row[2]        # set maximum amount of resource r
    ST_0_r[resource] = row[3]          # set initial amount of resource r
    P_r[resource] = row[4]             # set price of resource r
    I_r[resource] = set()              # create set of tasks performed with resource r

# read input matrix for recipe
for row in recipe:                          # for each step of the recipe...
    resource = row[0]
    task = row[1]
    if task not in I:                  # if task i not yet in set of tasks...
        I.add(task)                    # add task i to set of tasks
    Mu_c_ri[resource, task] = row[2]   # setMu_c_ri 
    Mu_p_ri[resource, task] = row[3]   # set Mu_p_ri 
    Nu_c_ri[resource, task] = row[4]   # set nu
    Nu_p_ri[resource, task] = row[5]   # set nu_
    I_r[resource].add(task)            # add task i to set of tasks performed by resource r

# read input matrix for tasks
for row in tasks:                           # for each task from the input...
    task = row[0]
    V_min_i[task] = row[1]             # set V_min
    V_max_i[task] = row[2]             # set V_max
    Alpha_i[task] = row[3]             # set alpha
    Beta_i[task] = row[4]              # set beta

## scale parameters
scale_st = param_scheduling['scale_st']           # scale state - /tonne: 1000
scale_b = param_scheduling['scale_b']             # scale batch size - /100 kg: 100
scale_valueRTN = param_scheduling['scale_valueRTN'] # scale revenues from production - /1000 €: 1000

model = pyo.ConcreteModel()
model.cons = pyo.ConstraintList()

## define variables

# Binary variable that assigns the beginning of task 𝑖 to event point 𝑡
model.w_in = pyo.Var(I, N, domain=pyo.Boolean)

# Binary variable that assigns the end of task 𝑖, which began at time t, to point t‘
model.w_inn = pyo.Var(I, N, N, domain=pyo.Boolean)

# Batch size: Total amount of material processed by the instance of task 𝑖 starting at event point 𝑡
model.b_in = pyo.Var(I, N, domain=pyo.NonNegativeReals, bounds=(0,max(V_max_i[i] for i in I)))

# Batch size: Total amount of material processed by the instance of task 𝑖 starting at event point 𝑡 and finishing at event point 𝑡′
model.b_inn = pyo.Var(I, N, N, domain=pyo.NonNegativeReals, bounds=(0,max(V_max_i[i] for i in I)))

# Excess amount of resource 𝑟 at event point 𝑡
model.st_rn = pyo.Var(R, N, domain=pyo.NonNegativeReals, bounds=(min(ST_min_r[r] for r in R), max(ST_max_r[r] for r in R)))

# Absolute time of event point 𝑡
model.tau_n = pyo.Var(N, domain=pyo.NonNegativeReals, bounds=(0,H))

## define constraints

# relaxed timing constraint
# if a task starts at event point t and ends at event point t_, its processing time must be equal or smaller than the difference between the event points
for i in I:
    for n in N:
        for n_ in N[n+1:]:
                model.cons.add(model.tau_n[n_]-model.tau_n[n] >= Alpha_i[i]*model.w_inn[i,n,n_]+Beta_i[i]*model.b_inn[i,n,n_]*scale_b)
model.cons.add(model.tau_n[N[-1]] <= H)

# slot allocation constraints
# if a task begins at some event point t, it must certainly end at some point t_ > t...
for i in I:
    for n in N[:-1]:
        model.cons.add(model.w_in[i,n] == sum(model.w_inn[i,n,n_] for n_ in N[n+1:]))
# ... while still processing the same amount of material
        model.cons.add(model.b_in[i,n]*scale_b == sum(model.b_inn[i,n,n_]*scale_b for n_ in N[n+1:]))
    model.cons.add(model.w_in[i,N[-1]] == 0)
    model.cons.add(model.b_in[i,N[-1]]*scale_b == 0)

# operational constraints
# if a processing task occurs, then the amount of material being processed must lie within the range between V_min and V_max  
for i in I:
    for n in N[:-1]:
        model.cons.add(V_min_i[i]*model.w_in[i,n] <= model.b_in[i,n]*scale_b)
        model.cons.add(model.b_in[i,n]*scale_b <= V_max_i[i]*model.w_in[i,n])
        for n_ in N[n+1:]:
                model.cons.add(V_min_i[i]*model.w_inn[i,n,n_] <= model.b_inn[i,n,n_]*scale_b)
                model.cons.add(model.b_inn[i,n,n_]*scale_b <= V_max_i[i]*model.w_inn[i,n,n_])

# excess resource balance constraints (states: mass balance)
# the excess amount of a resource at event point t is equal to that at the previous event point t-1 adjusted by the amount 
# consumed by all tasks starting at event point t and by the amount produced by all tasks ending at this point
for r in R:
    for n in N[1:]:
        model.cons.add(model.st_rn[r,n]*scale_st == model.st_rn[r,n-1]*scale_st+sum(Mu_c_ri[r,i]*model.w_in[i,n]+Nu_c_ri[r,i]*model.b_in[i,n]*scale_b for i in I_r[r])+sum(sum(\
            Mu_p_ri[r,i]*model.w_inn[i,n_,n]+Nu_p_ri[r,i]*model.b_inn[i,n_,n]*scale_b for n_ in N[:n]) for i in I_r[r]))
    model.cons.add(model.st_rn[r,0]*scale_st == ST_0_r[r]+sum(Mu_c_ri[r,i]*model.w_in[i,0]+Nu_c_ri[r,i]*model.b_in[i,0]*scale_b for i in I_r[r]))

# capacity constraints
# the excess amount of each resource must lie within given upper and lower bounds for storage (ST_max_r = 0 leads to ni intermediate storage (NIS) constraints)
for r in R:
    for n in N:
        model.cons.add(ST_min_r[r] <= model.st_rn[r,n]*scale_st)
        model.cons.add(model.st_rn[r,n]*scale_st <= ST_max_r[r])

## objective function

# max profit
model.value_RTN = pyo.Var(domain=pyo.NonNegativeReals)
model.cons.add(model.value_RTN*scale_valueRTN == sum(model.st_rn[r, N[-1]]*scale_st*P_r[r] for r in R)) 
model.obj_fun = pyo.Objective(expr = model.value_RTN, sense = pyo.maximize)

solver = pyo.SolverFactory('gurobi')             # choose solver
# model.display()                            # display results
results = solver.solve(model, options={'MaxTime':3600,'EpsR':1e-8, 'threads':6}, tee=True, logfile="baron_logfile.log").write(filename="myresults.txt")


## gantt chart

gap = H/400
idx = 1
lbls = []   # labels
ticks = []

# determine which resources are units
J = set()
for r in R:
    if all((Nu_c_ri[r,i] == 0 and Nu_p_ri[r,i] == 0) for i in I_r[r]):
        J.add(r)

# create a list of units sorted by time of first assignment
jstart = {j:H+1 for j in J}
for j in J:
    for i in I_r[j]:
        for n in N:
            if model.w_in[i,n]() > 0:
                jstart[j] = min(jstart[j],n)
jsorted = [j for (j,n) in sorted(jstart.items(), key=lambda x: x[1])]


# number of horizontal bars to draw
nbars = -1
for j in jsorted:
    for i in sorted(I_r[j]):
        nbars += 1
    nbars += 0.5
plt.figure(figsize=(H+1,(nbars+1)/2+0.5)) # sets width and height of figure in inches

for j in jsorted:   # for every unit ...
    idx -= 0.5
    for i in sorted(I_r[j]):  # for every task ...
        idx -= 1
        ticks.append(idx)
        lbls.append("{0:s} -> {1:s}".format(j,i))
        plt.plot([0,H],[idx,idx],lw=24,alpha=.3,color='y')
        for n1 in N[:-1]:
            for n2 in N[n1+1:]:
                if model.w_inn[i,n1,n2]() > 0:
                    tau_start = model.tau_n[n1]()
                    tau_end = model.tau_n[n2]()
                    plt.plot([tau_start,tau_end], [idx,idx],'k', lw=24, alpha=0.5, solid_capstyle='butt')
                    plt.plot([tau_start+gap,tau_end-gap], [idx,idx],'b', lw=20, solid_capstyle='butt')
                    txt = "{0:.2f}".format(model.b_inn[i,n1,n2]())
                    plt.text(tau_start+(tau_end-tau_start)/2, idx, txt, color='white', weight='bold', ha='center', va='center')
plt.xlim(0,H)
plt.ylim(-nbars-0.5,0)
plt.gca().set_yticks(ticks)
plt.gca().set_yticklabels(lbls)

plt.show()