# Scheduling Optimization

This repository contains three different mathematical scheduling models and a script to visualize the results with a gantt chart. Everything is implemented with Pyomo.

## continuous time scheduling

### global event points

The model is based on the paper of Castro, Barbosa-Póvoa and Matos (2001): "An Improved RTN Continuous-Time Formulation 
for the Short-term Scheduling of Multipurpose Batch Plants" (DOI: 10.1021/ie000683r).

Syntax is oriented towards Mendez (2006): "State-of-the-art review of optimization methods for short-term scheduling 
of batch processes" (Doi: 10.1016/j.compchemeng.2006.02.008).

All parameters and variables are written as "name_index".

### unit-specific event points

The model is based on the paper of Ierapetritou and Floudas (1998): "Effective Continuous-Time Formulation for Short-Term  Scheduling. 1. Multipurpose Batch Processes" (Doi:[ 10.1021/ie970927g](https://pubs.acs.org/doi/abs/10.1021/ie970927g)).  

The paper is from a paper series. Further constraints are explained in Part 2 and 3 but not implemented here:  
 - Part 2: includes continuous processes  
 - Part 3: includes intermediate due dates  

Syntax is inspired by Mendez (2006): "State-of-the-art review of optimization methods for short-term scheduling  of batch processes" (Doi: [10.1016/j.compchemeng.2006.02.008](https://www.sciencedirect.com/science/article/abs/pii/S0098135406000287?via%3Dihub)).  

All parameters and variables are written as "name_index".

## discrete time scheduling

## Gantt chart
