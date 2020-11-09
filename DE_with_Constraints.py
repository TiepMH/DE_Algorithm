''' Author: Tiep M. Hoang 
This code performs the differential evolution (DE) algorithm 
to solve a constrained optimization problem. The problem is as follows:
    
    Find x = [x_1, x_2] to minimize f = x_1^2 + x_2^2
    subject to -100 <= x_1 <= 100, -30 <= x_2 <= 30,
                g = - (x_1^2 + x_2^2) + 5 <= 0 
                
In order to handle the constraint g, we convert the constrained problem into 
the unconstrained one by the following effective constraint-handling technique:

    F = f + mu * max(0,g)^2
    
If the constraint is satisfied, i.e., g <= 0, then F = f + mu * 0^2 = f
Otherwise, we have F > f
               
In general, the DE algorithm will try to find the optimal x_opt = [x_1opt, x_2opt]
so that F is as close to f as possible. 

References: 
    [1] Zou, Dexuan, et al. "A novel modified differential evolution algorithm 
    for constrained optimization problems." Computers & Mathematics with Applications 
    61.6 (2011): 1608-1623.
    [2] https://pablormier.github.io
'''

import numpy as np
import matplotlib.pyplot as plt

''' A function that yields OPT_x, OPT_f, OPT_g, OPT_penalty over iterations '''
def DE(bounds, mutation_factor=0.8, cross_prob=0.7, number_of_x=17, iterations=1000):
    dim_of_x = len(bounds) 
    an_ARRAY_of__x_NORMALIZED = np.random.rand(number_of_x, dim_of_x)
    min_b, max_b = np.asarray(bounds).T
    a_LIST_of__x = [ min_b + an_ARRAY_of__x_NORMALIZED[i]*np.abs(max_b - min_b) 
                     for i in range(number_of_x) ]
    
    ''' Find the optimal x that minimizes f '''
    a_LIST_of__f = np.array([function(x) for x in a_LIST_of__x])  
    a_LIST_of__g = np.array([LHS_of_constraint(x) for x in a_LIST_of__x]) 
    a_LIST_of__penalty = np.array([penalty_function(x) for x in a_LIST_of__x]) 
    OPT_index = np.argmin(a_LIST_of__penalty) 
    OPT_x = a_LIST_of__x[OPT_index]
    OPT_f = a_LIST_of__f[OPT_index]
    OPT_g = a_LIST_of__g[OPT_index]
    OPT_penalty = a_LIST_of__penalty[OPT_index]
    
    for i in range(iterations):
        for j in range(number_of_x):
            target_x = a_LIST_of__x[j] 
            ''' STEP 1: INITIALIZING THE FIRST GENERATION
            Assume that a_LIST_of__x[j] is the TARGET vector, 
            We're going to select a, b, c from a_LIST_of__f 
            so that they are not the target vector '''
            indices_ALL = [idx for idx in range(number_of_x) if idx != j]
            
            idx_of_a, idx_of_b, idx_of_c = np.random.choice(indices_ALL, 3, replace = False)
                                            # We randomly select 3 indices WITHOUT replacement 
            a = a_LIST_of__x[ idx_of_a ]
            b = a_LIST_of__x[ idx_of_b ]
            c = a_LIST_of__x[ idx_of_c ]
            
            ''' STEP 2: MUTATION '''
            mutant_x = a + mutation_factor*(b - c) # create a mutant vector by combining a, b and c
            mutant_x = np.clip( mutant_x, min_b, max_b ) # min_b <= mutant_x <= max_b
                                # For example, mutant_x = [x1, x2, x3] and min_b = [2, 10, 7]
                                # We restrict 2 <= x1, 10 <= x2 and 7 <= x3.
                                # So x = [220, 15, 8] satisfies the constraint on the lower bound
                 
            ''' STEP 3: RECOMBINATION (a.k.a., CROSSOVER) is all about mixing the information 
            of the mutant vector with the information of the current/target vector.
            This is done by exchanging some elements of mutant_x for some elements 
            of a_LIST_of__x[j] at some positions. At each position, we decide if the
            corresponding element will be replaced or not. The decision is made with
            a probability cross_prob. 
            *** In this step, we use the binomial crossover method '''
            crossover_TRUE_or_FALSE = np.random.rand(dim_of_x) < cross_prob
                                        # For example, [0.11, 0.22, 0.88] < 0.7 = 70%
                                        # Result = [True, True, False]
                                        # This means that at the first position,
                                        # there is an exchange of information.
                                        # At the second position, there is also an
                                        # exchange of informaton. However, there is
                                        # no exchange of information at the third position 
                
            trial_x = np.where(crossover_TRUE_or_FALSE, mutant_x, target_x)
                        # For example, crossover_TRUE_or_FALSE = [True, True, False]
                        # mutant_x = [x1, x2, x3] and a_LIST_of__x[j] = [y1, y2, y3]
                        # We will obtain trial_x = [y1, y2, x3]
            
            ''' STEP 4: SELECTION (REPLACEMENT)
            In this step, we select the next generation. 
            With trial_x, we can evaluate how good it is by comparing trial_x with target_x
            If trial_x is a better solution than target_x, we assign 
                                target_x = trial_x 
                                a_LIST_of__x[j] = target_x
            because the element j in a_LIST_of_x is the current target vector'''

            ##########
            penalty_at_xj = penalty_function(a_LIST_of__x[j])
            
            f_at_trial_x = function(trial_x)
            g_at_trial_x = LHS_of_constraint(trial_x)
            penalty_at_trial_x = penalty_function(trial_x)
            
            if (penalty_at_trial_x < penalty_at_xj)  :
                a_LIST_of__penalty[j] = penalty_at_trial_x
                a_LIST_of__f[j] = f_at_trial_x          
                a_LIST_of__g[j] = g_at_trial_x
                a_LIST_of__x[j] = trial_x # Do NOT write target_x = trial_x
                                          # we need to make change to a_LIST_of__x 
                ''' Compare the trial vector with the current best vector '''
                if (penalty_at_trial_x < OPT_penalty) : #########
                    OPT_index = j
                    OPT_x = trial_x
                    OPT_f = a_LIST_of__f[OPT_index]
                    OPT_g = a_LIST_of__g[OPT_index]
                    OPT_penalty = a_LIST_of__penalty[OPT_index]
                    
                    
        yield OPT_x, OPT_f, OPT_g, OPT_penalty

###############################################################################
def penalty_function(x, mu=1000):
    f_x = function(x)
    g_x = LHS_of_constraint(x)
    penalty_Fx = f_x + mu*(np.maximum(0.0, g_x)**2)
    return penalty_Fx

def LHS_of_constraint(x):
    temp = 0
    for i in range(len(x)):
      temp += x[i]**2
    LHS = - temp + 5 # constraint: - temp + 5 <= 0
    return LHS

###############################################################################
def function(x): # x must be a LIST or a row ARRAY
  value = 0
  for i in range(len(x)):
      value += x[i]**2
  return value / len(x)

results_over_iter = list( DE( bounds=[(-100, 100), (-30,30)],
                              iterations = 400 ) ) #return x_op and f_opt 

x_over_iter, f_over_iter, g_over_iter, penalty_over_iter = zip(*results_over_iter)

final_result = results_over_iter[-1]
print("x_opt = ", final_result[0])
print("f_opt = ", final_result[1])
print("g_opt = ", final_result[2])
print("penalty_opt = ", final_result[3])

###############################################################################
plt.figure(1)
plt.plot(f_over_iter, label='Objective function f(x)', color='b', marker=' ', markersize=8, markerfacecolor='none', linestyle='-', linewidth=1)
# plt.legend(loc='best', fontsize=12)

# plt.figure(2)
plt.plot(penalty_over_iter, label='Penalty function F(x)', color='r', marker=' ', markersize=8, markerfacecolor='none', linestyle=':', linewidth=2)
plt.legend(loc='best', fontsize=12)
#plt.ylim(0,20)

plt.figure(3)
plt.plot(g_over_iter, label='LHS of constraint, i.e., g(x)', color='k')
plt.legend(loc='best', fontsize=12)