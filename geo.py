"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  
%  Golden Eagle Optimizer (GEO) source codes version 1.0
%  
%  Developed in:	Python 3.8
%  
%  Implementers/    Gabriel Lois, Ramon Labbe,
%   Programmers:	Eduardo Zamorano, Jaime Olguin,
%                   Alfredo Escudero
%                   
%  
%  Original paper:	Abdolkarim Mohammadi-Balani, Mahmoud Dehghan Nayeri, 
%					Adel Azar, Mohammadreza Taghizadeh-Yazdi, 
%					Golden Eagle Optimizer: A nature-inspired 
%					metaheuristic algorithm, Computers & Industrial Engineering.
%
%                  https://doi.org/10.1016/j.cie.2020.107050               
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
#%%Libraries
import numpy as np

# %%Parameters
iterations = 1000
populationSize = 50
attackPropensityStart = 0.5
attackPropensityEnd = 2
cruisePropensityStart = 2
cruisePropensityEnd = 0.5
functionNumber = "F6"
nvars = 30
lowerLimit = -100
upperLimit = 100

# %% Functions Definition

#Return Euclidean Norm of a given vector (Equacion 7)
def getNormOfVector(vector):
    norm = np.sum(np.power(vector, 2))
    norm = pow(norm,0.5) 
    return norm

#Return Fitness Score for provided vector (variables)
def fitnessFunction(vector):
    if functionNumber.upper() == 'F6':
        return np.sum(np.power(vector, 2))
    elif functionNumber.upper() == 'F7':
        return ""
    elif functionNumber.upper() == 'F8':
        return ""
    elif functionNumber.upper() == 'F9':
        return ""
    else:
        return None

# %%Initialization

#Populate with first random solutions bounded to function's domain 
x = np.random.rand(populationSize,nvars)
x = lowerLimit + x * (upperLimit - lowerLimit)

#Initialize flockMemoryX with a copy of generated solutions
flockMemoryX = np.copy(x)

#Initialize flockMemoryF with evaluated solutions in fitness function
flockMemoryF = []
for solution in x:
     flockMemoryF.append(fitnessFunction(solution))

#Initialize Attack and Cruise Propensy
attackpropensy = 0
cruisepropensy = 0

#Initialize Array to store best solution for each iteration
ConvergenceCurve = []

#Main Loop
for i in range(iterations):
    #Update Attack and Cruise Propensy (Equation 9)
    attackpropensy = attackPropensityStart + (i/iterations) * abs((attackPropensityEnd - attackPropensityStart))
    cruisepropensy = cruisePropensityStart - (i/iterations) * abs((cruisePropensityEnd - cruisePropensityStart))

    #Initialize randomly one-to-one mapping between eagle and prey from population’s memory
    mapping = np.random.permutation(np.arange(populationSize))
    preySelection = np.empty((0,nvars))
    for k in mapping:
        preySelection = np.append(preySelection, np.array([flockMemoryX[k]]), axis=0)
    
    #Eagles loop
    for j in range(populationSize):
        eagle = x[j]
        prey = preySelection[j]

        #Get Attack Vector (Equation 1)
        attackvectorinitial = prey - eagle

        radius = getNormOfVector(attackvectorinitial)
        
        if radius != 0: 
            #Get scalar form of the hyperplane equation in n-dimensional (Equation 2)
            d = np.sum(attackvectorinitial*eagle)
            #Randomly choose index of one variable as a fixed variable (Variable does't has to be 0)
            idx = np.random.choice(np.nonzero(attackvectorinitial)[0])

            #Summation of attack vector except fixed variable
            attackvectorsummation = 0
            for index, item in enumerate(attackvectorinitial):
                if index != idx:
                    attackvectorsummation = attackvectorsummation + item
            #Find the value of the fixed variable (Equation 4)
            ck = (d-attackvectorsummation)/attackvectorinitial[idx]
            
            #Assign random values to all the variables except the k-th variable because the k-th variable is fixed (Equation 5)
            cruisevectordestination = 2 * np.random.rand(nvars) - 1; # [-1,1]
            cruisevectordestination[idx] = ck

            #Get Cruice Vector
            cruisevectorinitial = cruisevectordestination - eagle
            
            #Get attack and cruise unit vectors (Equation 7)
            AttackVectorUnit = attackvectorinitial/getNormOfVector(attackvectorinitial)
            CruiseVectorUnit = cruisevectorinitial/getNormOfVector(cruisevectorinitial)
            
            #Get final attack and cruise vector for Equation 6
            attackvector = np.random.rand() * attackpropensy * AttackVectorUnit * radius
            cruisevector = np.random.rand() * cruisepropensy * CruiseVectorUnit * radius
            
            #Get Step Vector (Equation 6)
            stepVector = attackvector + cruisevector
            
            #Move eagle to new position (Equation 8)
            eagle = eagle + stepVector
            
            #Fix out-of-bound variables towards the function limit (No feasible solution handling)
            for index, item in enumerate(eagle):
                if item > upperLimit:
                    eagle[index] = upperLimit
                if item < lowerLimit:
                    eagle[index] = lowerLimit
            
            #Evaluate fitness function for the new position
            FitnessScore = fitnessFunction(eagle)

            #Check if new position fitness score is better than the fitness score of the eagle in memory (flockMemory)
            if flockMemoryF[j] > FitnessScore:
                #Replace the new position in eagle memory (flockMemory)
                flockMemoryF[j] = FitnessScore
                flockMemoryX[j] = eagle
            x[j] = eagle
    ConvergenceCurve.append(np.min(flockMemoryF))
    
#%%Final Result
print(np.min(flockMemoryF))
print(flockMemoryX[np.argmin(flockMemoryF)])

import matplotlib.pyplot as plt
plt.plot(ConvergenceCurve)
plt.show()