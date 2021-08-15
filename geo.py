import numpy as np

#Parametros que debiesen ser recibidos
iterations = 1000
populationSize = 50
attackPropensityStart = 0.5
attackPropensityEnd = 2
cruisePropensityStart = 2
cruisePropensityEnd = 0.5
function = "No se ocupa aun"
nvars = 30
lowerLimit = -100
upperLimit = 100

#Funciones

#Devuelve la raiz cuadrada de la suma de los elementos elevados al cuadrado (Equacion 7)
def getModuloVector(vector):
    modulo = np.sum(np.power(vector, 2))
    modulo = pow(modulo,0.5) 
    return modulo

def fitnessfunction(function, vector):
    return np.sum(np.power(vector, 2))

#Inicializacion

#Generamos una matriz de aguilas x soluciones con soluciones random entre 0 y 1
x = np.random.rand(populationSize,nvars)

#Movemos los puntos a los limites del dominio
x = lowerLimit + x * (upperLimit - lowerLimit)

#Se evalua las soluciones en la funcion
flockMemoryX = np.copy(x)

#Se guarda las soluciones en un y su evaluacion en variables apartes
flockMemoryF = []
for var in x:
     flockMemoryF.append(fitnessfunction(var))

curvaConvergencia = []

#Main Loop
for i in range(iterations):
    #Se actualiza el attack y cruise propension (Ecuacion 9)
    attackpropensy = attackPropensityStart + (i/iterations) * abs((attackPropensityEnd - attackPropensityStart))
    cruisepropensy = cruisePropensityStart - (i/iterations) * abs((cruisePropensityEnd - cruisePropensityStart))

    #Se inicializa las aguilas y las presas(Una a una, sin repeticion)
    mapping = np.random.permutation(np.arange(populationSize))
    preySelection = np.empty((0,nvars))
    for k in mapping:
        preySelection = np.append(preySelection, np.array([flockMemoryX[k]]), axis=0)
    
    for j in range(populationSize):
        eagle = x[j]
        prey = preySelection[j]

        #Ecuacion 1
        attackvectorinitial = prey - eagle

        radio = getModuloVector(attackvectorinitial)
        
        if radio != 0: 
            #Se calcula el hiperplano (Ecuacion 2)
            d = np.sum(attackvectorinitial*eagle)
            #Se elige una columna al azar (que no contenga 0)
            idx = np.random.choice(np.nonzero(attackvectorinitial)[0])
            #Sumatoria de todos los elementos del vector de ataque excepto la columna fija
            sumatoriaattackvector = 0
            for index, item in enumerate(attackvectorinitial):
                if index != idx:
                    sumatoriaattackvector = sumatoriaattackvector + item
            #Se obtiene el valor de la columna fija (Ecuacion 4)
            ck = (d-sumatoriaattackvector)/attackvectorinitial[idx]
            
            #Genero el punto de destino del vector de crucero (Ecuacion 5)
            cruisevectordestination = 2 * np.random.rand(nvars) - 1; # [-1,1]
            cruisevectordestination[idx] = ck

            #Calculo el vector de crucero
            cruisevectorinitial = cruisevectordestination - eagle
            
            #Calculo de vectores unitarios (Ecuacion 7)
            AttackVectorUnit = attackvectorinitial/getModuloVector(attackvectorinitial)
            CruiseVectorUnit = cruisevectorinitial/getModuloVector(cruisevectorinitial)
            
            #Se calcula los vectores de ataque y crucero finales (Ecuacion 6)
            attackvector = np.random.rand() * attackpropensy * AttackVectorUnit * radio
            cruisevector = np.random.rand() * cruisepropensy * CruiseVectorUnit * radio
            
            #Se calcula el vector de paso
            stepVector = attackvector + cruisevector
            
            #Ecaucion 8
            eagle = eagle + stepVector
            
            #Reviso/Corrigo el dominio de las nuevas soluciones
            for index, item in enumerate(eagle):
                if item > upperLimit:
                    eagle[index] = upperLimit
                if item < lowerLimit:
                    eagle[index] = lowerLimit
            
            #Evaluar la nueva solucion en la funcion
            FitnessScore = fitnessfunction(eagle)

            #Comparo el nuevo resultado con el almacenado (Memoria de la bandada)
            if flockMemoryF[j] > FitnessScore:
                flockMemoryF[j] = FitnessScore
                flockMemoryX[j] = eagle
            x[j] = eagle
    curvaConvergencia.append(np.min(flockMemoryF))
    
#Final Result
print(np.min(flockMemoryF))
print(flockMemoryX[np.argmin(flockMemoryF)])

import matplotlib.pyplot as plt
plt.plot(curvaConvergencia)
plt.show()