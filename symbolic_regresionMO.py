import random
import operator
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from deap import base
from deap import creator
from deap import tools
from deap import gp

from functools import partial

import elitism


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

CSV_FILE = "datos.csv"

# Genetic Algorithm constants
POPULATION_SIZE = 70
P_CROSSOVER = 0.5
P_MUTATION = 0.9
MAX_GENERATIONS = 100
PARETO_FRONT_SIZE = 50

# Genetic Programming constants
MIN_TREE_HEIGHT = 2
MAX_TREE_HEIGHT = 5

LIMIT_TREE_HEIGHT = 17

MUT_MIN_TREE_HEIGHT = 0
MUT_MAX_TREE_HEIGHT = 2

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ==========================================================
# CARGA DEL DATASET
# ==========================================================

data = pd.read_csv(CSV_FILE)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

NUM_INPUTS = X.shape[1]

print(f"Dataset cargado:")
print(f"Muestras: {X.shape[0]}")
print(f"Variables independientes: {NUM_INPUTS}")


# ==========================================================
# FUNCIONES PROTEGIDAS
# ==========================================================

def protectedDiv(left, right):
    """División protegida."""
    try:
        if abs(right) < 1e-10:
            return 1.0
        return left / right
    except Exception:
        return 1.0


def protectedLog(x):
    """Logaritmo protegido."""
    try:
        return math.log(abs(x) + 1e-10)
    except Exception:
        return 0.0


def protectedSqrt(x):
    """Raíz cuadrada protegida."""
    try:
        return math.sqrt(abs(x))
    except Exception:
        return 0.0


def protectedExp(x):
    """Exponencial protegida."""
    try:
        x = np.clip(x, -20, 20)
        return math.exp(x)
    except Exception:
        return 0.0


# ==========================================================
# TOOLBOX
# ==========================================================

toolbox = base.Toolbox()


# ==========================================================
# PRIMITIVE SET
# ==========================================================

primitiveSet = gp.PrimitiveSet("MAIN", NUM_INPUTS)

# Renombrar variables
for i in range(NUM_INPUTS):
    primitiveSet.renameArguments(**{f"ARG{i}": f"x{i+1}"})


# Operadores aritméticos
primitiveSet.addPrimitive(operator.add, 2)
primitiveSet.addPrimitive(operator.sub, 2)
primitiveSet.addPrimitive(operator.mul, 2)
primitiveSet.addPrimitive(protectedDiv, 2)

# Operadores matemáticos
primitiveSet.addPrimitive(np.sin, 1)
primitiveSet.addPrimitive(np.cos, 1)
primitiveSet.addPrimitive(np.tanh, 1)

primitiveSet.addPrimitive(protectedLog, 1)
primitiveSet.addPrimitive(protectedSqrt, 1)
primitiveSet.addPrimitive(protectedExp, 1)

# Constantes aleatorias
primitiveSet.addEphemeralConstant(
    "rand",
#    lambda: random.uniform(-10.0, 10.0)
    partial(random.uniform, -10.0, 10.0)
)


# ==========================================================
# DEFINICIÓN DEL PROBLEMA
# ==========================================================

# Evitar recrear clases si se ejecuta varias veces
if "FitnessMulti" not in creator.__dict__:
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0,-1.0))

if "Individual" not in creator.__dict__:
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti    )

# ==========================================================
# CREACIÓN DE INDIVIDUOS
# ==========================================================

toolbox.register("expr", gp.genHalfAndHalf, pset=primitiveSet, min_=MIN_TREE_HEIGHT, max_=MAX_TREE_HEIGHT)
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
toolbox.register("compile", gp.compile, pset=primitiveSet)

# ==========================================================
# FITNESS
# ==========================================================

def regressionError(individual):
    """
    Error cuadrático medio.
    """
    func = toolbox.compile(expr=individual)
    predictions = []

    for row in X:
        try:
            pred = func(*row)
            if np.isnan(pred) or np.isinf(pred):
                pred = 1e6
        except Exception:
            pred = 1e6
        predictions.append(pred)
    predictions = np.asarray(predictions)
    mse = np.mean((predictions - y) ** 2)

    return mse


def evaluateMultiObjective(individual):
    """
    Fitness:
    error + penalización por complejidad
    """
    mse = regressionError(individual)
    complexity = len(individual)

    return mse, complexity


toolbox.register("evaluate", evaluateMultiObjective)


# ==========================================================
# OPERADORES GENÉTICOS
# ==========================================================

toolbox.register("select", tools.selSPEA2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=MUT_MIN_TREE_HEIGHT, max_=MUT_MAX_TREE_HEIGHT)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=primitiveSet)


# ==========================================================
# CONTROL DE BLOAT
# ==========================================================

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=LIMIT_TREE_HEIGHT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=LIMIT_TREE_HEIGHT))

# ==========================================================
# FUNCIÓN PRINCIPAL
# ==========================================================

def main():

    # ------------------------------------------------------
    # Población inicial
    # ------------------------------------------------------

    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # ------------------------------------------------------
    # Estadísticas
    # ------------------------------------------------------

    stats_mse = tools.Statistics( lambda ind: ind.fitness.values[0])
    stats_mse.register("min", np.min)
    stats_mse.register("avg", np.mean)
    
    stats_complexity = tools.Statistics(lambda ind: ind.fitness.values[1])
    stats_complexity.register("min", np.min)
    stats_complexity.register("avg", np.mean)
    
    mstats = tools.MultiStatistics(mse=stats_mse, complexity=stats_complexity)

    # ------------------------------------------------------
    # Hall of Fame
    # ------------------------------------------------------

    #hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    hof = tools.ParetoFront()

    # ------------------------------------------------------
    # Evolución
    # ------------------------------------------------------

    population, logbook = elitism.eaSimpleWithElitism(
        population,
        toolbox,
        cxpb=P_CROSSOVER,
        mutpb=P_MUTATION,
        ngen=MAX_GENERATIONS,
        stats=mstats,
        halloffame=hof,
        verbose=True
    )

    # ------------------------------------------------------
    # Mejor individuo
    # ------------------------------------------------------

    print("SOLUCIONES PARETO")
    
    for i, ind in enumerate(hof):
        mse = ind.fitness.values[0]
        complexity = ind.fitness.values[1]
        print(f"{i+1}: MSE={mse:.4f}, " f"Complexity={complexity}")
        print("Expresión simbólica:")
        print(ind)

    # ------------------------------------------------------
    # Graficar el frente de Pareto
    # ------------------------------------------------------
    
    pareto_mse = [ind.fitness.values[0] for ind in hof ]
    
    pareto_complexity = [ ind.fitness.values[1] for ind in hof ]
    
    plt.figure(figsize=(8,6))
    plt.scatter(pareto_complexity, pareto_mse)
    plt.xlabel("Tree Size")
    plt.ylabel("MSE")
    plt.title("Pareto Front")
    plt.show()

    return population, logbook, hof


# ==========================================================
# EJECUCIÓN
# ==========================================================

if __name__ == "__main__":
    main()
