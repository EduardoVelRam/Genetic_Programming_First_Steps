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
POPULATION_SIZE = 250
P_CROSSOVER = 0.9
P_MUTATION = 0.3
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 10

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
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

if "Individual" not in creator.__dict__:
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin    )

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


def getCost(individual):
    """
    Fitness:
    error + penalización por complejidad
    """
    mse = regressionError(individual)
    complexity_penalty = len(individual) * 10

    return mse + complexity_penalty,


toolbox.register("evaluate", getCost)


# ==========================================================
# OPERADORES GENÉTICOS
# ==========================================================

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=MUT_MIN_TREE_HEIGHT, max_=MUT_MAX_TREE_HEIGHT)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=primitiveSet)


# ==========================================================
# CONTROL DE BLOAT
# ==========================================================

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=LIMIT_TREE_HEIGHT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=LIMIT_TREE_HEIGHT))

def tres_graficas(y_pred):
	
	# Extraer columnas
	x1 = X[:, 0]
	x2 = X[:, 1]
	x3 = X[:, 2]

	# --------------------------------------------------
	# Gráfica 1: x1, x2, y
	# --------------------------------------------------

	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(x1, x2, y, marker='o', label='y real')
	ax.scatter(x1, x2, y_pred, marker='^', label='y pred')

	ax.set_xlabel("x1")
	ax.set_ylabel("x2")
	ax.set_zlabel("y")
	ax.set_title("x1 vs x2 vs y")

	plt.show()

	# --------------------------------------------------
	# Gráfica 2: x1, x3, y
	# --------------------------------------------------

	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(x1, x3, y, marker='o', label='y real')
	ax.scatter(x1, x3, y_pred, marker='^', label='y pred')

	ax.set_xlabel("x1")
	ax.set_ylabel("x3")
	ax.set_zlabel("y")
	ax.set_title("x1 vs x3 vs y")

	plt.show()

	# --------------------------------------------------
	# Gráfica 3: x2, x3, y
	# --------------------------------------------------

	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(x2, x3, y, marker='o', label='y real')
	ax.scatter(x2, x3, y_pred, marker='^', label='y pred')

	ax.set_xlabel("x2")
	ax.set_ylabel("x3")
	ax.set_zlabel("y")
	ax.set_title("x2 vs x3 vs y")

	plt.show()
	

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

    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # ------------------------------------------------------
    # Hall of Fame
    # ------------------------------------------------------

    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # ------------------------------------------------------
    # Evolución
    # ------------------------------------------------------

    population, logbook = elitism.eaSimpleWithElitism(
        population,
        toolbox,
        cxpb=P_CROSSOVER,
        mutpb=P_MUTATION,
        ngen=MAX_GENERATIONS,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    # ------------------------------------------------------
    # Mejor individuo
    # ------------------------------------------------------

    best = hof.items[0]

    print("\n" + "=" * 60)
    print("MEJOR SOLUCIÓN ENCONTRADA")
    print("=" * 60)

    print("\nExpresión simbólica:")
    print(best)

    print(f"\nLongitud: {len(best)}")
    print(f"Altura  : {best.height}")

    print("\nFitness:", best.fitness.values[0])

    # ------------------------------------------------------
    # Evaluación final
    # ------------------------------------------------------

    func = toolbox.compile(expr=best)

    y_pred = []

    for row in X:
        try:
            y_pred.append(func(*row))
        except Exception:
            y_pred.append(0.0)

    y_pred = np.array(y_pred)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("\nMétricas:")
    print(f"MSE  = {mse:.6f}")
    print(f"RMSE = {rmse:.6f}")
    print(f"MAE  = {mae:.6f}")
    print(f"R²   = {r2:.6f}")

    # ------------------------------------------------------
    # Gráfica valores reales vs predicción
    # ------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.scatter(y, y_pred)

    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())

    plt.plot([min_val, max_val], [min_val, max_val])

    plt.xlabel("Valores reales")
    plt.ylabel("Predicciones")
    plt.title("Regresión simbólica")
    plt.grid(True)

    plt.show()
    
    tres_graficas(y_pred)

    # ------------------------------------------------------
    # Árbol GP
    # ------------------------------------------------------

    nodes, edges, labels = gp.graph(best)

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.spring_layout(g, seed=RANDOM_SEED)

    plt.figure(figsize=(12, 8))

    nx.draw_networkx_nodes(g, pos, node_color="lightblue")
    nx.draw_networkx_nodes(g, pos, nodelist=[0], node_color="red", node_size=500)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8)

    plt.title("Árbol de la mejor solución")
    plt.axis("off")
    plt.show()
    
    # extract statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    #plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')
    plt.show()

    return population, logbook, hof


# ==========================================================
# EJECUCIÓN
# ==========================================================

if __name__ == "__main__":
    main()
