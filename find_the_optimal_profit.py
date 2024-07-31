import numpy as np
import pandas as pd
import random
from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sense, Problem, Options, Sum

#DEFINE DATA
n = 8  # number of products
S = 2  # number of scenarios
ps = [0.5, 0.5]  # scenario probabilities
m = 5  # number of units to be ordered before production

A_matrix = np.random.randint(0, 11, size=(8, 5))  

# Generate random demand vector
D1_vector = np.random.binomial(10, 0.5, 8)  # demand vector
D2_vector = np.random.binomial(10, 0.5, 8)  # demand vector
#CREATE CONTAINER
M = Container()
#DECLARE SETS
product = Set(M, name = "product", records = ["product_1", "product_2", "product_3","product_4", "product_5", "product_6", "product_7", "product_8"], description = "Product")
material = Set(M, name = "material", records = ["material_1", "material_2", "material_3", "material_4", "material_5"], description = "material")

#DECLARE PARAMETERS
b_records = [[f"material_{i}", round(random.uniform(30, 50), 2)] for i in range(1, 6)]
b = Parameter(M, name="b", domain=material, description="cost of materials", records = b_records)
s_records = [[f"material_{i}", round(random.uniform(10, 20), 2)] for i in range(1, 6)]
s = Parameter(M, name="s", domain=material, description="salvage value of materials", records = s_records)
#s[material] = round(random.uniform(10, 20), 2)
l_records = [[f"product_{i}", round(random.uniform(1000, 2000), 2)] for i in range(1, 9)]
l = Parameter(M, name="l", domain=product, description="additional cost", records = l_records)
#l[product] = round(random.uniform(1000, 20000), 2)
q_records = [[f"product_{i}", round(random.uniform(2000, 3000), 2)] for i in range(1, 9)]
q = Parameter(M, name="q", domain=product, description="selling price", records = q_records)
#q[product] = round(random.uniform(2000, 3000), 2)

A_records = pd.DataFrame(A_matrix, index=[f"product_{i}" for i in range(1,9)], columns=[f"material_{j}" for j in range(1,6)])
A = Parameter(M,name = "A", domain = [product, material], description = "material for each product", records = A_records, uels_on_axes=True)

D1 = Parameter(M, name = "D1",domain = [product], description = "demand for scenario 1", records = D1_vector)
D2 = Parameter(M, name = "D2",domain = [product], description = "demand for scenario 2", records = D2_vector)
print("\nDemand D1: ")
print(D1.records)

print("\nDemand D2: ")
print(D2.records)

print("\nPre-order cost for each material: ")
print(b.records)
print("\nAdditional cost for each product: ")
print(l.records)
print("\nSelling price for each product: ")
print(q.records)
print("\nSavage cost for each part: ")
print(s.records)

print("\nMatrix A for demanding materials: ")
print(A.records.pivot(index="product",columns="material", values = "value"))

#DECLARE VARIABLES
x = Variable(M, 'x', domain= [material], type ="free", description = "Number of pre-order materials")
y1 = Variable(M, 'y1', domain = [material], type ="free", description = "Number of leftover materials")
z1 = Variable(M, 'z1', domain= [product], type ="free", description = "Number of sold products")

y2 = Variable(M, 'y2', domain = [material], type ="free", description = "Number of leftover materials")
z2 = Variable(M, 'z2', domain= [product], type ="free", description = "Number of sold products")

x.lo[material] = 0
y1.lo[material] = 0
z1.up[product] = 10
z1.lo[product] = 0

y2.lo[material] = 0
z2.up[product] = 10
z2.lo[product] = 0
#DEFINE EQUATION
Eq1Sce1 = Equation(
                    M, 
                    name = "Eq1Sce1", 
                    domain = [product],
                    description = "Demand always higher than sales", 
                    )
Eq1Sce1[product] = z1[product] <= D1[product]
Eq2Sce1 = Equation(
               M,
               name = "Eq2Sce1",
               domain = [material],
               description = "Define values for remaining materials", 
               )
Eq2Sce1[material] = y1[material] == x[material] - Sum( product, A[product,material]*z1[product])

Eq1Sce2 = Equation(
                    M, 
                    name = "Eq1Sce2", 
                    domain = [product],
                    description = "Demand always higher than sales", 
                    )
Eq1Sce2[product] = z2[product] <= D2[product]
Eq2Sce2 = Equation(
               M,
               name = "Eq2Sce2",
               domain = [material],
               description = "Define values for remaining materials", 
               )
Eq2Sce2[material] = y2[material] == x[material] - Sum( product, A[product,material]*z2[product])

#DEFINE OBJECTIVE
obj = Sum( material, b[material]*x[material] ) + 0.5 * (Sum( product, (l[product] - q[product])* z1[product] ) - Sum( material, s[material]*y1[material])) + 0.5 * (Sum( product, (l[product] - q[product])* z2[product] ) - Sum( material, s[material]*y2[material]))
#DEFINE MODEL 
Model = Model( M, "hello", equations = [Eq1Sce1, Eq2Sce1,Eq1Sce2, Eq2Sce2], problem = "LP", sense = Sense.MIN, objective = obj)
#SOLVE 
Model.solve()
print("\nObjective function value ", Model.objective_value)
print("\nNumber of pre-order material (x): ")
print(x.records)

print("\nSCENARIO 1: ")
print("\nNumber of sold product (z) in scenario 1: ")
print(z1.records)

print("Number of leftover parts (y) in scenario 1: ")
print(y1.records)

print("\nSCENARIO 2: ")
print("\nNumber of sold product (z) in scenario 2: ")
print(z2.records)

print("\nNumber of leftover parts (y) in scenario 2: ")
print(y2.records)

#RESULT