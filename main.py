from model.coffee import CoffeeModel

model = CoffeeModel('CoffeeRatings.csv')
model.visualize()
model.clean()
model.visualize_limpieza()
model.standarize()
print("SVM LINEAL")
model.svm_lineal(cost_parameter=0.01)
print("SVM GAUSSIANO")
model.svm_gaussiano(cost_parameter=0.01)
print("RANDOM FOREST")
model.random_forest()
