from model.coffee import CoffeeModel

model = CoffeeModel('CoffeeRatings.csv')

#model.visualize()
model.clean()
model.standarize()
#model.visualize()
print("SVM LINEAL")
model.svm_lineal()
print("SVM GAUSSIANO")
model.svm_gaussiano()
print("RANDOM FOREST")
model.random_forest()
