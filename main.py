from model.coffee import CoffeeModel

model = CoffeeModel('CoffeeRatings.csv')

model.visualize()
model.clean()
model.standarize()
model.visualize()
model.SVMLineal()
model.SVMGaussiano()
model.RandomForest()
