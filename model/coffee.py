import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.tree import plot_tree


def _load_file(filename):
    """
    Esta función se encarga de leer el archivo csv correspondiente agregando las validaciones en el caso de que no
    exista o surja un error Nos devuelve el dataset
    """
    try:
        data = pd.read_csv(filename, delimiter=';')
        return data
    except FileNotFoundError:
        print("El archivo CSV no se encuentra.")
    except pd.errors.EmptyDataError:
        print("El archivo CSV está vacío.")
    except pd.errors.ParserError:
        print("Error al analizar el archivo CSV. Verifica su formato.")
    except Exception as e:
        print(f"Se produjo un error inesperado: {str(e)}")
    return None


class CoffeeModel:
    def __init__(self, data) -> None:
        self.data = _load_file(data)
        self.data_clean = None

    def visualize(self):
        """
        Esta función se encarga de la ver los datos y gráficos
        """
        try:
            print('Información del dataset (datos nulos y tipos de datos por columna)')
            print(self.data.info())
            print('Descripción estadística de los datos (min, max, media, mediana, etc.)')
            print(self.data.describe())

            columnas_numericas = self.data.select_dtypes(include='number')

            matriz_correlacion = columnas_numericas.corr()

            sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', fmt=".2f")
            plt.ylim(len(matriz_correlacion), 0)
            plt.xlim(0, len(matriz_correlacion))
            plt.title("Matriz de Correlación")
            plt.show()

            for column in columnas_numericas:
                outliers_count = self._count_outliers(column)
                print(f'Cantidad de outliers en la columna {column}: {outliers_count}')

            plt.figure(figsize=(12, 8))

            for i, column in enumerate(self.data.select_dtypes(include=['float64', 'int64']).columns):
                plt.subplot(2, 2, i + 1)
                plt.boxplot(self.data[column])
                plt.title(f'Boxplot de la columna: {column}')

            plt.tight_layout()
            plt.show()
            plt.figure(figsize=(12, 8))

            for i, column in enumerate(self.data.select_dtypes(include=['float64', 'int64']).columns):
                plt.subplot(2, 2, i + 1)
                sns.histplot(self.data[column], kde=True)
                plt.title(f'Distribución de la columna: {column}')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error en la visualización de datos: {e}")

    def clean(self):
        """
        Esta función se encarga de la limpieza del dataset
        """
        try:
            print(self.data.dtypes)
            color_dummies = pd.get_dummies(self.data['Color'], prefix='Color')
            # Concatena las variables dummy al conjunto de datos original
            self.data_clean = pd.concat([self.data, color_dummies], axis=1)
            # Elimina la columna original 'Color'
            # self.data_clean = self.data_clean.drop(columns=['Color'])
            # self.data_clean.to_csv('data_clean.csv')
        except Exception as e:
            print(f"Error en la limpieza de datos: {e}")

    def _count_outliers(self, column):
        try:
            q1 = self.data[column].quantile(0.25)
            q3 = self.data[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers_count = ((self.data[column] < lower_bound) | (self.data[column] > upper_bound)).sum()

            return outliers_count
        except Exception as e:
            print(f"Error al contar outliers en la columna {column}: {e}")

    def standarize(self, standarize_type='zscore'):
        """
        Esta función se encarga de estandarizar las variables numéricas del dataset.
        """
        try:
            if standarize_type == 'zscore':
                scaler = StandardScaler()
            elif standarize_type == 'minmax':
                scaler = MinMaxScaler()
            elif standarize_type == 'robust':
                scaler = RobustScaler()
            elif standarize_type == 'maxabs':
                scaler = MaxAbsScaler()
            elif standarize_type == 'quantile':
                scaler = QuantileTransformer(output_distribution='uniform')
            else:
                print(f"Tipo de estandarización no válido: {standarize_type}")
                return

            self.data_clean[self.data_clean.select_dtypes(['float64', 'int64']).columns] = scaler.fit_transform(
                self.data_clean.select_dtypes(['float64', 'int64'])
            )

            print("Estandarización completada.")
        except Exception as e:
            print(f"Error en la estandarización de datos: {e}")

    def svm_lineal(self, cost_parameter=1.0):
        try:
            # Paso 1: Divide tu conjunto de datos en características (x) y etiquetas (y)
            x = self.data_clean.drop(columns=['Color'])
            y = self.data_clean['Color']

            # Paso 2: Divide los datos en conjuntos de entrenamiento y prueba
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Paso 3: Configura y entrena el modelo SVM con kernel lineal
            model = SVC(kernel='linear', C=cost_parameter)
            model.fit(x_train, y_train)

            # Paso 4: Realiza validación cruzada y muestra resultados
            cv_scores = cross_val_score(model, x_test, y_test, cv=5, scoring='accuracy')

            print(f"Resultados de la validación cruzada (k=5) para C={cost_parameter} (SVM Lineal):")
            print("Precisión promedio:", cv_scores.mean())
            print("Precisión por partición:", cv_scores)

            # Llama a la función _Metrics para calcular y mostrar otras métricas
            self._metrics(model, x_test, y_test)

        except Exception as e:
            print(f"Error en la predicción y evaluación del modelo SVM Lineal: {e}")

    @staticmethod
    def _metrics(model, x_test, y_test):
        try:
            # Calcula y muestra otras métricas como exhaustividad y exactitud
            predictions = model.predict(x_test)
            print("Exactitud:", accuracy_score(y_test, predictions))
            print("Exhaustividad:", recall_score(y_test, predictions, average='weighted'))
            print("Precisión:", precision_score(y_test, predictions, average='weighted'))

        except Exception as e:
            print(f"Error al calcular y mostrar métricas: {e}")

    def svm_gaussiano(self, cost_parameter=1.0, gamma_parameter='scale'):
        try:
            # Paso 1: Divide tu conjunto de datos en características (x) y etiquetas (y)
            x = self.data_clean.drop(columns=['Color'])
            y = self.data_clean['Color']

            # Paso 2: Divide los datos en conjuntos de entrenamiento y prueba
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Paso 3: Configura y entrena el modelo SVM con kernel gaussiano
            model = SVC(kernel='rbf', C=cost_parameter, gamma=gamma_parameter)
            model.fit(x_train, y_train)

            # Paso 4: Realiza validación cruzada y muestra resultados
            cv_scores = cross_val_score(model, x_test, y_test, cv=5, scoring='accuracy')

            print(
                f"Resultados de la validación cruzada (k=5) para C={cost_parameter}, gamma={gamma_parameter} (SVM Gaussiano):")
            print("Precisión promedio:", cv_scores.mean())
            print("Precisión por partición:", cv_scores)

            # Llama a la función _Metrics para calcular y mostrar otras métricas
            self._metrics(model, x_test, y_test)

        except Exception as e:
            print(f"Error en la predicción y evaluación del modelo SVM Gaussiano: {e}")

    def random_forest(self, n_estimators=100, max_depth=None):
        try:
            # Paso 1: Divide tu conjunto de datos en características (x) y etiquetas (y)
            x = self.data_clean.drop(columns=['Color'])
            y = self.data_clean['Color']

            # Paso 2: Divide los datos en conjuntos de entrenamiento y prueba
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Paso 3: Configura y entrena el modelo Random Forest
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(x_train, y_train)


            # Paso 4: Realiza validación cruzada y muestra resultados
            cv_scores = cross_val_score(model, x_test, y_test, cv=5, scoring='accuracy')

            print(
                f"Resultados de la validación cruzada (k=5) para n_estimators={n_estimators}, max_depth={max_depth} (Random Forest):")
            print("Precisión promedio:", cv_scores.mean())
            print("Precisión por partición:", cv_scores)

            # Llama a la función _Metrics para calcular y mostrar otras métricas
            self._metrics(model, x_test, y_test)

        except Exception as e:
            print(f"Error en la predicción y evaluación del modelo Random Forest: {e}")



