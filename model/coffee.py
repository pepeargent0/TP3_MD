import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score
from io import BytesIO
from tabulate import tabulate
from sklearn.tree import plot_tree

from dataclasses import dataclass
from pydantic import BaseModel
@dataclass
class Graph:
    title: str
    imgs: str
    description: str



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
            graphs = []
            columnas = self.data.columns
            formatted_table = tabulate([columnas], headers='keys', tablefmt='pretty')
            graph_info = Graph(
                title='Columnas del CSV',
                imgs=formatted_table,
                description=''
            )
            graphs.append(graph_info)
            data_info = self.data.info  # Llama a la función
            print(data_info, self.data.describe())
            formatted_table = tabulate([data_info], headers='keys', tablefmt='pretty')

            graph_info = Graph(
                title='Info del CSV',
                imgs=formatted_table,
                description='Tiene un total de 835 datos y 11 variables Se puede observar que se cuenta con 10 variables '
                            'numéricas (int64) y 1 variable categórica (object).No hay datos nulos'
            )
            graphs.append(graph_info)
            formatted_table = tabulate([self.data.describe()], headers='keys', tablefmt='pretty')
            graph_info = Graph(
                title='Descripción estadística de los datos (min, max, media, mediana, etc.)',
                imgs=formatted_table,
                description=''
            )
            graphs.append(graph_info)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(x='Color', data=self.data, palette='viridis', ax=ax)
            ax.set_title('Distribución de Colores')

            # Guarda la figura en el objeto BytesIO
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)

            # Puedes acceder a los bytes de la imagen usando buffer.getvalue()
            bytes_imagen = buffer.getvalue()

            # Cierra la figura para liberar recursos
            plt.close()
            graph_info = Graph(
                title='Distribución de Colores',
                imgs=bytes_imagen,
                description='Se puede observar que el dataset esta desbalanceado, hay una gran cantidad de color green, casi 700, mientras que de Blue-Green apenas llega a 50 y Bluish-Green a 100.'
            )
            graphs.append(graph_info)
            columnas_numericas = self.data.select_dtypes(include=['int64']).columns
            ax = self.data[columnas_numericas].hist(bins=20, figsize=(15, 10), color='blue', edgecolor='black', grid=False)
            plt.suptitle('Histogramas de variables numéricas', x=0.5, y=1.05, fontsize=16)
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plt.close()
            graph_info = Graph(
                title='Histogramas de variables numéricas',
                imgs=buffer,
                description='En todos los gráficos se observan 2 picos muy marcados en los extremos'
            )
            graphs.append(graph_info)

            fig, axes = plt.subplots(3, 4, figsize=(15, 10))
            plt.tight_layout()
            for i, feature in enumerate(columnas_numericas, 1):
                row = (i - 1) // 4
                col = (i - 1) % 4
                sns.boxplot(x=self.data[feature], color='skyblue', ax=axes[row, col])
                axes[row, col].set_title(f'Boxplot de {feature}')
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            bytes_imagen = buffer.getvalue()
            plt.close()
            graph_info = Graph(
                title='Boxplot para visualizar outliers',
                imgs=bytes_imagen,
                description=':Se puede observar que hay una gran presencia de lavores outliers en casi todas las columnas menos en Scores_Moisture'
            )
            graphs.append(graph_info)
            return graphs
        except Exception as e:
            print(f"Error en la visualización de datos: {e}")

    def clean(self):
        """
        Esta función se encarga de la limpieza del dataset
        """
        try:
            color_dummies = pd.get_dummies(self.data['Color'], prefix='Color')
            self.data_clean = pd.concat([self.data, color_dummies], axis=1)
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

    def random_forest(self, n_estimators=100, max_depth=1):
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



