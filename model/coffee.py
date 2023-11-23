import pandas as pd
import matplotlib as plt
import seaborn as sns

class CoffeeModel:
    def __init__(self, data) -> None:
        self.data = self._LoadFile(data)
        self.data_clean = None
    
    def _LoadFile(self, filename):
        """
        Esta función se encarga de leer el archivo csv correspondiente agregando las validaciones en el caso de que no exista o surja un error
        Nos devuelve el dataset
        """
        try:
            data = pd.read_csv(data)
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
            self.data.drop_duplicates(subset=["Name"], keep="first")
            self.data_clean = self.data
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.data_clean['Type'])
            self.data_clean.to_csv('pop.csv')
        except Exception as e:
            print(f"Error en la limpieza de datos: {e}")

    def _count_outliers(self, column):
        try:
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_count = ((self.data[column] < lower_bound) | (self.data[column] > upper_bound)).sum()

            return outliers_count
        except Exception as e:
            print(f"Error al contar outliers en la columna {column}: {e}")
    
    def standarize(self, standarize_type):
        pass

    def SVMLineal (self, k = 5):
        pass

    def SVMGaussiano (self, k = 5):
        pass

    def RandomForest (self, k = 5):
        pass

    def _Metrics (self, model):
        pass