from model.coffee import CoffeeModel
import dash
from dash import dcc, html


model = CoffeeModel('CoffeeRatings.csv')


"""

import dash
from dash import dcc, html
import plotly.express as px

df = px.data.iris()



# Inicializar la aplicación Dash
app = dash.Dash(__name__,
                external_stylesheets=[
                    'https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.1.3/css/bootstrap.css',
                    'https://cdn.datatables.net/1.10.21/css/dataTables.bootstrap4.min.css',
                    'https://cdn.datatables.net/responsive/2.2.4/css/responsive.bootstrap4.min.css',
                    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.13.0/css/all.min.css'
                ],
                external_scripts=[
                    'https://code.jquery.com/jquery-3.5.1.js',
                    'https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js',
                    'https://cdn.datatables.net/1.10.21/js/dataTables.bootstrap4.min.js',
                    'https://cdn.datatables.net/responsive/2.2.4/js/dataTables.responsive.min.js',
                    'https://cdn.datatables.net/responsive/2.2.4/js/responsive.bootstrap4.min.js'
                ]

                )

# Diseño del navbar
navbar = html.Nav(
    className='navbar navbar-expand-lg navbar-light bg-light',
    children=[
        html.A(
            className='navbar-brand',
            href='#',
            children='TP3 Mineria de Datos'
        ),
        html.Button(
            className='navbar-toggler',
            type='button',
            **{'data-toggle': 'collapse', 'data-target': '#navbarNav', 'aria-controls': 'navbarNav',
               'aria-expanded': 'false', 'aria-label': 'Toggle navigation'},
            children=[
                html.Span(className='navbar-toggler-icon')
            ]
        ),
        html.Div(
            className='collapse navbar-collapse',
            id='navbarNav',
            children=[
                html.Ul(
                    className='navbar-nav',
                    children=[
                        html.Li(
                            className='nav-item active',
                            children=[
                                html.A(
                                    className='nav-link',
                                    href='#',
                                    children='DataSet'
                                )
                            ]
                        ),
                        html.Li(
                            className='nav-item',
                            children=[
                                html.A(
                                    className='Visualizaciones',
                                    href='#',
                                    children='Otra Página'
                                )
                            ]
                        ),
                        html.Li(
                            className='nav-item',
                            children=[
                                html.A(
                                    className='Limpieza',
                                    href='#',
                                    children='Otra Página'
                                )
                            ]
                        ),
                        html.Li(
                            className='nav-item',
                            children=[
                                html.A(
                                    className='SVM LINEAL',
                                    href='#',
                                    children='Otra Página'
                                )
                            ]
                        ),
html.Li(
                            className='nav-item',
                            children=[
                                html.A(
                                    className='SVM LINEAL',
                                    href='#',
                                    children='Otra Página'
                                )
                            ]
                        ),

                    ]
                )
            ]
        )
    ]
)

# Diseño de la aplicación Dash
app.layout = html.Div(children=[
    navbar,  # Agregar la barra de navegación
    html.Div(
        className='container mt-3',
        children=[
            html.H1(
                children='CoffeeRatings.csv',
                style={'textAlign': 'center'}
            ),
            dcc.Graph(
                id='scatter-plot',
                figure=px.scatter(df, x='sepal_width', y='sepal_length', color='species', size='petal_length')
            )
        ]
    )
])

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
"""
graphs = model.visualize()
print(graphs)
for graph in graphs:
    print(graph)
model.standarize()
print("SVM LINEAL")
model.svm_lineal()
print("SVM GAUSSIANO")
model.svm_gaussiano()
print("RANDOM FOREST")
model.random_forest()


