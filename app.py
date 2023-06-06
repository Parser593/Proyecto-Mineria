import os, pickle
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from datetime import datetime
import shutil
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor, plot_tree, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

plt.switch_backend('Agg')



app = Flask(__name__)
secret_key = os.urandom(16).hex()
app.secret_key = secret_key

# Ruta de la carpeta de carga
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')

# Verificar si la carpeta de carga existe, y crearla si no existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/")
def home():
    return render_template("index.html")


# Recibir archivo
@app.route("/upload", methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if (uploaded_file.filename.endswith('.csv') and uploaded_file.mimetype == "text/csv"):

        timestamp = datetime.now().strftime('%H%M%S')
        csv_file_name = f"{uploaded_file.filename}_{timestamp}.csv"
        file_path = os.path.join(UPLOAD_FOLDER, csv_file_name)
        uploaded_file.save(file_path)

        csv = pd.read_csv(file_path)
        csv =  csv.dropna()
        # Generar un nombre único para el archivo pickle
        pickle_file_name = f"{uploaded_file.filename}_{timestamp}.pkl"

        # Guardar el DataFrame en un archivo pickle
        pickle_file_path = os.path.join(UPLOAD_FOLDER, pickle_file_name)
        csv.to_pickle(pickle_file_path)


        nombreCSV = uploaded_file.filename
        nombreCSV = nombreCSV+"_"+timestamp

        dir_path = os.path.join(app.static_folder, "graphs", nombreCSV)
        os.makedirs(dir_path)

        # Guardar la información en la sesión
        session['file_uploaded'] = True
        session['file_path'] = pickle_file_path
        session['path_graphs'] = dir_path
        
        return redirect(url_for("project"))
    else:
        session['archivo_incorrecto'] = True
        return redirect(url_for('home'))

#CARGAR PAGINA DE PROYECTO
@app.route("/project")
def project():
    return render_template("project.html")


#BORRAR COOKIES DE SESION Y FILES GENERADOS EN LA MINERÍA
@app.route("/clear_session", methods=['GET'])
def clear_session():
    # Eliminar las variables de sesión
    graphs_path = session.get('path_graphs')
    #PKL CSV
    file_path_pkl = session.get('file_path')

    if os.path.exists(graphs_path):
        shutil.rmtree(graphs_path)

    if os.path.exists(file_path_pkl):
        os.remove(file_path_pkl)

    file_path_pkl = os.path.splitext(file_path_pkl)[0]
    file_path_pkl = file_path_pkl + ".csv"

    if os.path.exists(file_path_pkl):
        os.remove(file_path_pkl)

    session.clear()

    return jsonify(success=True)











#PROCEDIMIENTOS QUE REALIZAN EL EDA
@app.route("/eda_summ", methods=['POST'])
def edaStatSummary():
    if 'file_path' in session:
        file_path = session.get('file_path')
        csv = pd.read_pickle(file_path)
        
        described = csv.describe().to_json(orient='columns')
        return jsonify(described)
    else:
        clear_session()
        session['archivo_incorrecto'] = True
        return redirect(url_for('home'))


@app.route("/get_edahead", methods=['POST'])
def edahead15():
    file_path = session.get('file_path')
    csv = pd.read_pickle(file_path)
    head15 = csv.head(15).to_json(orient='index')
    return jsonify(head15)


@app.route("/generate_graph", methods=['POST'])
def EDA():
    if 'graficos_generados' in session:
        eda_paths = session['eda_paths']
        nombreCSV = os.path.basename(session.get('file_path'))
    else:
        file_path = session.get('file_path')
        if file_path:
            csv_file = pd.read_pickle(file_path)
            nombreCSV = os.path.basename(file_path)
            eda_paths = []
            histogramas = []
            numeric_columns = csv_file.select_dtypes(include='number')
            graphs_path = session.get('path_graphs')
            graph_text = "graphs/" + os.path.splitext(nombreCSV)[0] + "/"

            for columna in numeric_columns:
                fig = px.histogram(numeric_columns, x=columna)
                histogramas.append(fig)

            for i, histograma in enumerate(histogramas):
                filename = f"histd{i}{nombreCSV}.html"
                hist_path = os.path.join(graphs_path, filename)

                histograma.write_html(hist_path)
                eda_paths.append(
                    url_for('static', filename=graph_text + filename))

            i = 0
            for col in csv_file.select_dtypes(include='object'):
                if csv_file[col].nunique() < 10:
                    fig = px.histogram(csv_file, y=col)
                    filename = f"histcat{i}{nombreCSV}.html"
                    hist_path = os.path.join(graphs_path, filename)
                    fig.write_html(hist_path)
                    eda_paths.append(
                        url_for('static', filename=graph_text + filename))
                    i += 1

            nombres_columnas = numeric_columns.columns.tolist()
            triangle_lower = np.tril(numeric_columns.corr(), k=0)
            triangle_lower = np.where(
                triangle_lower != 0, np.round(triangle_lower, 3), np.nan)

            
            fig = go.Figure(data=go.Heatmap(z=triangle_lower,
                                            x=nombres_columnas, y=nombres_columnas,
                                            colorscale='RdBu_r',
                                            texttemplate="%{z}"))
            fig.update_layout(
                width=1200,
                height=600
            )

            fig.update_traces(textfont={'size': 10})

            filename = f"corr{nombreCSV}.html"
            hist_path = os.path.join(graphs_path, filename)
            fig.write_html(hist_path)
            eda_paths.append(url_for('static', filename=graph_text+ filename))

            
            histogramas = []
            q1 = csv_file.quantile(0.25)
            q3 = csv_file.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            mask = (csv_file >= lower_bound) & (csv_file <= upper_bound)
            csv_file_filter = csv_file[mask]
            
            
            csv_file_filter = csv_file_filter.select_dtypes(include='number')

            for columna in csv_file_filter:
                fig = px.histogram(csv_file_filter, x=columna)
                histogramas.append(fig)

            for i, histograma in enumerate(histogramas):
                filename = f"finhist{i}{nombreCSV}.html"
                hist_path = os.path.join(graphs_path, filename)
                histograma.write_html(hist_path)
                eda_paths.append(
                    url_for('static', filename=graph_text + filename))

            session['eda_paths'] = eda_paths
            session['graficos_generados'] = True
        else:
            return 'Error'

    return jsonify({'graficoPaths': eda_paths,
                    'nombreCSV': nombreCSV,
                    })





#PROCEDIMIENTOS QUE REALIZAN EL PCA
@app.route("/pca_correlation", methods=['POST'])
def pcaCorrelation():
    pca_path = ""
        
    file_path = session.get('file_path')
    # correlacion
    csvfile = pd.read_pickle(file_path)
    graph_text = "graphs/" + os.path.splitext(os.path.basename(file_path))[0] + "/"
    graphs_path = session.get('path_graphs')
        
    corrPcsv = csvfile.corr(method='pearson')
    numeric_columns = csvfile.select_dtypes(include='number')

    nombres_columnas = numeric_columns.columns.tolist()
    triangle_lower = np.tril(numeric_columns.corr(), k=0)
    triangle_lower = np.where(
    triangle_lower != 0, np.round(triangle_lower, 3), np.nan)

    fig = go.Figure(data=go.Heatmap(z=triangle_lower,
                                        x=nombres_columnas, y=nombres_columnas,
                                        colorscale='RdBu_r',
                                        texttemplate="%{z}"))
    fig.update_layout(
            width=1200,
            height=600)

        # Personaliza el trazo de la matriz de correlación
        # Ajusta el tamaño de fuente aquí
    fig.update_traces(textfont={'size': 10})

    filename = f"pca_corr{os.path.basename(file_path)}.html"
    hist_path = os.path.join(graphs_path, filename)
    fig.write_html(hist_path)
    pca_path = url_for('static', filename=graph_text + filename)
    session['corr_pca'] = pca_path


    corrPcsv = corrPcsv.round(5)
    corrPcsv = corrPcsv.to_json(orient='columns')
    return jsonify(corrPcsv=corrPcsv, pca_path=pca_path)


@app.route("/covarComp", methods=['POST'])
def PCAcovarComp():
    pca_var = ""
    file_path_csv = session.get('file_path')
    csvfile = pd.read_pickle(file_path_csv)

    Estandarizar = StandardScaler()
    NuevaMatriz = csvfile.select_dtypes(include='number')
    MEstandarizada = Estandarizar.fit_transform(NuevaMatriz)

    pca = PCA(n_components=None)     # pca=PCA(n_components=None), pca=PCA(.85)
    pca.fit(MEstandarizada)          # Se obtiene los componentes
    components = pca.components_

    varianza = pca.explained_variance_ratio_

    if 'var_pca' in session:
        pca_var = session.get('var_pca')
    else:
        graph_text = "graphs/" + os.path.splitext(os.path.basename(file_path_csv))[0]+"/"
        
        graphs_path = session.get('path_graphs')



        fig = px.line(np.cumsum(pca.explained_variance_ratio_), labels={
                  'x': 'Número de Componentes', 'y': 'Varianza Acumulada'})
        fig.update_layout(showlegend=False, xaxis=dict(
        gridcolor='lightgray'), yaxis=dict(gridcolor='lightgray'))

        filename = f"pcavar{os.path.basename(file_path_csv)}.html"
        hist_path = os.path.join(graphs_path, filename)
        fig.write_html(hist_path)
        pca_var = url_for('static', filename=graph_text+ filename)
        session['var_pca'] = pca_var

    cargasComponentes = pd.DataFrame(abs(components), columns=NuevaMatriz.columns)
    cargasComponentes = cargasComponentes.to_json(orient='index')

    return jsonify({
        'pca_var': pca_var,
        'varianza': varianza.tolist(),
        'components': components.tolist(),
        'cargasComponentes': cargasComponentes
    })


@app.route("/procesarComponentes", methods=['POST'])
def m2PCA():
    datos = request.json
    comp_seleccionados = datos['compSeleccionados']
    file_path = session.get('file_path')
    csvfile = pd.read_pickle(file_path)
    csvpca = csvfile.filter(items=comp_seleccionados)
    file_pca = "tree"+os.path.basename(file_path)
    file_pca = os.path.join(os.path.dirname(file_path), file_pca)
    csvpca.to_pickle(file_pca)
    session['path_pca_pickle'] = file_pca

    csvpca = csvpca.head(15)
    csvpca_list = csvpca.to_json(orient='index')


    return jsonify({
        'csvpca': csvpca_list
    })


@app.route('/treeSum', methods=['POST'])
def treeSumm():
    file_path = session.get('file_path')
    pkl_path = "tree"+os.path.basename(file_path)
    pkl_path = os.path.join(os.path.dirname(file_path), pkl_path)
    csv = pd.read_pickle(pkl_path)
    
    described = round(csv.describe(),5) .to_json(orient='index')
    return jsonify(described)

@app.route('/grafica_Arboles', methods=['POST'])
def arbolesGraph():

    file_path_csv = session.get('file_path')
    pkl_path = "tree"+os.path.basename(file_path_csv)
    pkl_path = os.path.join(os.path.dirname(file_path_csv), pkl_path)
    csv = pd.read_pickle(pkl_path)
    numeric_columns = csv.select_dtypes(include='number')
    
    fig = px.line(csv, x=csv.index, y=numeric_columns.columns)  # Usar 'Index' como la columna 'x'
    fig.update_traces(marker=dict(symbol='cross', size=10))
    fig.update_layout(
    xaxis_title='Index',
    showlegend=True,
    height=600,
    width=1200,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    graphs_path = session.get('path_graphs')
    graph_text = "graphs/" + os.path.splitext(os.path.basename(file_path_csv))[0]+"/"

    filename = f"{os.path.basename(pkl_path)}.html"
    graph_path = os.path.join(graphs_path, filename)
    fig.write_html(graph_path)
    graph_var = url_for('static', filename=graph_text+ filename)

    return graph_var

@app.route('/RegressionDecisionTree', methods=['POST'])
def regresiontree():
    data = request.get_json()
    # Acceder a los datos recibidos
    var_dependiente = data['varDependiente']
    caracteristicas = data['caracteristicas']
    max_depth = int(data['max_depth'])
    min_samples_split = int(data['min_samples_split'])
    min_samples_leaf = int(data['min_samples_leaf'])
    random_state = int(data['random_state'])


    #Abrir archivo
    file_path_csv = session.get('file_path')
    pkl_path = "tree"+os.path.basename(file_path_csv)
    pkl_path = os.path.join(os.path.dirname(file_path_csv), pkl_path)
    pkl = pd.read_pickle(pkl_path)

    x = np.array(pkl.filter(items=caracteristicas))
    
    y = np.array(pkl[var_dependiente])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)
    
    pronosticoAD = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, 
                                         min_samples_leaf=min_samples_leaf, random_state=random_state)
    pronosticoAD.fit(x_train, y_train)

    y_Pronostico = pronosticoAD.predict(x_test)

    r2score = round((r2_score(y_test, y_Pronostico)*100),5)

    criterio = pronosticoAD.criterion
    varImportance = pronosticoAD.feature_importances_
    mae = round(mean_absolute_error(y_test, y_Pronostico),5)
    mse = round((mean_squared_error(y_test, y_Pronostico)*100),5)
    rmse = round((mean_squared_error(y_test, y_Pronostico, squared=False)*100),5)

    df = pd.DataFrame({'Real': y_test.flatten(), 'Pronostico': y_Pronostico.flatten()})
    if(df.shape[0]>70):
        df = df.drop(df.index[70:])

    df = df.round(5)

    # Crear el gráfico de línea
    fig = px.line(df, y=['Real', 'Pronostico'])

    fig.update_traces(marker=dict(symbol='cross', size=10))
    fig.update_layout(
    xaxis_title='Index',
    showlegend=True,
    height=600,
    width=1200,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    graphs_path = session.get('path_graphs')
    graph_text = "graphs/" + os.path.splitext(os.path.basename(file_path_csv))[0]+"/"

    filename = f"treegraph{os.path.basename(pkl_path)}.html"
    graph_path = os.path.join(graphs_path, filename)
    fig.write_html(graph_path)
    test_graph = url_for('static', filename=graph_text+ filename)



    plt.figure(figsize=(10,10))  
    plot_tree(pronosticoAD, feature_names = caracteristicas)
    filename = f"treerep{os.path.basename(pkl_path)}.png"
    graph_path = os.path.join(graphs_path, filename)
    
    plt.savefig(graph_path)
    tree_graph = url_for('static', filename=graph_text + filename)
    

    varImportance_df = pd.DataFrame({'Importance': varImportance})
    varImportance_df = varImportance_df.round(5)
    varImportance_json = varImportance_df.to_json(orient='index')


    modelo_pkl = f"arbolpron_{os.path.basename(pkl_path)}"
    modelo_pkl = os.path.join(os.path.dirname(pkl_path), modelo_pkl)

    with open(modelo_pkl, 'wb') as archivo_pkl:
        pickle.dump(pronosticoAD, archivo_pkl)

    results = {
        'r2score': r2score,
        'criterio': criterio,
        'varImportance': varImportance_json,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'valores': df.to_json(orient='index'),
        'test_graph': test_graph,
        'tree_graph': tree_graph
    }

    # Enviar los resultados como una respuesta JSON
    return jsonify(results)

@app.route('/Pronosticar', methods=['POST'])
def pronosticar():
    file_path_csv = session.get('file_path')
    pkl_path = "tree"+os.path.basename(file_path_csv)
    pkl_path = os.path.join(os.path.dirname(file_path_csv), pkl_path)
    modelo_pkl = f"arbolpron_{os.path.basename(pkl_path)}"
    modelo_pkl = os.path.join(os.path.dirname(pkl_path), modelo_pkl)

    pronosticoAD = []


    # Cargar el modelo desde el archivo pkl
    with open(modelo_pkl, 'rb') as archivo_pkl:
        pronosticoAD = pickle.load(archivo_pkl)
    
    caracteristicas = request.json  # Obtener los datos enviados desde el formulario
    
    # Acceder a los valores de las características
    caracteristicas_valores = [float(value) for value in caracteristicas.values()]


    # Realizar la predicción utilizando el modelo pronosticado
    pronostico = pronosticoAD.predict([caracteristicas_valores])

    pronostico = str(round(pronostico[0],5))

    # Devolver el resultado de la predicción como respuesta JSON
    return pronostico







@app.route('/RegressionRandomForest', methods=['POST'])
def regresionforest():
    data = request.get_json()
    # Acceder a los datos recibidos
    var_dependiente = data['varDependiente']
    caracteristicas = data['caracteristicas']
    n_estimators = int(data['n_estimators'])
    min_samples_split = int(data['min_samples_split'])
    min_samples_leaf = int(data['min_samples_leaf'])
    random_state = int(data['random_state'])


    #Abrir archivo
    file_path_csv = session.get('file_path')
    pkl_path = "tree"+os.path.basename(file_path_csv)
    pkl_path = os.path.join(os.path.dirname(file_path_csv), pkl_path)
    pkl = pd.read_pickle(pkl_path)

    x = np.array(pkl.filter(items=caracteristicas))
    
    y = np.array(pkl[var_dependiente])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)
    
    pronosticoBA = RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split, 
                                         min_samples_leaf=min_samples_leaf, random_state=random_state)
    pronosticoBA.fit(x_train, y_train)

    y_Pronostico = pronosticoBA.predict(x_test)

    r2score = round((r2_score(y_test, y_Pronostico)*100),5)

    criterio = pronosticoBA.criterion
    varImportance = pronosticoBA.feature_importances_
    mae = round(mean_absolute_error(y_test, y_Pronostico),5)
    mse = round((mean_squared_error(y_test, y_Pronostico)*100),5)
    rmse = round((mean_squared_error(y_test, y_Pronostico, squared=False)*100),5)

    df = pd.DataFrame({'Real': y_test.flatten(), 'Pronostico': y_Pronostico.flatten()})
    if(df.shape[0]>70):
        df = df.drop(df.index[70:])
    df = df.round(5)

    # Crear el gráfico de línea
    fig = px.line(df, y=['Real', 'Pronostico'])

    fig.update_traces(marker=dict(symbol='cross', size=10))
    fig.update_layout(
    xaxis_title='Index',
    showlegend=True,
    height=600,
    width=1200,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    graphs_path = session.get('path_graphs')
    graph_text = "graphs/" + os.path.splitext(os.path.basename(file_path_csv))[0]+"/"

    filename = f"forestgraph{os.path.basename(pkl_path)}.html"
    graph_path = os.path.join(graphs_path, filename)
    fig.write_html(graph_path)
    test_graph = url_for('static', filename=graph_text+ filename)
    

    varImportance_df = pd.DataFrame({'Importance': varImportance})
    varImportance_df = varImportance_df.round(5)
    varImportance_json = varImportance_df.to_json(orient='index')


    modelo_pkl = f"bosquepron_{os.path.basename(pkl_path)}"
    modelo_pkl = os.path.join(os.path.dirname(pkl_path), modelo_pkl)

    with open(modelo_pkl, 'wb') as archivo_pkl:
        pickle.dump(pronosticoBA, archivo_pkl)

    results = {
        'r2score': r2score,
        'criterio': criterio,
        'varImportance': varImportance_json,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'valores': df.to_json(orient='index'),
        'test_graph': test_graph
    }

    # Enviar los resultados como una respuesta JSON
    return jsonify(results)


@app.route('/GraphTreeForest', methods=['POST'])
def graphForest():
    data = request.get_json()
    # Acceder a los datos recibidos
    numeroArbol = int(data['numeroArbol'])
    caracteristicas = data['caracteristicas']

    file_path_csv = session.get('file_path')
    pkl_path = "tree"+os.path.basename(file_path_csv)
    pkl_path = os.path.join(os.path.dirname(file_path_csv), pkl_path)

    modelo_pkl = f"bosquepron_{os.path.basename(pkl_path)}"
    modelo_pkl = os.path.join(os.path.dirname(pkl_path), modelo_pkl)

    pronosticoBA = []


    # Cargar el modelo desde el archivo pkl
    with open(modelo_pkl, 'rb') as archivo_pkl:
        pronosticoBA = pickle.load(archivo_pkl)

    graphs_path = session.get('path_graphs')
    graph_text = "graphs/" + os.path.splitext(os.path.basename(file_path_csv))[0]+"/"

    estimador = pronosticoBA.estimators_[numeroArbol]


    plt.figure(figsize=(10,10))  
    plot_tree(estimador, feature_names = caracteristicas)


    filename = f"forestrep{os.path.basename(pkl_path)}.png"
    graph_path = os.path.join(graphs_path, filename)
    
    plt.savefig(graph_path)
    forest_graph = url_for('static', filename=graph_text + filename)

    return forest_graph


@app.route('/PronosticarForest', methods=['POST'])
def pronosticarForest():
    file_path_csv = session.get('file_path')
    pkl_path = "tree"+os.path.basename(file_path_csv)
    pkl_path = os.path.join(os.path.dirname(file_path_csv), pkl_path)
    modelo_pkl = f"bosquepron_{os.path.basename(pkl_path)}"
    modelo_pkl = os.path.join(os.path.dirname(pkl_path), modelo_pkl)

    pronosticoBA = []


    # Cargar el modelo desde el archivo pkl
    with open(modelo_pkl, 'rb') as archivo_pkl:
        pronosticoBA = pickle.load(archivo_pkl)
    
    caracteristicas = request.json  # Obtener los datos enviados desde el formulario
    
    # Acceder a los valores de las características
    caracteristicas_valores = [float(value) for value in caracteristicas.values()]


    # Realizar la predicción utilizando el modelo pronosticado
    pronostico = pronosticoBA.predict([caracteristicas_valores])

    pronostico = str(round(pronostico[0],5))

    # Devolver el resultado de la predicción como respuesta JSON
    return pronostico








@app.route('/ClasificationDecisionTree', methods=['POST'])
def clasificationtree():

    data = request.get_json()
    # Acceder a los datos recibidos
    var_dependiente = data['varDependiente']
    caracteristicas = data['caracteristicas']
    max_depth = int(data['max_depth'])
    min_samples_split = int(data['min_samples_split'])
    min_samples_leaf = int(data['min_samples_leaf'])
    random_state = int(data['random_state'])


    #Abrir archivo
    file_path_csv = session.get('file_path')
    pkl_path = "tree"+os.path.basename(file_path_csv)
    pkl_path = os.path.join(os.path.dirname(file_path_csv), pkl_path)
    pkl = pd.read_pickle(pkl_path)

    x = np.array(pkl.filter(items=caracteristicas))
    
    y = np.array(pkl[var_dependiente])
    print("inicio")
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)
    

    
    ClasificacionAD = DecisionTreeClassifier(max_depth=max_depth, 
                                          min_samples_split=min_samples_split, 
                                         min_samples_leaf=min_samples_leaf, 
                                         random_state=random_state)
    ClasificacionAD.fit(x_train, y_train)
    
    y_Pronostico = ClasificacionAD.predict(x_test)
    print("Final")
    accuracyscore = round((accuracy_score(y_test, y_Pronostico)*100),5)

    criterio = ClasificacionAD.criterion
    varImportance = ClasificacionAD.feature_importances_
    matrizClas = pd.crosstab(y_test.ravel(), y_Pronostico)
    matrizClas = matrizClas.to_json(orient='index')
    

    report = classification_report(y_test, y_Pronostico, output_dict=True)
    report = pd.DataFrame(report).round(5)
    report = report.T

    df = pd.DataFrame({'Real': y_test.flatten(), 'Pronostico': y_Pronostico.flatten()})
    print(df.shape)
    if(df.shape[0]>70):
        df = df.drop(df.index[70:])
        print(df.shape)

    df = df.round(5)


    graphs_path = session.get('path_graphs')
    graph_text = "graphs/" + os.path.splitext(os.path.basename(file_path_csv))[0]+"/"

    print("Inicio2")
    plt.figure(figsize=(15,15))  
    plot_tree(ClasificacionAD, feature_names = caracteristicas)
    filename = f"treeclass{os.path.basename(pkl_path)}.png"
    graph_path = os.path.join(graphs_path, filename)
    
    plt.savefig(graph_path)
    tree_graph = url_for('static', filename=graph_text + filename)
    print("Fin2")

    varImportance_df = pd.DataFrame({'Importance': varImportance})
    varImportance_df = varImportance_df.round(5)
    varImportance_json = varImportance_df.to_json(orient='index')


    modelo_pkl = f"arbol_class{os.path.basename(pkl_path)}"
    modelo_pkl = os.path.join(os.path.dirname(pkl_path), modelo_pkl)

    with open(modelo_pkl, 'wb') as archivo_pkl:
        pickle.dump(ClasificacionAD, archivo_pkl)

    results = {
        'accuracyscore': accuracyscore,
        'criterio': criterio,
        'varImportance': varImportance_json,
        'matrizClass': matrizClas,
        'report': report.to_json(orient='index'),
        'valores': df.to_json(orient='index'),
        'tree_graph': tree_graph
    }

    # Enviar los resultados como una respuesta JSON
    return jsonify(results)


@app.route('/PronosticarClasifiaction', methods=['POST'])
def pronosticarClasification():
    file_path_csv = session.get('file_path')
    pkl_path = "tree"+os.path.basename(file_path_csv)
    pkl_path = os.path.join(os.path.dirname(file_path_csv), pkl_path)
    modelo_pkl = f"arbol_class{os.path.basename(pkl_path)}"
    modelo_pkl = os.path.join(os.path.dirname(pkl_path), modelo_pkl)

    pronosticoAD = []


    # Cargar el modelo desde el archivo pkl
    with open(modelo_pkl, 'rb') as archivo_pkl:
        pronosticoAD = pickle.load(archivo_pkl)
    
    caracteristicas = request.json  # Obtener los datos enviados desde el formulario
    
    # Acceder a los valores de las características
    caracteristicas_valores = [float(value) for value in caracteristicas.values()]
    print(caracteristicas_valores)

    # Realizar la predicción utilizando el modelo pronosticado
    pronostico = pronosticoAD.predict([caracteristicas_valores])

    pronostico = str(pronostico[0])

    # Devolver el resultado de la predicción como respuesta JSON
    return pronostico