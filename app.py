import os, pickle
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, send_file, make_response
from datetime import datetime
import shutil
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn import model_selection
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor, plot_tree, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix, accuracy_score, RocCurveDisplay, pairwise_distances_argmin_min, roc_curve, auc
import matplotlib.pyplot as plt
from kneed import KneeLocator

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

    datos = [pronosticoAD, caracteristicas]

    with open(modelo_pkl, 'wb') as archivo_pkl:
        pickle.dump(datos, archivo_pkl)

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

    pronosticoAD = pronosticoAD[0]
    
    caracteristicas = request.json  # Obtener los datos enviados desde el formulario
    
    # Acceder a los valores de las características
    caracteristicas_valores = [float(value) for value in caracteristicas.values()]


    # Realizar la predicción utilizando el modelo pronosticado
    pronostico = pronosticoAD.predict([caracteristicas_valores])

    pronostico = str(round(pronostico[0],5))

    # Devolver el resultado de la predicción como respuesta JSON
    return pronostico

@app.route("/descargar_regtree", methods=['GET'])
def descargar_regtree():
    # Ruta al archivo que se va a descargar
    file_path_csv = session.get('file_path')
    pkl_path = "tree"+os.path.basename(file_path_csv)
    pkl_path = os.path.join(os.path.dirname(file_path_csv), pkl_path)
    modelo_pkl = f"arbolpron_{os.path.basename(pkl_path)}"
    modelo_pkl = os.path.join(os.path.dirname(pkl_path), modelo_pkl)
    
    filename = f'arbolpron_{os.path.basename(file_path_csv)}.pkl'

    response = make_response(send_file(modelo_pkl, as_attachment=True))
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"

    return response







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

    datos = [pronosticoBA, caracteristicas]

    with open(modelo_pkl, 'wb') as archivo_pkl:
        pickle.dump(datos, archivo_pkl)

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

    pronosticoBA = pronosticoBA[0]

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

    pronosticoBA = pronosticoBA[0]
    
    caracteristicas = request.json  # Obtener los datos enviados desde el formulario
    
    # Acceder a los valores de las características
    caracteristicas_valores = [float(value) for value in caracteristicas.values()]


    # Realizar la predicción utilizando el modelo pronosticado
    pronostico = pronosticoBA.predict([caracteristicas_valores])

    pronostico = str(round(pronostico[0],5))

    # Devolver el resultado de la predicción como respuesta JSON
    return pronostico


@app.route("/descargar_regforest", methods=['GET'])
def descargar_regforest():
    # Ruta al archivo que se va a descargar
    file_path_csv = session.get('file_path')
    pkl_path = "tree"+os.path.basename(file_path_csv)
    pkl_path = os.path.join(os.path.dirname(file_path_csv), pkl_path)
    modelo_pkl = f"bosquepron_{os.path.basename(pkl_path)}"
    modelo_pkl = os.path.join(os.path.dirname(pkl_path), modelo_pkl)
    
    filename = f'bosquepron_{os.path.basename(file_path_csv)}.pkl'

    response = make_response(send_file(modelo_pkl, as_attachment=True))
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"

    return response











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
    

    y_test_bin = label_binarize(y_test, classes=np.unique(y_train))

    # Predicciones de probabilidades para cada clase
    y_probs = ClasificacionAD.predict_proba(x_test)

    # Calcular la curva ROC y el área bajo la curva para cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    num_classes = len(ClasificacionAD.classes_)

    graphs_path = session.get('path_graphs')
    graph_text = "graphs/" + os.path.splitext(os.path.basename(file_path_csv))[0]+"/"

    filename = f"roc_tree{os.path.basename(pkl_path)}.svg"
    graph_path = os.path.join(graphs_path, filename)

    if num_classes == 2:
        # Problema de clasificación binaria
        fig, ax = plt.subplots()
        roc_display_ad = RocCurveDisplay.from_estimator(ClasificacionAD, x_test, y_test, ax=ax)
        roc_display_ad.figure_.savefig(graph_path, format='svg')
    else:# Graficar las curvas ROC para cada clase
        fig, ax = plt.subplots()
        for class_index in range(num_classes):  # num_classes es el número de clases en tu problema
            fpr[class_index], tpr[class_index], _ = roc_curve(y_test_bin[:, class_index], y_probs[:, class_index])
            roc_auc[class_index] = auc(fpr[class_index], tpr[class_index])
            ax.plot(fpr[class_index], tpr[class_index], label=f'Clase {class_index + 1} (AUC = {roc_auc[class_index]:.2f})')
        ax.set_title('Curva ROC Multiclase')
            # Configurar el diseño del gráfico
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Tasa de Falsos Positivos')
        ax.set_ylabel('Tasa de Verdaderos Positivos')
        ax.legend(loc="lower right")
        fig.savefig(graph_path, format='svg')
    
    roc_graph = url_for('static', filename=graph_text+ filename)
    


    accuracyscore = round((accuracy_score(y_test, y_Pronostico)*100),5)

    criterio = ClasificacionAD.criterion
    varImportance = ClasificacionAD.feature_importances_
    matrizClas = pd.crosstab(y_test.ravel(), y_Pronostico)
    matrizClas = matrizClas.to_json(orient='index')
    

    report = classification_report(y_test, y_Pronostico, output_dict=True)
    report = pd.DataFrame(report).round(5)
    report = report.T

    df = pd.DataFrame({'Real': y_test.flatten(), 'Pronostico': y_Pronostico.flatten()})
    
    if(df.shape[0]>70):
        df = df.drop(df.index[70:])


    plt.figure(figsize=(15,15))  
    plot_tree(ClasificacionAD, feature_names = caracteristicas)
    filename = f"treeclass{os.path.basename(pkl_path)}.png"
    graph_path = os.path.join(graphs_path, filename)
    
    plt.savefig(graph_path)
    tree_graph = url_for('static', filename=graph_text + filename)

    varImportance_df = pd.DataFrame({'Importance': varImportance})
    varImportance_df = varImportance_df.round(5)
    varImportance_json = varImportance_df.to_json(orient='index')


    modelo_pkl = f"arbol_class{os.path.basename(pkl_path)}"
    modelo_pkl = os.path.join(os.path.dirname(pkl_path), modelo_pkl)

    datos = [ClasificacionAD, caracteristicas]

    with open(modelo_pkl, 'wb') as archivo_pkl:
        pickle.dump(datos, archivo_pkl)


    results = {
        'accuracyscore': accuracyscore,
        'criterio': criterio,
        'varImportance': varImportance_json,
        'matrizClass': matrizClas,
        'report': report.to_json(orient='index'),
        'valores': df.to_json(orient='index'),
        'tree_graph': tree_graph,
        'roc_graph': roc_graph,
        'encabezado': ClasificacionAD.classes_.tolist()
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

    pronosticoAD = pronosticoAD[0]
    
    caracteristicas = request.json  # Obtener los datos enviados desde el formulario
    
    # Acceder a los valores de las características
    caracteristicas_valores = [float(value) for value in caracteristicas.values()]

    # Realizar la predicción utilizando el modelo pronosticado
    pronostico = pronosticoAD.predict([caracteristicas_valores])

    pronostico = str(pronostico[0])

    # Devolver el resultado de la predicción como respuesta JSON
    return pronostico


@app.route("/descargar_classtree", methods=['GET'])
def descargar_classtree():
    # Ruta al archivo que se va a descargar
    file_path_csv = session.get('file_path')
    pkl_path = "tree"+os.path.basename(file_path_csv)
    pkl_path = os.path.join(os.path.dirname(file_path_csv), pkl_path)
    modelo_pkl = f"arbol_class{os.path.basename(pkl_path)}"
    modelo_pkl = os.path.join(os.path.dirname(pkl_path), modelo_pkl)
    
    filename = f'arbol_class{os.path.basename(file_path_csv)}.pkl'

    response = make_response(send_file(modelo_pkl, as_attachment=True))
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"

    return response








@app.route('/ClasificationForest', methods=['POST'])
def clasificationForest():

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
    

    
    ClasificacionBA = RandomForestClassifier(n_estimators=n_estimators, 
                                          min_samples_split=min_samples_split, 
                                         min_samples_leaf=min_samples_leaf, 
                                         random_state=random_state)
    ClasificacionBA.fit(x_train, y_train)
    
    y_Pronostico = ClasificacionBA.predict(x_test)
    
    y_test_bin = label_binarize(y_test, classes=np.unique(y_train))

    # Predicciones de probabilidades para cada clase
    y_probs = ClasificacionBA.predict_proba(x_test)

    # Calcular la curva ROC y el área bajo la curva para cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    num_classes = len(ClasificacionBA.classes_)

    graphs_path = session.get('path_graphs')
    graph_text = "graphs/" + os.path.splitext(os.path.basename(file_path_csv))[0]+"/"

    filename = f"roc_tree{os.path.basename(pkl_path)}.svg"
    graph_path = os.path.join(graphs_path, filename)

    if num_classes == 2:
        # Problema de clasificación binaria
        fig, ax = plt.subplots()
        roc_display_ad = RocCurveDisplay.from_estimator(ClasificacionBA, x_test, y_test, ax=ax)
        roc_display_ad.figure_.savefig(graph_path, format='svg')
    else:# Graficar las curvas ROC para cada clase
        fig, ax = plt.subplots()
        for class_index in range(num_classes):  # num_classes es el número de clases en tu problema
            fpr[class_index], tpr[class_index], _ = roc_curve(y_test_bin[:, class_index], y_probs[:, class_index])
            roc_auc[class_index] = auc(fpr[class_index], tpr[class_index])
            ax.plot(fpr[class_index], tpr[class_index], label=f'Clase {class_index + 1} (AUC = {roc_auc[class_index]:.2f})')
        ax.set_title('Curva ROC Multiclase')
            # Configurar el diseño del gráfico
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Tasa de Falsos Positivos')
        ax.set_ylabel('Tasa de Verdaderos Positivos')
        ax.legend(loc="lower right")
        fig.savefig(graph_path, format='svg')
    
    roc_graph = url_for('static', filename=graph_text+ filename)


    accuracyscore = round((accuracy_score(y_test, y_Pronostico)*100),5)

    criterio = ClasificacionBA.criterion
    varImportance = ClasificacionBA.feature_importances_
    matrizClas = pd.crosstab(y_test.ravel(), y_Pronostico)
    matrizClas = matrizClas.to_json(orient='index')
    

    report = classification_report(y_test, y_Pronostico, output_dict=True)
    report = pd.DataFrame(report).round(5)
    report = report.T

    df = pd.DataFrame({'Real': y_test.flatten(), 'Pronostico': y_Pronostico.flatten()})
    if(df.shape[0]>70):
        df = df.drop(df.index[70:])

    df = df.round(5)


    varImportance_df = pd.DataFrame({'Importance': varImportance})
    varImportance_df = varImportance_df.round(5)
    varImportance_json = varImportance_df.to_json(orient='index')


    modelo_pkl = f"forest_class{os.path.basename(pkl_path)}"
    modelo_pkl = os.path.join(os.path.dirname(pkl_path), modelo_pkl)

    datos = [ClasificacionBA, caracteristicas]

    with open(modelo_pkl, 'wb') as archivo_pkl:
        pickle.dump(datos, archivo_pkl)
    

    results = {
        'accuracyscore': accuracyscore,
        'criterio': criterio,
        'varImportance': varImportance_json,
        'matrizClass': matrizClas,
        'report': report.to_json(orient='index'),
        'valores': df.to_json(orient='index'),
        'roc_graph': roc_graph,
        'encabezado': ClasificacionBA.classes_.tolist()
    }

    # Enviar los resultados como una respuesta JSON
    return jsonify(results)




@app.route('/PronosticarClasificationF', methods=['POST'])
def pronosticarClasificationF():
    file_path_csv = session.get('file_path')
    pkl_path = "tree"+os.path.basename(file_path_csv)
    pkl_path = os.path.join(os.path.dirname(file_path_csv), pkl_path)
    modelo_pkl = f"forest_class{os.path.basename(pkl_path)}"
    modelo_pkl = os.path.join(os.path.dirname(pkl_path), modelo_pkl)

    pronosticoBA = []


    # Cargar el modelo desde el archivo pkl
    with open(modelo_pkl, 'rb') as archivo_pkl:
        pronosticoBA = pickle.load(archivo_pkl)

    pronosticoBA = pronosticoBA[0]
    
    caracteristicas = request.json  # Obtener los datos enviados desde el formulario
    
    # Acceder a los valores de las características
    caracteristicas_valores = [float(value) for value in caracteristicas.values()]
    

    # Realizar la predicción utilizando el modelo pronosticado
    pronostico = pronosticoBA.predict([caracteristicas_valores])

    pronostico = str(pronostico[0])

    # Devolver el resultado de la predicción como respuesta JSON
    return pronostico




@app.route("/descargar_classforest", methods=['GET'])
def descargar_classforest():
    # Ruta al archivo que se va a descargar
    file_path_csv = session.get('file_path')
    pkl_path = "tree"+os.path.basename(file_path_csv)
    pkl_path = os.path.join(os.path.dirname(file_path_csv), pkl_path)
    modelo_pkl = f"forest_class{os.path.basename(pkl_path)}"
    modelo_pkl = os.path.join(os.path.dirname(pkl_path), modelo_pkl)
    
    filename = f'forest_class{os.path.basename(file_path_csv)}.pkl'

    response = make_response(send_file(modelo_pkl, as_attachment=True))
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"

    return response








@app.route('/K-means', methods=['POST'])
def k_meansMethod():
    componentes = request.json

    file_path_csv = session.get('file_path')
    pkl_path = "tree"+os.path.basename(file_path_csv)
    pkl_path = os.path.join(os.path.dirname(file_path_csv), pkl_path)
    pkl = pd.read_pickle(pkl_path)
    pkl = pkl.filter(items=componentes)

    estandarizar = StandardScaler()

    MEstandarizada = estandarizar.fit_transform(pkl)

    SSE = []
    for i in range(2, 10):
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(MEstandarizada)
        SSE.append(km.inertia_)
    
    kl = KneeLocator(range(2, 10), SSE, curve="convex", direction="decreasing")

    # Crear dataframe con los valores de SSE y k
    data = {'k': range(2, 10), 'SSE': SSE}
    df = pd.DataFrame(data)

    # Graficar SSE en función de k usando Plotly Express
    fig = px.line(df, x='k', y='SSE', markers=True)
    # Trazar el punto de codo en la gráfica
    fig.add_vline(x=kl.elbow, line_dash='dash', line_color='red', annotation_text="Punto de codo", annotation_position='top left')

    fig.update_layout(title='Elbow Method', xaxis_title='Cantidad de clusters *k*', 
                      yaxis_title='SSE', showlegend=True, height=600,width=1200,
                      legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))

    graphs_path = session.get('path_graphs')
    graph_text = "graphs/" + os.path.splitext(os.path.basename(file_path_csv))[0]+"/"

    filename = f"elbowM{os.path.basename(pkl_path)}.html"
    graph_path = os.path.join(graphs_path, filename)
    fig.write_html(graph_path)
    elbow_graph = url_for('static', filename=graph_text+ filename)



    #K-MEANS
    MParticional = KMeans(n_clusters=kl.elbow, random_state=0).fit(MEstandarizada)
    MParticional.predict(MEstandarizada)
    pkl['cluster'] = MParticional.labels_

    #Cantidad de elementos en los clusters
    numeroCluster = pkl.groupby(['cluster'])['cluster'].count()

    centroides = pkl.groupby('cluster').mean()


    # Lista de colores para asignar a los clusters
    colores = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan', 'yellow']

    fig = go.Figure()

    # Scatter plot de las muestras
    for cluster in range(MParticional.n_clusters):
        puntos_cluster = MParticional.labels_ == cluster
        fig.add_trace(go.Scatter3d(
            x=MEstandarizada[puntos_cluster, 0],
            y=MEstandarizada[puntos_cluster, 1],
            z=MEstandarizada[puntos_cluster, 2],
            mode='markers',
            marker=dict(
                color=colores[cluster],
                size=6
            ),
            name=f'Cluster {cluster + 1}'  # Etiqueta del cluster
        ))

    # Scatter plot de los centroides
    fig.add_trace(go.Scatter3d(
        x=MParticional.cluster_centers_[:, 0],
        y=MParticional.cluster_centers_[:, 1],
        z=MParticional.cluster_centers_[:, 2],
        mode='markers',
        marker=dict(
            color=colores[:MParticional.n_clusters],  # Seleccionar colores para los centroides según el número de clusters
            size=12,
            symbol='circle'
        ),
        name='Centroides'
    ))

    # Configuración del diseño del gráfico
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Componente 1'),
            yaxis=dict(title='Componente 2'),
            zaxis=dict(title='Componente 3')
        ),
        legend=dict(
            title='Clusters'
        )
    )

    # Guardar el gráfico
    filename = f"3dclusters{os.path.basename(pkl_path)}.html"
    graph_path = os.path.join(graphs_path, filename)
    fig.write_html(graph_path)
    cluster_graph = url_for('static', filename=graph_text+ filename)


    modelo_pkl = f"K-Means{os.path.basename(pkl_path)}.csv"
    modelo_pkl = os.path.join(os.path.dirname(pkl_path), modelo_pkl)
    pkl.to_csv(modelo_pkl)

    results = {
        'elbow_graph': elbow_graph,
        'elbow': str(kl.elbow),
        'numeroCluster': numeroCluster.to_json(),
        'centroides': centroides.to_json(orient='index'),
        'cluster_graph': cluster_graph,
    }

    return jsonify(results)



@app.route("/descargar_kmeans", methods=['GET'])
def descargar_kmeans():
    # Ruta al archivo que se va a descargar
    file_path_csv = session.get('file_path')
    pkl_path = "tree"+os.path.basename(file_path_csv)
    pkl_path = os.path.join(os.path.dirname(file_path_csv), pkl_path)
    modelo_pkl = f"K-Means{os.path.basename(pkl_path)}.csv"
    modelo_pkl = os.path.join(os.path.dirname(pkl_path), modelo_pkl)
    
    filename = f'KMeans_{os.path.basename(file_path_csv)}.csv'

    response = make_response(send_file(modelo_pkl, as_attachment=True))
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"

    return response










@app.route("/upload_pkl", methods=['POST'])
def upload_pkl():
    uploaded_file = request.files['file-pkl']
    
    if ( uploaded_file.filename.endswith('.pkl')):
        timestamp = datetime.now().strftime('%H%M%S')
        model_file_name = f"{uploaded_file.filename}_{timestamp}.pkl"
        file_path = os.path.join(UPLOAD_FOLDER, model_file_name)
        print(file_path)
        uploaded_file.save(file_path)

        session['model_uploaded'] = True
        session['model_path'] = file_path

        return redirect(url_for("predicts"))
    else:
        session['model_incorrecto'] = True
        return redirect(url_for('home'))


@app.route("/predicts")
def predicts():
    return render_template("predicts.html")



@app.route("/obtenerFeatures", methods=['POST'])
def obtenerFeatures():
    model_path = session.get('model_path')
    with open(model_path, 'rb') as archivo_pkl:
        features = pickle.load(archivo_pkl)

    features = features[1]

    return jsonify(features)



@app.route("/modelPrediction", methods=['POST'])
def modelPrediction():

    model_path = session.get('model_path')
    caracteristicas = request.json  # Obtener los datos enviados desde el formulario

    with open(model_path, 'rb') as archivo_pkl:
        model = pickle.load(archivo_pkl)

    model = model[0]

    caracteristicas_valores = [float(value) for value in caracteristicas.values()]

    prediccion = model.predict([caracteristicas_valores])

    prediccion = str(prediccion[0])

    return prediccion