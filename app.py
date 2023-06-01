import os
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from datetime import datetime
import shutil
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
    #PJKL NOT NULL
    file_path = file_path_pkl+"notnull.pkl"
    if os.path.exists(graphs_path):
        shutil.rmtree(graphs_path)

    if os.path.exists(file_path_pkl):
        os.remove(file_path_pkl)
    if os.path.exists(file_path):
        os.remove(file_path)

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
            # csv_file = csv_file.dropna()
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

            # sns.heatmap(numeric_columns.corr(), cmap='RdBu_r', annot=True, mask=maskInf)
            fig = go.Figure(data=go.Heatmap(z=triangle_lower,
                                            x=nombres_columnas, y=nombres_columnas,
                                            colorscale='RdBu_r',
                                            texttemplate="%{z}"))
            fig.update_layout(
                width=1200,
                height=600
            )

            # Personaliza el trazo de la matriz de correlación
            # Ajusta el tamaño de fuente aquí
            fig.update_traces(textfont={'size': 10})

            filename = f"corr{nombreCSV}.html"
            hist_path = os.path.join(graphs_path, filename)
            fig.write_html(hist_path)
            eda_paths.append(url_for('static', filename=graph_text+ filename))

            # i+=1
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
    if 'csv_notnull' in session:
        file_path = session.get('csv_notnull')
        corrPcsv = pd.read_pickle(file_path)
        corrPcsv = corrPcsv.corr(method='pearson')
        pca_path = session.get('corr_pca')
    else:
        file_path = session.get('file_path')
        # correlacion
        csvfile = pd.read_pickle(file_path)
        csvfile = csvfile.dropna()
        graph_text = "graphs/" + os.path.splitext(os.path.basename(file_path))[0] + "/"
        graphs_path = session.get('path_graphs')

        # Generar un nombre único para el archivo pickle
        pickle_file_name = f"{os.path.basename(file_path)}notnull.pkl"

        # Guardar el DataFrame en un archivo pickle
        pickle_file_path = os.path.join(UPLOAD_FOLDER, pickle_file_name)
        csvfile.to_pickle(pickle_file_path)
        session['csv_notnull'] = pickle_file_path
        
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
            height=600
        )

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
    if 'csv_notnull' in session:
        file_path = session.get('csv_notnull')
        csvfile = pd.read_pickle(file_path)
    else:
        csvfile = pd.read_pickle(file_path_csv)
        csvfile = csvfile.dropna()

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
    csvfile = csvfile.dropna()
    csvpca = csvfile.filter(items=comp_seleccionados)
    csvpca = csvpca.head(15)
    csvpca_list = csvpca.to_json(orient='index')

    return jsonify({
        'csvpca': csvpca_list
    })



