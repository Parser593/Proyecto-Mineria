import os
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import pandas as pd              
import numpy as np              
import matplotlib.pyplot as plt   
import seaborn as sns           
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler  

plt.switch_backend('Agg')

app = Flask(__name__)
secret_key = os.urandom(16).hex()
app.secret_key = secret_key

UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')  # Ruta de la carpeta de carga

# Verificar si la carpeta de carga existe, y crearla si no existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/")
def home():
    return render_template("index.html")

#Recibir archivo
@app.route("/upload", methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if(uploaded_file.filename.endswith('.csv') and uploaded_file.mimetype == "text/csv"):
        
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
        
        uploaded_file.save(file_path)

        session['file_uploaded'] = True
        session['file_path'] = file_path

        return render_template("project.html")
    else:
        session['archivo_incorrecto'] = True
        return redirect(url_for('home'))

@app.route("/generate_graph", methods=['POST'])
def EDA():
    if 'graficos_generados' in session:
        eda_paths = session['eda_paths']
        nombreCSV = os.path.basename(session.get('file_path'))
    else:
        file_path = session.get('file_path')
        if file_path:
            csv_file = pd.read_csv(file_path)
            #csv_file = csv_file.dropna()
            csv_file.hist(figsize=(10,10), xrot=45)
            nombreCSV = os.path.basename(file_path)

            filename = f"hist1{nombreCSV}.svg"
            hist_path = os.path.join(app.static_folder, "graphs", filename)
            plt.savefig(hist_path)
            eda_paths = [url_for('static', filename="graphs/" + filename)]

            
            plt.clf()
            i = 2
            plt.figure(figsize=(5,4))
            for col in csv_file.select_dtypes(include='object'):
                if csv_file[col].nunique() < 10:
                    sns.countplot(y=col, data=csv_file)

                    filename = f"hist{i}{nombreCSV}.svg"
                    hist_path = os.path.join(app.static_folder, "graphs", filename)
                    i += 1
                    plt.savefig(hist_path)
                    eda_paths.append(url_for('static', filename="graphs/" + filename))

                    
            plt.clf()        
            plt.figure(figsize=(10, 4))
            numeric_columns = csv_file.select_dtypes(include='number')
            maskInf = np.triu(numeric_columns.corr())  
            sns.heatmap(numeric_columns.corr(), cmap='RdBu_r', annot=True, mask=maskInf)

            filename = f"corr{nombreCSV}.svg"
            hist_path = os.path.join(app.static_folder, "graphs", filename)
            plt.savefig(hist_path)
            eda_paths.append(url_for('static', filename="graphs/" + filename))

            i+=1


            q1 = csv_file.quantile(0.25)
            q3 = csv_file.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            mask = (csv_file >= lower_bound) & (csv_file <= upper_bound)
            csv_file_filter = csv_file[mask]

            csv_file_filter.hist(figsize=(10,10), xrot=45)
            filename = f"finhist{nombreCSV}.svg"
            hist_path = os.path.join(app.static_folder, "graphs", filename)
            plt.savefig(hist_path)
            eda_paths.append(url_for('static', filename="graphs/" + filename))
            plt.clf()   

            session['eda_paths'] = eda_paths
            session['graficos_generados'] = True
        else:
            return 'Error'

    return json.dumps({'graficoPaths': eda_paths, 'nombreCSV':nombreCSV })



@app.route("/correlation", methods=['POST'])
def mPCA():
    if 'pca' in session:
        file_path = session.get('file_path')
        pca_paths = session.get('pca_paths')

        nombreCSV = os.path.basename(file_path)

        csvfile = pd.read_csv(file_path)
        csvfile = csvfile.dropna()
        Estandarizar = StandardScaler()                                  # Se instancia el objeto StandardScaler o MinMaxScaler 
        NuevaMatriz = csvfile.select_dtypes(include='number')
        MEstandarizada = Estandarizar.fit_transform(NuevaMatriz)         # Se calculan la media y desviación para cada variable, y se escalan los datos

        #Calculo de Matriz de Covarianzas y calculo de componentes
        pca = PCA(n_components=None)
        pca.fit(MEstandarizada)
        components = pca.components_
        varianza = pca.explained_variance_ratio_
        
    else:
        file_path = session.get('file_path')
        if file_path:
            nombreCSV = os.path.basename(session.get('file_path'))
            #correlacion
            csvfile = pd.read_csv(file_path)
            csvfile = csvfile.dropna()
            corrPcsv = csvfile.corr(method='pearson')
            
            #heatmap correlacion
            plt.clf()        
            plt.figure(figsize=(12,6))
            matInf = np.triu(corrPcsv)
            sns.heatmap(corrPcsv, cmap='RdBu_r', annot=True, mask=matInf)

            filename = f"pcacorr{nombreCSV}.svg"
            hist_path = os.path.join(app.static_folder, "graphs", filename)
            plt.savefig(hist_path)
            pca_paths = [(url_for('static', filename="graphs/" + filename))]

            #Estandarización
            Estandarizar = StandardScaler()                                  # Se instancia el objeto StandardScaler o MinMaxScaler 
            NuevaMatriz = csvfile.select_dtypes(include='number')
            MEstandarizada = Estandarizar.fit_transform(NuevaMatriz)         # Se calculan la media y desviación para cada variable, y se escalan los datos

            #Calculo de Matriz de Covarianzas y calculo de componentes
            pca = PCA(n_components=None)
            pca.fit(MEstandarizada)
            components = pca.components_

            varianza = pca.explained_variance_ratio_


            #Grafica varianza
            plt.clf()        
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('Número de Componentes')
            plt.ylabel('Varianza Acumulada')
            plt.grid()
            
            filename = f"pcavar{nombreCSV}.svg"
            hist_path = os.path.join(app.static_folder, "graphs", filename)
            plt.savefig(hist_path)
            pca_paths.append(url_for('static', filename="graphs/" + filename))
            
            #Proporcion de relevancias
            cargasComponentes = pd.DataFrame(abs(pca.components_), columns=NuevaMatriz.columns)

            session['pca_paths'] = pca_paths
            session['corrPcsv'] = corrPcsv.to_json(orient='columns')
            session['cargaComponentes'] = cargasComponentes.to_json(orient='columns')
            session['pca']=True

    return jsonify({
    'pca_paths': pca_paths,
    'corrPcsv': json.loads(session.get('corrPcsv', '{}')),
    'varianza': varianza.tolist(),
    'components': components.tolist(),
    'cargasComponentes': json.loads(session.get('cargaComponentes', '{}')),
})



@app.route("/procesarComponentes", methods=['POST'])
def m2PCA():
    datos = request.json
    comp_seleccionados = datos['compSeleccionados']
    file_path = session.get('file_path')
    csvfile = pd.read_csv(file_path)
    csvfile = csvfile.dropna()
    csvpca = csvfile.filter(items=comp_seleccionados)
    csvpca = csvpca.head(15)
    csvpca_list = csvpca.to_dict(orient='records')

    return jsonify({
        'csvpca': csvpca_list
    })
    


#['Longtitude', 'Lattitude', 'Postcode', 'Propertycount', 'Rooms', 'YearBuilt']