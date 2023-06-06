function parseCSV2matrix(describedCSV, flag) {
  let i = -1;
  const matrizaux = [];
  const stringArray = [];
  const encabezado = [];

  for (const key in describedCSV) {
    if (describedCSV.hasOwnProperty(key)) {
      const obj = describedCSV[key];
      i += 1;
      let j = 0;
      matrizaux[i] = [];
      if (flag === 3) {
        encabezado.push(key);
      }
      for (const prop in obj) {
        if (obj.hasOwnProperty(prop)) {
          const value = obj[prop];
          matrizaux[i][j] = value;
          if (flag > 1) {
            stringArray.push(prop);
          }
          j += 1;
        }
      }
    }
  }

  if (flag === 1) {
    return matrizaux
  } else if (flag === 2) {
    return { matrizaux, stringArray };
  } else if(flag === 3){
    return { matrizaux, stringArray, encabezado };
  }
}


new Vue({

  el: '#app',
  data: {
    nombreCSV: '', // Aquí se almacenará el nombre del archivo CSV
    activeTab: 'tab1',
    activesubTab: 'tab1',
    head15_csv: [],
    eda_paths: [],
    pca_var: "",
    histdis: [],
    histcat: [],
    histfindis: [],
    varianza: [],
    corr: '',
    componentes: [],
    variables: [],
    variablesTotales: [],
    stringArray3: [],
    mheadeda: [],
    corr_path: "",
    matriz: [],
    matriz2: [],
    matriz3: [],
    matriz4: [],
    matriz5: [],
    matrizVarImportance: [],
    matrizValores: [],
    componentesSeleccionados: [],
    componentesSeleccionados2: [],
    numComponentes: 0,
    mostrarTabla: false,
    mostrarDataArbol: false,
    mostrarDataForest: false,
    mostrarDataArbol2: false,

    varDependiente: '',
    graphVarDep: '',
    caracteristicas: [],
    max_depth: null,
    min_samples_split: null,
    min_samples_leaf: null,
    random_state: null,
    r2score: null,
    criterio: null,
    mae: null,
    mse: null,
    rmse: null,
    test_graph: null,
    tree_graph: null,
    formulario: {},

    pronostico: 0,
    n_estimators: 0,
    min_samples_split2: 0,
    min_samples_leaf2: 0,
    random_state2: 0,
    caracteristicas2: [],
    varDependiente2: null,
    r2scoreForest: null,
    criterioForest: null,
    maeForest: null,
    mseForest: null,
    rmseForest: null,
    test_graphForest: null,
    forest_graph: null,
    matrizVarImportanceF: [],
    matrizValoresF: [],
    numberTree: 0,
    mostrarGraphF: false,
    pronosticoF: 0,
    formulario2: {},

    varDependienteClas: null,
    max_depthClas: null,
    min_samples_splitClas: null,
    min_samples_leafClas: null,
    random_stateClas: null,
    caracteristicasClas: [],

    accuracyscore: null,
    criterioClas: null,
    reporte: null,
    matrizConf: null,
    tree_graphClas: null,
    formularioClas: {},
    matrizVarImportanceClas: [],
    matrizValoresClas: [],
    pronosticoClas: 0,
    encabezadoF: {},
    encabezadoF2: {},
    filasF: {},
    filasF2: {}
  },


  methods: {
    changesubTab(tab) {
      this.activesubTab = tab;
      if (tab === 'tab1') {

      }
    },

    changeTab(tab) {
      this.activeTab = tab;
      if (tab === 'tab2') {
        this.obtainCorrelation();
        this.calculateCovarComp();
      } else if ((tab === 'tab3' || tab === 'tab4') && this.componentesSeleccionados2.length > 1) {
        this.getPCASummary();
      }
      if (tab === 'tab4') {
        this.activesubTab = 'tab3';
      }
    },
    confirmClearSession() {
      swal({
        title: "Confirmar",
        text: "¿Estás seguro de que desea salir? Si sale deberá cargar de nuevo el archvo.",
        icon: "warning",
        buttons: ["Cancelar", "Aceptar"],
        dangerMode: true,
      }).then((confirmed) => {
        if (confirmed) {
          this.clearSession();
        }
      });
    },
    clearSession() {
      // Realizar una solicitud AJAX para limpiar las variables de sesión y redirigir al inicio
      axios.get('/clear_session')
        .then(() => {
          // Redirigir al inicio
          window.location.href = "/";
        })
        .catch(error => {
          console.error(error);
        });
    },




    getGraphClass(index, arrayLength) {
      index = index + 1;
      if (index % 2 === 0 && index === arrayLength) {
        return ''; // Sin clase adicional para elementos pares
      } else if (index === arrayLength) {
        return 'centered-item'; // Clase para el último elemento impar
      } else {
        return ''; // Sin clase adicional para otros casos
      }
    },

    getheadEDA() {
      axios.post('/get_edahead')
        .then(response => {
          const head15csv = JSON.parse(response.data);

          const result = parseCSV2matrix(head15csv, 2);
          this.head15_csv = result.matrizaux;
          this.variablesTotales = result.stringArray;

          const indice = this.variablesTotales.length / 15;
          this.variablesTotales.splice(indice)
        })
        .catch(error => {
          console.log(error)
        });
    },
    getEDASummary() {
      axios.post('/eda_summ')
        .then(response => {
          this.described_csv = JSON.parse(response.data);
          let i = -1;
          for (const key in this.described_csv) {
            if (this.described_csv.hasOwnProperty(key)) {
              const obj = this.described_csv[key];
              i += 1;
              let j = 0;
              this.matriz4[i] = [];
              this.variables.push(key);
              for (const prop in obj) {
                if (obj.hasOwnProperty(prop)) {
                  const value = obj[prop];
                  this.matriz4[i][j] = value;
                  this.stringArray3.push(prop);
                  j += 1;
                }
              }
            }
          }

          const indice = this.stringArray3.length / this.variables.length;
          this.stringArray3.splice(indice)
        })
        .catch(error => {
          window.location.href = '/';
          console.error(error);
        });
    },
    generateGraph() {
      // Llamar a la función en Flask que genera los gráficos
      axios.post('/generate_graph')
        .then(response => {
          // Obtener las rutas de los gráficos desde la respuesta de Flask
          this.eda_paths = response.data.graficoPaths;
          this.nombreCSV = response.data.nombreCSV;

          this.eda_paths.forEach(ruta => {
            if (ruta.includes('histd')) {
              //parrafo = ' Distribución de variables numéricas';
              this.histdis.push(ruta);
            } else if (ruta.includes('histcat')) {
              //parrafo = ' Histograma de Variables Categóricas';
              this.histcat.push(ruta);
            } else if (ruta.includes('finhist')) {
              //parrafo = 'Eliminación de datos Atípicos';
              this.histfindis.push(ruta);
            } else {
              this.corr = ruta;
            }
          });

        })
        .catch(error => {
          // Manejar el error si ocurre
          console.error(error);
        });
    },





    obtainCorrelation() {
      axios.post('/pca_correlation')
        .then(response => {
          const corrPcsv = JSON.parse(response.data.corrPcsv);
          this.corr_path = response.data.pca_path;
          const matriz = parseCSV2matrix(corrPcsv, 1)
          this.matriz = matriz
        })
        .catch(error => {
          console.log(error)
        });
    },
    calculateCovarComp() {
      axios.post('/covarComp')
        .then(response => {

          this.pca_var = response.data.pca_var;
          this.varianza = response.data.varianza;
          this.componentes = response.data.components;
          const cargasComponentes = JSON.parse(response.data.cargasComponentes);
          const matriz2 = parseCSV2matrix(cargasComponentes, 1)
          this.matriz2 = matriz2

        })
        .catch(error => {
          console.error(error);
        });
    },
    enviarComponentes() {
      const componentesSeleccionados = this.componentesSeleccionados;

      axios.post('/procesarComponentes', {
        compSeleccionados: componentesSeleccionados
      })
        .then(response => {
          this.matriz3 = []
          this.mostrarTabla = false;
          const csvpca = JSON.parse(response.data.csvpca);

          const result = parseCSV2matrix(csvpca, 1);
          this.matriz3 = result;
          this.componentesSeleccionados2 = this.componentesSeleccionados;

          this.mostrarTabla = true;
          this.crearGraficaTree();
          swal({
            icon: 'success',
            title: '¡Éxito!',
            text: 'Desplacese hacia abajo para observar los resultados.',
          });
        })
        .catch(error => {
          console.error(error);
        });
    },




    getPCASummary() {
      axios.post('/treeSum')
        .then(response => {
          this.matriz5 = [];
          //described_pca = JSON.parse(response.data);
          this.matriz5 = parseCSV2matrix(JSON.parse(response.data), 1)

        })
        .catch(error => {
          console.error(error);
        });
    },

    crearGraficaTree() {
      axios.post('/grafica_Arboles', {
      })
        .then(response => {
          this.graphVarDep = response.data

        })
        .catch(error => {
          console.error(error);
        });
    },






    correrArbolDecision1() {
      this.mostrarDataArbol = false;
      if (this.caracteristicas.includes(this.varDependiente)) {
        swal({
          icon: 'error',
          title: '¡Error!',
          text: 'La variable dependiente no puede estar seleccionada como característica',
        });
      } else {
        // Enviar los datos utilizando Axios
        axios.post('/RegressionDecisionTree', {
          varDependiente: this.varDependiente,
          caracteristicas: this.caracteristicas,
          max_depth: this.max_depth,
          min_samples_split: this.min_samples_split,
          min_samples_leaf: this.min_samples_leaf,
          random_state: this.random_state,
        })
          .then(response => {
            const { r2score, criterio, varImportance, mae, mse, rmse, valores, test_graph, tree_graph } = response.data;
            this.mae = mae;
            this.mse = mse;
            this.rmse = rmse;
            this.r2score = r2score;
            this.criterio = criterio;
            this.test_graph = test_graph;
            this.tree_graph = tree_graph;

            this.matrizVarImportance = parseCSV2matrix(JSON.parse(varImportance), 1)
            this.matrizValores = parseCSV2matrix(JSON.parse(valores), 1)

            this.mostrarDataArbol = true;
            swal({
              icon: 'success',
              title: '¡Éxito!',
              text: 'Desplacese hacia abajo para observar los resultados.',
            });

          })
          .catch(error => {
            // Manejar el error si ocurre
            console.error(error);
          });
      }

    },

    crearPronostico() {
      axios.post('/Pronosticar', this.formulario)
        .then(response => {
          this.pronostico = response.data;
          swal({
            icon: 'success',
            title: '¡Éxito!',
            text: 'Caluclo de Pronostico realizado.',
          });
        })
        .catch(error => {
          console.log(error);
        });
    },




    correrBosque1() {
      this.mostrarDataForest = false;
      if (this.caracteristicas2.includes(this.varDependiente2)) {
        swal({
          icon: 'error',
          title: '¡Error!',
          text: 'La variable dependiente no puede estar seleccionada como característica',
        });
      } else {
        // Enviar los datos utilizando Axios
        axios.post('/RegressionRandomForest', {
          varDependiente: this.varDependiente2,
          caracteristicas: this.caracteristicas2,
          n_estimators: this.n_estimators,
          min_samples_split: this.min_samples_split2,
          min_samples_leaf: this.min_samples_leaf2,
          random_state: this.random_state2,
        })
          .then(response => {
            const { r2score, criterio, varImportance, mae, mse, rmse, valores, test_graph } = response.data;
            this.maeForest = mae;
            this.mseForest = mse;
            this.rmseForest = rmse;
            this.r2scoreForest = r2score;
            this.criterioForest = criterio;
            this.test_graphForest = test_graph;

            this.matrizVarImportanceF = parseCSV2matrix(JSON.parse(varImportance), 1)
            this.matrizValoresF = parseCSV2matrix(JSON.parse(valores), 1)

            this.mostrarDataForest = true;
            swal({
              icon: 'success',
              title: '¡Éxito!',
              text: 'Desplacese hacia abajo para observar los resultados.',
            });

          })
          .catch(error => {
            // Manejar el error si ocurre
            console.error(error);
          });
      }
    },

    CrearGraphTree() {
      this.mostrarGraphF= false;
      axios.post('/GraphTreeForest', {
        caracteristicas: this.caracteristicas2,
        numeroArbol: this.numberTree
        
      })
        .then(response => {
          this.forest_graph = response.data;
          this.mostrarGraphF= true;
        })
        .catch(error => {
          console.log(error);
        });
    },

    crearPronosticoForest1() {
      axios.post('/PronosticarForest', this.formulario2)
        .then(response => {
          this.pronosticoF = response.data;
          swal({
            icon: 'success',
            title: '¡Éxito!',
            text: 'Caluclo de Pronostico realizado.',
          });
        })
        .catch(error => {
          console.log(error);
        });
    },
    
    


    correrArbolDecision2() {
      this.mostrarDataArbol2 = false;
      if (this.caracteristicasClas.includes(this.varDependienteClas)) {
        swal({
          icon: 'error',
          title: '¡Error!',
          text: 'La variable dependiente no puede estar seleccionada como característica',
        });
      } else {
        // Enviar los datos utilizando Axios
        axios.post('/ClasificationDecisionTree', {
          varDependiente: this.varDependienteClas,
          caracteristicas: this.caracteristicasClas,
          max_depth: this.max_depthClas,
          min_samples_split: this.min_samples_splitClas,
          min_samples_leaf: this.min_samples_leafClas,
          random_state: this.random_stateClas,
        })
          .then(response => {
            const { accuracyscore, criterio, varImportance, matrizClass, report, valores, tree_graph } = response.data;
            this.matrizConf = JSON.parse(matrizClass);
            this.reporte = report;
            this.accuracyscore = accuracyscore;
            this.criterioClas = criterio;
            this.tree_graphClas = tree_graph;

            this.matrizVarImportanceClas = parseCSV2matrix(JSON.parse(varImportance), 1)
            this.matrizValoresClas = parseCSV2matrix(JSON.parse(valores), 1)

            this.mostrarDataArbol2 = true;

            const result = parseCSV2matrix(JSON.parse(this.reporte), 3);
            this.reporte = result.matrizaux;
            this.encabezadoF = result.stringArray;
            this.filasF = result.encabezado;
            
            const indice = this.encabezadoF.length / this.filasF.length;
            this.encabezadoF.splice(indice)

            const result2 = parseCSV2matrix(this.matrizConf, 2);
            this.matrizConf = result2.matrizaux;
            this.encabezadoF2 = result2.stringArray;
            this.encabezadoF2.splice(2)

            swal({
              icon: 'success',
              title: '¡Éxito!',
              text: 'Desplacese hacia abajo para observar los resultados.',
            });

          })
          .catch(error => {
            // Manejar el error si ocurre
            console.error(error);
          });
      }

    },
    crearPronosticoClasificacion() {
      axios.post('/PronosticarClasifiaction', this.formularioClas)
        .then(response => {
          this.pronosticoClas = response.data;
          swal({
            icon: 'success',
            title: '¡Éxito!',
            text: 'Caluclo de Pronostico realizado.',
          });
        })
        .catch(error => {
          console.log(error);
        });
    },

  },
  mounted() {
    this.getheadEDA();
    this.generateGraph();
    this.getEDASummary();
  }
});
