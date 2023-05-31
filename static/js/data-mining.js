function parseCSV2matrix(describedCSV, flag) {
  let i = -1;
  const matrizaux = [];
  const stringArray = [];

  for (const key in describedCSV) {
    if (describedCSV.hasOwnProperty(key)) {
      const obj = describedCSV[key];
      i += 1;
      let j = 0;
      matrizaux[i] = [];
      for (const prop in obj) {
        if (obj.hasOwnProperty(prop)) {
          const value = obj[prop];
          matrizaux[i][j] = value;
          stringArray.push(prop);
          j += 1;
        }
      }
    }
  }

  if (flag === 1) {
    return matrizaux
  } else if (flag === 2) {
    return { matrizaux, stringArray };
  }
}


new Vue({

  el: '#app',
  data: {
    loading: true,
    nombreCSV: '', // Aquí se almacenará el nombre del archivo CSV
    activeTab: 'tab1',
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
    componentesSeleccionados: [],
    maxComponentes: 0,
    csvpca: [],
    mostrarTabla: false
  },


  methods: {

    changeTab(tab) {
      this.activeTab = tab;
      if (tab === 'tab2') {
        this.obtainCorrelation();
        this.calculateCovarComp();
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
          // Manejar el error si ocurre
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
          console.log(this.pca_paths)
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
          this.csvpca = JSON.parse(response.data.csvpca);
          console.table(this.csvpca)
          let i = -1;
          for (const key in this.csvpca) {
            if (this.csvpca.hasOwnProperty(key)) {
              const obj = this.csvpca[key];
              i += 1;
              let j = 0;
              this.matriz3[i] = [];
              for (const prop in obj) {
                if (obj.hasOwnProperty(prop)) {
                  const value = obj[prop];
                  this.matriz3[i][j] = value;
                  j += 1;
                }
              }
            }
          }
          this.mostrarTabla = true;
          console.table(this.matriz3);
        })
        .catch(error => {
          console.error(error);
        });
    }
  },
  mounted() {
    this.generateGraph();
    this.getheadEDA();
    this.getEDASummary();
  }
});
