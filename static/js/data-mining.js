document.getElementById('formComponentes').addEventListener

new Vue({
  el: '#app',
  data: {
    nombreCSV: '', // Aquí se almacenará el nombre del archivo CSV
    activeTab: 'tab1',
    eda_paths: [],
    pca_paths: [],
    corrPcsv: {},
    varianza: [],
    componentes: [],
    cargasComponentes: {},
    stringArray: [],
    matriz: [],
    matriz2: [],
    matriz3: [],
    componentesSeleccionados: [],
    maxComponentes: 0,
    csvpca: [],
    mostrarTabla: false
  },
  computed: {
    graficosConParrafosVinculados() {
      return this.eda_paths.map((ruta, indice) => {
        let parrafo = '';

        if (indice < 2) {
          if (indice == 0) {
            parrafo = ' Distribución de variables numéricas';
          } else {
            parrafo = ' Histograma de Variables Categóricas';
          }
        } else if (ruta.includes('corr')) {
          parrafo = 'Identificación de relaciones entre pares variables';
        } else if (ruta.includes('finhist')) {
          parrafo = 'Eliminación de datos Atípicos';
        }

        return {
          imagen: ruta,
          parrafo: parrafo
        };
      });
    }
  },

  methods: {
    changeTab(tab) {
      this.activeTab = tab;
      if (tab === 'tab2') {
        this.calculateCorrelation();
      }
    },
    generateGraph() {
      // Llamar a la función en Flask que genera los gráficos
      axios.post('/generate_graph')
        .then(response => {
          // Obtener las rutas de los gráficos desde la respuesta de Flask
          this.eda_paths = response.data.graficoPaths;
          this.nombreCSV = response.data.nombreCSV;

        })
        .catch(error => {
          // Manejar el error si ocurre
          console.error(error);
        });
    },
    calculateCorrelation() {
      axios.post('/correlation')
        .then(response => {

          let i = -1;
          this.corrPcsv = JSON.parse(JSON.stringify(response.data.corrPcsv));
          this.pca_paths = response.data.pca_paths;
          this.varianza = response.data.varianza;
          this.componentes = response.data.components;
          this.cargasComponentes = JSON.parse(JSON.stringify(response.data.cargasComponentes));
          this.stringArray = [];
          for (const key in this.corrPcsv) {
            if (this.corrPcsv.hasOwnProperty(key)) {
              const obj = this.corrPcsv[key];
              i += 1;
              let j = 0;
              this.matriz[i] = [];
              this.stringArray.push(key);
              for (const prop in obj) {
                if (obj.hasOwnProperty(prop)) {
                  const value = obj[prop];
                  this.matriz[i][j] = value;
                  j += 1;
                }
              }
            }
          }

          i = -1
          for (const key in this.cargasComponentes) {
            if (this.corrPcsv.hasOwnProperty(key)) {
              const obj = this.cargasComponentes[key];
              i += 1;
              let j = 0;
              this.matriz2[i] = [];
              for (const prop in obj) {
                if (obj.hasOwnProperty(prop)) {
                  const value = obj[prop];
                  this.matriz2[i][j] = value;
                  j += 1;
                }
              }
            }
          }

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
          this.csvpca = JSON.parse(JSON.stringify(response.data.csvpca));
          console.table(this.csvpca)
          let i=-1;
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
/*           for (const key in this.csvpca) {
            if (this.csvpca.hasOwnProperty(key)) {
              const obj = this.csvpca[key];
              for (const prop in obj) {
                if (obj.hasOwnProperty(prop)) {
                  const value = obj[prop];
                  console.log(` ${value}`);
                }
              }
            }
          } */
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
    //this.calculateCorrelation();
  }
});
