new Vue({
    el: '#app',
    data: {
        features: [],  // Variable para almacenar las características recibidas
        pronostico: 0,
        formulario: {}
    },

    methods: {
        obtenerFeatures(){
            axios.post('/obtenerFeatures')
            .then(response => {
                this.features = response.data;
            })
            .catch(error => {
                console.log(error);
            }); 
        },
        crearPronostico() {
            axios.post('/modelPrediction', this.formulario)
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
                window.location.href = "/";
          },
      

    },
    mounted(){
        this.obtenerFeatures();
    }
});
