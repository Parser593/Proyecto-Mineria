const dropZone = document.querySelector('#drop-zone');
const fileInput = document.querySelector('#my-file');
const icon = document.getElementById('cloud');
const iconfile = document.getElementById("icsv");
const textdropZone = document.getElementById("mUpload")
const form = document.getElementById("csv-form");
iconfile.style.display = "none";


dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dover");
});

dropZone.addEventListener("click", () => {
    fileInput.click();
});


dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dover");

    const file = e.dataTransfer.files[0];

    if (file.name.endsWith(".csv")) {
        iconfile.style.display = "inline";
        icon.style.display = "none";
        textdropZone.textContent= "Se ha cargado el archivo "+file.name;
        fileInput.files = e.dataTransfer.files;
    } else {
        alert("El archivo debe ser un CSV");
    }



});

fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file && file.name.endsWith('.csv')) {
        iconfile.style.display = "inline";
        icon.style.display = "none";
        textdropZone.textContent= "Se ha cargado el archivo "+file.name;
    } else {
      alert('Selecciona un archivo CSV');
    }
  });


  window.addEventListener("load", () => {
    form.reset(); // Limpiar el formulario al cargar la página
  });
  