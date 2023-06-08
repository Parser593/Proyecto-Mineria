const dropZone = document.querySelector('#drop-zone');
const fileInput = document.querySelector('#my-file');
const icon = document.getElementById('cloud');
const iconfile = document.getElementById("icsv");
const textdropZone = document.getElementById("mUpload")
const form = document.getElementById("csv-form");
iconfile.style.display = "none";


const dropZonepkl = document.querySelector('#drop-zone-pkl');
const fileInputpkl = document.querySelector('#my-file-pkl');
const iconpkl = document.getElementById('cloud-pkl');
const iconfilepkl = document.getElementById("ipkl");
const textdropZonepkl = document.getElementById("mUploadpkl")
const formpkl = document.getElementById("pkl-form");
iconfilepkl.style.display = "none";



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
  




dropZonepkl.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZonepkl.classList.add("dover");
});

dropZonepkl.addEventListener("dragleave", () => {
    dropZonepkl.classList.remove("dover");
});

dropZonepkl.addEventListener("click", () => {
    fileInputpkl.click();
});


dropZonepkl.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZonepkl.classList.remove("dover");

    const file = e.dataTransfer.files[0];

    if (file.name.endsWith(".pkl")) {
        iconfilepkl.style.display = "inline";
        iconpkl.style.display = "none";
        textdropZonepkl.textContent= "Se ha cargado el archivo "+file.name;
        fileInputpkl.files = e.dataTransfer.files;
    } else {
        alert("El archivo debe ser un CSV");
    }



});

fileInputpkl.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file && file.name.endsWith('.pkl')) {
        iconfilepkl.style.display = "inline";
        iconpkl.style.display = "none";
        textdropZonepkl.textContent= "Se ha cargado el archivo "+file.name;
    } else {
      alert('Selecciona un archivo PKL');
    }
  });


  window.addEventListener("load", () => {
    formpkl.reset(); // Limpiar el formulario al cargar la página
  });
  