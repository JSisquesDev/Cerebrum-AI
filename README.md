# Cerebum - AI

Descripción del proyecto

## Índice

- [Cerebum - AI](#cerebum---ai)
  - [Índice](#índice)
  - [Introducción](#introducción)
  - [Instalación](#instalación)
  - [Cómo Usarlo](#cómo-usarlo)
  - [Autor](#autor)
  - [Licencia](#licencia)

## Introducción

La detección temprana y precisa de tumores cerebrales es crucial para mejorar los resultados del tratamiento y la supervivencia de los pacientes. Este proyecto utiliza algoritmos avanzados de aprendizaje profundo para analizar imágenes de resonancia magnética (MRI) y proporcionar resultados precisos y útiles para los profesionales de la salud.

## Instalación

Para ejecutar este proyecto en tu máquina local, sigue los siguientes pasos:

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/tu-usuario/tu-proyecto.git
   ```
2. **Navega al directorio del proyecto**:
   ```bash
   cd tu-proyecto
   ```
3. **Crea un entorno virtual**:
   ```bash
   python -m venv env
   ```
4. **Activa el entorno virtual**:
   - En Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - En macOS y Linux:
     ```bash
     source env/bin/activate
     ```
5. **Instala las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

## Cómo Usarlo

Para usar el modelo y realizar predicciones, sigue estos pasos:

1. **Preprocesa las imágenes**:

   - Asegúrate de que las imágenes de resonancia magnética (MRI) estén en el formato adecuado.
   - Utiliza el script `preprocess.py` para preprocesar las imágenes:
     ```bash
     python preprocess.py --input-dir ruta/a/las/imagenes --output-dir ruta/a/las/imagenes_procesadas
     ```

2. **Entrena el modelo**:

   - Utiliza el script `train.py` para entrenar el modelo con los datos preprocesados:
     ```bash
     python train.py --data-dir ruta/a/las/imagenes_procesadas --output-dir ruta/a/los/modelos
     ```

3. **Realiza predicciones**:
   - Utiliza el script `predict.py` para realizar predicciones en nuevas imágenes:
     ```bash
     python predict.py --model-dir ruta/a/los/modelos --input-dir ruta/a/nuevas/imagenes --output-dir ruta/a/resultados
     ```

## Autor

**Javier Plaza Sisqués**

- [GitHub](https://github.com/tu-usuario)
- [LinkedIn](https://www.linkedin.com/in/tu-perfil/)

## Licencia

Este proyecto está licenciado bajo la Licencia GNU GENERAL PUBLIC LICENSE v3. Consulta el archivo [LICENSE](LICENSE) para más detalles.
