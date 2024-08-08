import pandas as pd
import matplotlib.pyplot as plt
import os

def generar_graficas(csv_path):
    # Verifica si el archivo existe
    if not os.path.isfile(csv_path):
        print(f"Error: El archivo {csv_path} no se encuentra.")
        return

    try:
        # Lee el archivo CSV
        df = pd.read_csv(csv_path, delimiter=';')
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return

    # Crea la carpeta para guardar las gráficas si no existe
    if not os.path.exists('graphs'):
        os.makedirs('graphs')

    # Títulos fijos para las gráficas
    titulo_loss = "Pérdida"
    titulo_binary_accuracy = "Precisión"
    titulo_iou = "Intersección sobre la unión"
    titulo_dice_coef = "Coeficiente de Dice"

    # Función para crear una gráfica y guardarla
    def crear_grafica(x, y, x_label, y_label, titulo, nombre_archivo):
        plt.figure(figsize=(10, 6))
        plt.plot(df[x], label=x)
        plt.plot(df[y], label=y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(titulo)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'graphs/{nombre_archivo}.png')
        plt.close()

    # Crear y guardar las gráficas
    try:
        crear_grafica('loss', 'val_loss', 'Epoch', 'Loss', titulo_loss, 'loss_vs_val_loss')
        crear_grafica('binary_accuracy', 'val_binary_accuracy', 'Epoch', 'Binary Accuracy', titulo_binary_accuracy, 'binary_accuracy_vs_val_binary_accuracy')
        crear_grafica('iou', 'val_iou', 'Epoch', 'IOU', titulo_iou, 'iou_vs_val_iou')
        crear_grafica('dice_coef', 'val_dice_coef', 'Epoch', 'Dice Coefficient', titulo_dice_coef, 'dice_coef_vs_val_dice_coef')
    except Exception as e:
        print(f"Error al crear las gráficas: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Uso: python script.py <ruta_del_archivo_csv>")
    else:
        generar_graficas(sys.argv[1])
