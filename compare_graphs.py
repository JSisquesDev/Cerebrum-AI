import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def generar_graficas_comparativas(csv_path_a, csv_path_b, nombre_a, nombre_b):
    # Verifica si los archivos existen
    if not os.path.isfile(csv_path_a):
        print(f"Error: El archivo {csv_path_a} no se encuentra.")
        return
    if not os.path.isfile(csv_path_b):
        print(f"Error: El archivo {csv_path_b} no se encuentra.")
        return

    try:
        # Lee los archivos CSV
        df_a = pd.read_csv(csv_path_a, delimiter=';')
        df_b = pd.read_csv(csv_path_b, delimiter=';')
    except Exception as e:
        print(f"Error al leer los archivos CSV: {e}")
        return

    # Obtiene el número de épocas de cada archivo
    num_epochs_a = len(df_a)
    num_epochs_b = len(df_b)

    # Determina el número mínimo de épocas
    min_epochs = min(num_epochs_a, num_epochs_b)

    # Trunca los DataFrames a la longitud mínima
    df_a = df_a.iloc[:min_epochs]
    df_b = df_b.iloc[:min_epochs]

    # Crea la carpeta para guardar las gráficas si no existe
    if not os.path.exists('graphs'):
        os.makedirs('graphs')

    # Títulos y etiquetas para las gráficas comparativas
    graficas = {
        'loss': {'titulo': 'Pérdida', 'x_label': 'Epoch', 'y_label': 'Loss'},
        'val_loss': {'titulo': 'Val. Pérdida', 'x_label': 'Epoch', 'y_label': 'val_Loss'},
        'binary_accuracy': {'titulo': 'Precisión', 'x_label': 'Epoch', 'y_label': 'Binary Accuracy'},
        'val_binary_accuracy': {'titulo': 'Val. Precisión', 'x_label': 'Epoch', 'y_label': 'val_Binary Accuracy'},
        'iou': {'titulo': 'Intersección sobre la Unión', 'x_label': 'Epoch', 'y_label': 'IOU'},
        'val_iou': {'titulo': 'Val. Intersección sobre la Unión', 'x_label': 'Epoch', 'y_label': 'val_IOU'},
        'dice_coef': {'titulo': 'Coeficiente de Dice', 'x_label': 'Epoch', 'y_label': 'Dice Coefficient'},
        'val_dice_coef': {'titulo': 'Val. Coeficiente de Dice', 'x_label': 'Epoch', 'y_label': 'val_Dice Coefficient'}
    }

    # Función para crear una gráfica y guardarla
    def crear_grafica(x_a, x_b, grafica_key, nombre_archivo):
        plt.figure(figsize=(10, 6))
        plt.plot(df_a[x_a], label=f'{nombre_a}', color='blue')
        plt.plot(df_b[x_b], label=f'{nombre_b}', color='orange')
        plt.xlabel(graficas[grafica_key]['x_label'])
        plt.ylabel(graficas[grafica_key]['y_label'])
        plt.title(graficas[grafica_key]['titulo'])
        plt.legend()
        plt.grid(True)
        plt.savefig(f'graphs/{nombre_archivo}.png')
        plt.close()

    # Crear y guardar las gráficas comparativas
    try:
        crear_grafica('loss', 'loss', 'loss', f'loss_vs_loss_{nombre_a}_vs_{nombre_b}')
        crear_grafica('val_loss', 'val_loss', 'val_loss', f'val_loss_vs_val_loss_{nombre_a}_vs_{nombre_b}')
        crear_grafica('binary_accuracy', 'binary_accuracy', 'binary_accuracy', f'binary_accuracy_vs_binary_accuracy_{nombre_a}_vs_{nombre_b}')
        crear_grafica('val_binary_accuracy', 'val_binary_accuracy', 'val_binary_accuracy', f'val_binary_accuracy_vs_val_binary_accuracy_{nombre_a}_vs_{nombre_b}')
        crear_grafica('iou', 'iou', 'iou', f'iou_vs_iou_{nombre_a}_vs_{nombre_b}')
        crear_grafica('val_iou', 'val_iou', 'val_iou', f'val_iou_vs_val_iou_{nombre_a}_vs_{nombre_b}')
        crear_grafica('dice_coef', 'dice_coef', 'dice_coef', f'dice_coef_vs_dice_coef_{nombre_a}_vs_{nombre_b}')
        crear_grafica('val_dice_coef', 'val_dice_coef', 'val_dice_coef', f'val_dice_coef_vs_val_dice_coef_{nombre_a}_vs_{nombre_b}')
    except Exception as e:
        print(f"Error al crear las gráficas comparativas: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Uso: python script.py <ruta_del_archivo_csv_A> <nombre_A> <ruta_del_archivo_csv_B> <nombre_B>")
    else:
        generar_graficas_comparativas(sys.argv[1], sys.argv[3], sys.argv[2], sys.argv[4])
