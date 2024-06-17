import os
    
if __name__ == '__main__':
    
    print(f'Entrenando modelos de detección...')

    os.system(f'python .{os.sep}scripts{os.sep}detection{os.sep}train{os.sep}train_all.py')
    
    print(f'Modelos de detección entrenados...')
    
    print(f'Entrenando modelos de clasificación...')
    
    print(f'Modelos de clasificación entrenados...')
    
    print(f'Entrenando modelos de segmentación...')
    
    print(f'Modelos de segmentación entrenados...')