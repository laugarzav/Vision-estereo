""" Tarea 11: Sparse 3D reconstruction using stereo vision.

Fecha: 16/05/2024

Alumno: Laura Sofía Garza Villarreal    600650
         
"Doy mi palabra de que he realizado esta actividad con integridad académica"

Ejecución del código: python estereo_vision.py --l_img left_infrared_image.png --r_img right_infrared_image.png --dist 100
"""
# Importar librerias
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros de calibración
rectified_cx = 635.709
rectified_cy = 370.88
rectified_width = 1280
rectified_height = 720
f = 648.52
B = 94.926

# Variable para guardar las coordenadas de las selecciones
seleccion = []

def imgs():
    """
    Función para cargas las imagenes izquierda y derecha necesarias para el procesamiento.
    Returns:
    tuple: tupla que contiene las imágenes izquierda, derecha, asi como la distancia entre
    la cámara y el objeto no calibrado.
    """
    parser = argparse.ArgumentParser(description="Visión estéreo")
    # Agregar imagen izquierda
    parser.add_argument(
        '--l_img',
        type=str,
        required=True,
        help="Imagen izquierda"
    )
    # Agregar imagen derecha
    parser.add_argument(
        '--r_img',
        type=str,
        required=True,
        help="Imagen derecha"
    )
    # Agregar distancie entre la cámara y el objeto no calibrado
    parser.add_argument(
        '--distancia',
        type=float,
        required=True,
        help="Distancia entre las camara y el objeto no calibrado"
    )
    
    args = parser.parse_args()

    # Cargar imágenes y convertirla a escala de grises
    img_left = cv2.imread(args.l_img, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(args.r_img, cv2.IMREAD_GRAYSCALE)

    # Asegurarse de que ambas imágenes esten cargadas correctamente
    if img_left is None or img_right is None:
        raise ValueError("Error al cargar las imágenes. Asegúrate de que las rutas son correctas.")
    
    return img_left, img_right, args.distancia

def calcular_centro_imagen(width, height, cx, cy):
    """
    Esta función calcula las coordenadas del centro de la imagen
    Args:
        width (int): ancho de la imagen
        height (int): alto de la imagen
        cx (float): coordenada x del centro de la imagen
        cy (float): coordenada y del centro de la imagen
    Returns:
        centro_x: coordenada x del centro de la imagen
        centro_Y: coordenada y del centro de la imagen

    """
    centro_x = cx
    centro_y = cy
    return centro_x, centro_y

def ant_coords_izq(uL, vL, cx, cy):
    """
    Esta función calcula las coordenadas seleccionadas por el usuario con respecto al centro de la imagen
    Args:
        uL (int): Coordenada u del pixel de la imagen izquierda
        vL (int): Coordenada v del pixel en la imagen izquierda
        cx (float): Coordenada x del centro de la imagen
        cy (float): Coordenada y del cetro de la imagen
    Returns:
        ucL: coordenada x de imagen izq respecto al centro de la imagen
        vcL: coordenada y de imagen izq respecto al centro de la imagen
    """
    ucL = uL - cx
    vcL = vL - cy
    return ucL, vcL

def ant_coords_der(uR, vR, cx, cy):
    """
    Esta función calcula las coordenadas seleccionadas por el usuario con respecto al centro de la imagen
    Args:
        uR (int): Coordenada u del pixel de la imagen derecha
        vR (int): Coordenada v del pixel en la imagen derecha
        cx (float): Coordenada x del centro de la imagen
        cy (float): Coordenada y del cetro de la imagen
    Returns:
        ucR: coordenada x de imagen derecha respecto al centro de la imagen
        vcR: coordenada y de imagen derecha respecto al centro de la imagen
    """
    ucR = uR - cx
    vcR = vR - cy
    return ucR, vcR

def reconstruccion_3d(uL, vL, uR, vR):
    """
    Esta función calcula las coordenadas tridimensionales (X, Y, Z) a partir
    de las coordenadas de los pixeles
    Args:
        uL (int): Coordenada u del pixel de la imagen izquierda
        vL (int): Coordenada v del pixel en la imagen izquierda
        uR (int): Coordenada u del pixel de la imagen derecha
        vR (int): Coordenada v del pixel en la imagen derecha
    Returns:
        X: coordenada X
        Y: coordenada Y
        Z: coordenada Z
    """
    # Calcular las coordenadas respecto al centro
    ucL, vcL = ant_coords_izq(uL, vL, rectified_cx, rectified_cy)
    ucR, vcR = ant_coords_der(uR, vR, rectified_cx, rectified_cy)

    # Calcular disparidad
    disparidad = ucL - ucR

    # Calcular Z
    # f = distancia focal
    # B = baseline
    Z = ((f * B) / disparidad)

    # Calcular Y 
    Y = (vcR) * (Z / f)

    # Calcular X
    X = (ucL) * (Z / f)

    return X, Y, Z

def visualizacion_3d(puntos_3d):
    """
    Visualiza los puntos tridimensionales en una gráfica 3D
    Args:
        puntos_3d (nnumpy.ndarray): arreglo numpy que contiene las coordenadas
        X, Y, Z de los pixeles seleccionados.
    Returns:
        None
    """
    # Crea una figura vacia
    fig = plt.figure()
    # Agregar un subplot tridimensional
    ax = fig.add_subplot(111, projection='3d')
    
    # Extraer coordenadas 
    X = puntos_3d[:, 0]
    Y = puntos_3d[:, 1]
    Z = puntos_3d[:, 2]

    # Graficar puntos 3D y líneas que los unen
    ax.scatter(X, Y, Z, c='r', marker='o')
    ax.plot(X, Y, Z, color='b')

    # Indican las etiquetas de ls ejes
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    
    # Ajustar los límites de los ejes para mejorar la visualización
    ax.set_xlim(min(X), max(X))
    ax.set_ylim(min(Y), max(Y))
    #ax.set_zlim(min(Z), max(Z))

    # Configurar ejes con proporciones iguales
    plt.axis('equal')
    plt.show()

def mouse_event(event, x, y, flags, param):
    """
    Manejo de events del mouse para la selección de pixeles en las imágenes

    Args:
        event (int): Tipo de evento del mouse
        x (int): Coordenada x del pixel seleccionado por el mouse
        y (int): Coordenada y del pixel seleccionado por el mouse
        flags (int): Bandera de evento del mouse
        param (tuple): Tupla con imágen derecha, imagen izquierda, baseline, centros de la imagen
    Returns:
        None
    """
    global seleccion
    
    # Clic izquierdo del mouse
    if event == cv2.EVENT_LBUTTONDOWN:
        img_left, img_right, baseline, centro_left, centro_right = param
        # Coordenadas del píxel seleccionado en la imagen izquierda
        uL, vL = x, y
        # Misma coordenada vertical (v) en ambas imágenes
        vR = vL
        # Coordenada x en la imagen derecha (desplazada por la distancia entre cámaras)
        uR = int(uL - (baseline - 15))
        # Calcular coordenadas 3D
        X, Y, Z = reconstruccion_3d(uL, vL, uR, vR)
        
        # Añadir las coordenadas a la lista de selecciones
        seleccion.append((X, Y, Z))
        
        # Dibujar un círculo en las coordenadas seleccionadas
        cv2.circle(img_left, (uL, vL), 3, (255, 0, 0), -1)
        cv2.circle(img_right, (uR, vR), 3, (255, 0, 0), -1)
        
        # Mostrar imágenes actualizadas
        cv2.imshow('Imagen Izquierda', img_left)
        cv2.imshow('Imagen Derecha', img_right)

        # Verificar si se han seleccionado al menos 30 píxeles
        if len(seleccion) >= 30:
            print("Se han seleccionado al menos 30 píxeles.")
            print(seleccion)
            cv2.destroyAllWindows()
            
            # Convertir la lista de selecciones a un arreglo numpy
            puntos_3d = np.array(seleccion)
            
            # Visualizar reconstrucción 3D
            visualizacion_3d(puntos_3d)

def main():
    """
    Función principal que ejecuta el programa realizando lo siguiente:
    -Carga las imágenes izquierda y derecha, y la distancia entre las camaras y el objeto
    -Convierte las imagenes a color para dibujar los circulos de los pixeles seleccionados
    -Calcula el centro de las imagenes
    -Muestra las imagenes y espera la seleccion del usuario
    -Configura los eventos del mouse para que se lleve a cabo la seleccion de los pixeles
    -Espera que se pulse una tecla y luego cierra la ventana
    """
    try:
        img_left, img_right, baseline = imgs()

        # Convertir imágenes a color para dibujar círculos
        img_left = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)

        # Calcular centro de las imágenes
        centro_left = calcular_centro_imagen(rectified_width, rectified_height, rectified_cx, rectified_cy)
        centro_right = calcular_centro_imagen(rectified_width, rectified_height, rectified_cx, rectified_cy)

        # Mostrar imágenes y esperar la selección del usuario
        cv2.imshow('Imagen Izquierda', img_left)
        cv2.imshow('Imagen Derecha', img_right)

        # Configurar eventos del mouse
        cv2.setMouseCallback('Imagen Izquierda', mouse_event, param=(img_left, img_right, B, centro_left, centro_right))

        # Esperar que se pulse una tecla y luego cerrar ventanas
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()