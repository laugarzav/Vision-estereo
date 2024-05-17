import argparse
import cv2
import numpy as np

# Parámetros de calibración
rectified_cx = 635.709
rectified_cy = 370.88
rectified_width = 1280
rectified_height = 720
f = 648.52
B = 94.926

def imgs():
    parser = argparse.ArgumentParser(description="Visión estéreo")
    parser.add_argument(
        '--l_img',
        type=str,
        required=True,
        help="Imagen izquierda"
    )
    parser.add_argument(
        '--r_img',
        type=str,
        required=True,
        help="Imagen derecha"
    )
    parser.add_argument(
        '--distancia',
        type=float,
        required=True,
        help="Distancia entre las camara y el objeto no calibrado"
    )
    
    args = parser.parse_args()

    # Cargar imágenes
    img_left = cv2.imread(args.l_img, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(args.r_img, cv2.IMREAD_GRAYSCALE)

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
    ucL = uL - cx
    vcL = vL - cy
    return ucL, vcL

def ant_coords_der(uR, vR, cx, cy):
    ucR = uR - cx
    vcR = vR - cy
    return ucR, vcR

def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img_left, img_right, baseline, centro_left, centro_right = param
        
        # Coordenadas del píxel seleccionado en la imagen izquierda
        uL, vL = x, y
        
        # Misma coordenada vertical (v) en ambas imágenes
        vR = vL
        
        # Coordenada x en la imagen derecha (desplazada por la distancia entre cámaras)
        uR = int(uL - (baseline-15))

        # Calcular las coordenadas respecto al centro
        ucL, vcL = ant_coords_izq(uL, vL, centro_left[0], centro_left[1])
        ucR, vcR = ant_coords_der(uR, vR, centro_right[0], centro_right[1])

        # Calcular disparidad
        disparidad = ucL - ucR

        # Calcular Z
        Z = ((f * B) / disparidad)

        # Calcular Y 
        Y = (vcR) * (Z / f)

        # Calcular X
        X = (ucL) * (Z / f)

        # Imprimir las coordenadas de ambos píxeles
        print(f"Centro de la imagen izquierda: {centro_left}")
        print(f"Centro de la imagen derecha: {centro_right}")
        print(f"Coordenadas del píxel seleccionado en la imagen izquierda: ({uL}, {vL})")
        print(f"Coordenadas del píxel correspondiente en la imagen derecha: ({uR}, {vR})")
        print(f"Coordenadas del píxel de la imagen izquierda respecto al centro: ({ucL}, {vcL})")
        print(f"Coordenadas del píxel de la imagen derecha respecto al centro: ({ucR}, {vcR})")
        print(f"Disparidad= ({disparidad}) px")
        print(f"X = ({X}) mm")
        print(f"Y = ({Y}) mm")
        print(f"Z = ({Z}) mm")
        print(f"La coordenada del pixel seleccionado es: ({X}) mm, ({Y}) mm, ({Z}) mm")

def main():
    try:
        img_left, img_right, baseline = imgs()

        # Calcular centro de las imágenes
        centro_left = calcular_centro_imagen(rectified_width, rectified_height, rectified_cx, rectified_cy)
        centro_right = calcular_centro_imagen(rectified_width, rectified_height, rectified_cx, rectified_cy)

        # Mostrar imágenes y esperar la selección del usuario
        cv2.imshow('Imagen Izquierda', img_left)
        cv2.imshow('Imagen Derecha', img_right)

        # Configurar eventos del mouse
        cv2.setMouseCallback('Imagen Izquierda', mouse_event, param=(img_left, img_right, baseline, centro_left, centro_right))
        

        # Esperar que se pulse una tecla y luego cerrar ventanas
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
