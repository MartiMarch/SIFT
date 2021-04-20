"""
 Aalgoritmo SIFT

 El algoritmo SIFT busca los puntos clave y sus respectivos descriptores. La ventaja del algoritmo SIFT es
 que la imágen se puede rotar y redimensionar.

"""
import cv2
from datetime import datetime

# La cámara
camara = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Umbrales utilizados para detectar la varaicion de puntos calve
umbralSuperior = 1.10
umbralInferior = 0.90

# Cantidad de puntos clave
nEsquinas = 0

# Se crea el objeto SIFT
sift = cv2.SIFT_create()

# Permite reinicar los umbrales cada cierto tiempo
now = 0
tiempoActual = 0

while True:
    _, imagen = camara.read()

    # Imágen transformada a gris
    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Se generan los puntos clave, el segundoa parámetro es una máscara para buscar solo en cierta parte de la imagen
    kp = sift.detect(imagenGris, None)

    # Se dibujan los puntos clave
    imagenPuntos = cv2.drawKeypoints(imagenGris, kp, imagen)

    # Cada 15 minutos se renuevan los puntos de referencia usados para comparar
    nEsquinas = len(kp)
    now = datetime.now()
    if now.timestamp() - tiempoActual > 9000000 or tiempoActual == 0:
        tiempoActual = now.timestamp()
        puntosRelevantes = nEsquinas

    # Para saber si se ha producido movimiento lo que se hace es comparar si la cantidad de puntos entra dentro de los umbrales.
    if nEsquinas > (puntosRelevantes * umbralSuperior) or nEsquinas < (puntosRelevantes * umbralInferior):
        print("Movimiento detectado")

    # Se muestra en una ventana las imágenes
    cv2.imshow("SIFT - color", imagen)

    # Pulsar 'q' para salir
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break