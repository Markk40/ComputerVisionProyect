import cv2
import numpy as np
from picamera2 import Picamera2
import time
from password import detect_figures  # Importamos la función para detección de figuras

last_color = None  # Inicializamos la variable global last_color
container_names = {"yellow": "plastico", "green":"vidrio", "blue":"carton", "marron":"residuos organicos"}

def find_object(picam):
    global last_color
    """Busca un objeto basado en su color utilizando las funciones del archivo password.py."""
    track_window = None
    searching_for_object = True
    object_detected_time = None

    while True:
        frame = picam.capture_array()

        # Mostrar el mensaje de "CUADRADO {last_color} DESAPARECIDO" si last_color no es None
        if last_color is not None:
            cv2.putText(frame, f"OBJETO ENVIADO AL CONTENEDOR {container_names[last_color].upper()}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Llamamos a la función detect_figures para obtener el color de la figura detectada
        detected_color = detect_figures(frame)
        
        if detected_color:
            print(f"Objeto detectado de color {detected_color}")
            
            # Usamos la función cv2.boundingRect para crear una track window alrededor del objeto
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_bound, upper_bound = get_color_bounds(detected_color)  # Función auxiliar para obtener límites de color
            mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
            result = cv2.bitwise_and(frame, frame, mask=mask)
            gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:  # Si es un rectángulo
                    x, y, w, h = cv2.boundingRect(approx)
                    track_window = (x, y, w, h)
                    object_detected_time = time.time()  # Marcar el tiempo de detección
                    searching_for_object = False  # Detenemos la búsqueda
                    break

        cv2.imshow("Buscando objeto", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return None, None

        if not searching_for_object and track_window is not None:
            break

    cv2.destroyAllWindows()
    return track_window, object_detected_time

def track_object(picam, track_window, object_detected_time):
    global last_color
    while True:
        frame = picam.capture_array()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Llamamos a la función detect_figures para obtener el color de la figura detectada
        detected_color = detect_figures(frame)
        
        # Actualizamos last_color con el color detectado
        if detected_color:
            last_color = detected_color

        if detected_color:
            print(f"Detectando objeto de color {detected_color} durante el seguimiento.")

            lower_bound, upper_bound = get_color_bounds(detected_color)
            mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
            result = cv2.bitwise_and(frame, frame, mask=mask)
            gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:  # Si es un rectángulo
                    x, y, w, h = cv2.boundingRect(approx)
                    track_window = (x, y, w, h)

        # Si tenemos un track_window, significa que ya estamos rastreando el objeto
        if track_window is not None:
            x, y, w, h = track_window
            cx, cy = x + w // 2, y + h // 2  # Centro de masa del rectángulo

            # Comprobamos si han pasado 7 segundos sin detectar movimiento
            if time.time() - object_detected_time > 7:
                print("Han pasado 7 segundos sin detectar movimiento, buscando un nuevo objeto...")
                track_window = None  # Reiniciamos el rastreo
                object_detected_time = None  # Reiniciamos el tiempo de detección
                break

            # Dibujar el centro de masa y el rectángulo
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Centro real
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Dibujo del rectángulo

        cv2.imshow("Rastreo del objeto", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    return track_window, object_detected_time

def get_color_bounds(color):
    """Devuelve los límites de color en el espacio HSV para la detección de colores."""
    color_ranges = {
        "yellow": ((19, 85, 96), (47, 255, 255)),
        "green": ((35, 50, 70), (85, 255, 255)),
        "blue": ((75, 67, 73), (145, 255, 177)),
        "brown": ((0, 34, 63), (25, 200, 131))
    }
    return color_ranges.get(color, ((0, 0, 0), (0, 0, 0)))  # Devuelve un rango predeterminado si no se encuentra

def track(picam):
    """Función principal para buscar y rastrear un objeto basado en su color y forma."""
    track_window = None
    object_detected_time = None

    while True:
        # Si no tenemos un objeto rastreado, iniciamos la búsqueda de uno nuevo
        if track_window is None:
            print("Buscando un nuevo objeto...")
            track_window, object_detected_time = find_object(picam)

        if track_window is not None:
            # Si encontramos un objeto, lo seguimos
            track_window, object_detected_time = track_object(picam, track_window, object_detected_time)

        # Si el objeto desaparece del marco o no se detecta por un tiempo, volvemos a buscar uno nuevo
        if track_window is None:
            print("Buscando un nuevo objeto...")
            continue  # Vuelve a buscar un nuevo objeto

    picam.stop()  # Detener la cámara cuando se sale

if __name__ == "__main__":
    picam = Picamera2()
    picam.preview_configuration.main.size=(1280, 720)
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    track(picam)
