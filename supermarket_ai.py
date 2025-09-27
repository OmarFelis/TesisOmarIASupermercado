# Sistema de Supermercado Inteligente
import math
import cv2
import os
from ultralytics import YOLO

class SupermarketAI:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        # Modelo YOLO más potente para mejor detección
        self.model = YOLO('yolov8m.pt')  # Medium model (mejor que nano)
        
        # Cargar logo de la universidad
        self.logo = self.cargar_logo()

        # Productos de supermercado con precios en soles
        self.productos_supermercado = {
            # Frutas y verduras
            'banana': 3.50, 'apple': 5.00, 'orange': 4.50, 'broccoli': 8.00, 'carrot': 2.50,
            # Bebidas y contenedores
            'bottle': 2.50, 'wine glass': 15.00, 'cup': 8.00,
            # Utensilios de cocina
            'fork': 12.00, 'knife': 18.00, 'spoon': 10.00, 'bowl': 25.00,
            # Comida preparada
            'sandwich': 12.00, 'hot dog': 8.00, 'pizza': 35.00, 'donut': 6.00, 'cake': 45.00,
            # Productos de higiene
            'toothbrush': 15.00,
            # Otros productos
            'book': 35.00, 'scissors': 20.00, 'cell phone': 1200.00, 'mouse': 65.00, 'keyboard': 120.00
        }

        # Variables del carrito
        self.carrito = []
        self.total = 0
    
    def cargar_logo(self):
        # Buscar archivos de logo en la carpeta assets
        logo_files = ['assets/logoupn.PNG', 'assets/logo_universidad.png', 'assets/logo.png']
        
        for logo_path in logo_files:
            if os.path.exists(logo_path):
                print(f"Cargando logo: {logo_path}")
                logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
                if logo is not None:
                    # Redimensionar logo a 100x100 píxeles
                    logo = cv2.resize(logo, (100, 100))
                    print("Logo cargado exitosamente")
                    return logo
        
        print("No se encontró logo en assets/")
        return None
    
    def mostrar_logo(self, frame):
        if self.logo is not None:
            # Posición del logo (esquina superior derecha)
            y_offset = 10
            x_offset = frame.shape[1] - self.logo.shape[1] - 10
            
            # Si el logo tiene canal alpha (transparencia)
            if self.logo.shape[2] == 4:
                # Extraer canales RGB y Alpha
                logo_rgb = self.logo[:, :, :3]
                alpha = self.logo[:, :, 3] / 255.0
                
                # Aplicar transparencia
                for c in range(0, 3):
                    frame[y_offset:y_offset+self.logo.shape[0], 
                          x_offset:x_offset+self.logo.shape[1], c] = \
                        (alpha * logo_rgb[:, :, c] + 
                         (1 - alpha) * frame[y_offset:y_offset+self.logo.shape[0], 
                                            x_offset:x_offset+self.logo.shape[1], c])
            else:
                # Logo sin transparencia
                frame[y_offset:y_offset+self.logo.shape[0], 
                      x_offset:x_offset+self.logo.shape[1]] = self.logo
        return frame

    def dibujar_rectangulo(self, img, color, x1, y1, x2, y2):
        return cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    def dibujar_texto(self, img, texto, x, y, color=(0, 255, 0), tamaño=0.7, grosor=2):
        # Fondo negro para mejor legibilidad
        (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, tamaño, grosor)
        cv2.rectangle(img, (x, y - h - 10), (x + w, y), (0, 0, 0), -1)
        cv2.putText(img, texto, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, tamaño, color, grosor)
        return img

    def agregar_al_carrito(self, producto):
        if producto in self.productos_supermercado:
            # Verificar si el producto ya está en el carrito
            for item in self.carrito:
                if item['nombre'] == producto:
                    item['cantidad'] += 1
                    return
            
            # Si no está, agregarlo
            self.carrito.append({
                'nombre': producto,
                'precio': self.productos_supermercado[producto],
                'cantidad': 1
            })

    def calcular_total(self):
        self.total = sum(item['precio'] * item['cantidad'] for item in self.carrito)

    def dibujar_area_deteccion(self, frame):
        # Definir área de detección (centro de la pantalla)
        h, w = frame.shape[:2]
        x1, y1 = int(w * 0.25), int(h * 0.25)  # 25% desde arriba-izquierda
        x2, y2 = int(w * 0.75), int(h * 0.75)  # 75% hasta abajo-derecha
        
        # Dibujar rectángulo de área de detección
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Amarillo
        
        # Texto instructivo
        cv2.putText(frame, "AREA DE DETECCION", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Coloca el producto aqui", (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame, (x1, y1, x2, y2)
    
    def esta_en_area_deteccion(self, x1, y1, x2, y2, area_deteccion):
        # Verificar si el objeto está dentro del área de detección
        ax1, ay1, ax2, ay2 = area_deteccion
        centro_x = (x1 + x2) // 2
        centro_y = (y1 + y2) // 2
        
        return ax1 <= centro_x <= ax2 and ay1 <= centro_y <= ay2

    def mostrar_carrito(self, frame):
        y_pos = 50
        cv2.putText(frame, "CARRITO DE COMPRAS", (900, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        for item in self.carrito:
            texto = f"{item['nombre']}: S/{item['precio']:.2f} x{item['cantidad']}"
            cv2.putText(frame, texto, (900, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos += 25

        # Mostrar total
        cv2.putText(frame, f"TOTAL: S/{self.total:.2f}", (900, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Instrucciones
        cv2.putText(frame, "Presiona ESPACIO para agregar producto", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Presiona C para limpiar carrito", (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Presiona ESC para salir", (10, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def detectar_productos(self, frame, area_deteccion):
        # Configuración optimizada para mejor detección
        resultados = self.model(frame, 
                               stream=True, 
                               verbose=False,
                               conf=0.3,      # Confianza mínima
                               iou=0.5,       # Supresión no máxima
                               max_det=10)    # Máximo 10 detecciones
        productos_detectados = []
        productos_en_area = []

        for resultado in resultados:
            if resultado.boxes is not None:
                for box in resultado.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confianza = float(box.conf[0])
                    clase_id = int(box.cls[0])
                    
                    # Obtener nombre del producto
                    nombre_producto = self.model.names[clase_id]
                    
                    # Solo mostrar productos que están en nuestro supermercado
                    if nombre_producto in self.productos_supermercado and confianza > 0.4:
                        productos_detectados.append(nombre_producto)
                        
                        # Verificar si está en el área de detección
                        en_area = self.esta_en_area_deteccion(x1, y1, x2, y2, area_deteccion)
                        
                        if en_area:
                            productos_en_area.append(nombre_producto)
                            # Color verde para productos en área
                            color = (0, 255, 0)
                            frame = self.dibujar_rectangulo(frame, color, x1, y1, x2, y2)
                        else:
                            # Color gris para productos fuera del área
                            color = (128, 128, 128)
                            frame = self.dibujar_rectangulo(frame, color, x1, y1, x2, y2)
                        
                        precio = self.productos_supermercado[nombre_producto]
                        estado = "EN AREA" if en_area else "FUERA"
                        texto = f"{nombre_producto}: S/{precio:.2f} ({int(confianza*100)}%) {estado}"
                        frame = self.dibujar_texto(frame, texto, x1, y1, color)

        return frame, productos_en_area

    def ejecutar(self):
        print("Supermercado AI iniciado!")
        print("=== SUPERMERCADO PERUANO - PRODUCTOS DETECTABLES ===")
        print("\nFRUTAS Y VERDURAS:")
        print("- banana (S/3.50), apple (S/5.00), orange (S/4.50)")
        print("- broccoli (S/8.00), carrot (S/2.50)")
        print("\nBEBIDAS Y CONTENEDORES:")
        print("- bottle (S/2.50), wine glass (S/15.00), cup (S/8.00)")
        print("\nUTENSILIOS:")
        print("- fork (S/12.00), knife (S/18.00), spoon (S/10.00), bowl (S/25.00)")
        print("\nCOMIDA PREPARADA:")
        print("- sandwich (S/12.00), hot dog (S/8.00), pizza (S/35.00)")
        print("- donut (S/6.00), cake (S/45.00)")
        print("\nOTROS PRODUCTOS:")
        print("- toothbrush (S/15.00), book (S/35.00), scissors (S/20.00)")
        print("- cell phone (S/1200.00), mouse (S/65.00), keyboard (S/120.00)")
        print("\n¡Muestra estos objetos a la cámara para detectarlos!")
        print("Controles: ESPACIO=Agregar | C=Limpiar carrito | ESC=Salir\n")
        
        ultimo_producto_detectado = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Dibujar área de detección
            frame, area_deteccion = self.dibujar_area_deteccion(frame)
            
            # Detectar productos
            frame, productos_en_area = self.detectar_productos(frame, area_deteccion)
            
            # Guardar último producto EN ÁREA para agregar al carrito
            if productos_en_area:
                ultimo_producto_detectado = productos_en_area[0]

            # Mostrar logo y carrito
            frame = self.mostrar_logo(frame)
            self.mostrar_carrito(frame)
            self.calcular_total()

            # Mostrar frame
            cv2.imshow("Supermercado AI", frame)

            # Controles
            tecla = cv2.waitKey(1) & 0xFF
            
            if tecla == 27:  # ESC
                break
            elif tecla == ord(' ') and ultimo_producto_detectado:  # ESPACIO
                self.agregar_al_carrito(ultimo_producto_detectado)
                print(f"Agregado: {ultimo_producto_detectado}")
            elif tecla == ord('c'):  # C
                self.carrito = []
                self.total = 0
                print("Carrito limpiado")

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    supermercado = SupermarketAI()
    supermercado.ejecutar()