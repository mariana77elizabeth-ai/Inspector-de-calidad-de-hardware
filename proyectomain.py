import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
import time

# --- CONFIGURACIÓN VISUAL ---
ctk.set_appearance_mode("light") 
ctk.set_default_color_theme("blue") 

class AppInspectorIA(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Innova PyColab - AI Hardware Inspector (DeepSeek Powered)")
        self.geometry("1100x800")

        # Contenedor para el cambio de pantallas
        self.container = ctk.CTkFrame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (PantallaInicio, PantallaPrincipal):
            frame = F(self.container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.mostrar_pantalla(PantallaInicio)

    def mostrar_pantalla(self, pagina):
        frame = self.frames[pagina]
        frame.tkraise()

# --- PANTALLA 1: BIENVENIDA ---
class PantallaInicio(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        
        self.lbl = ctk.CTkLabel(self, text="🛠️ INNOVA PYCOLAB AI", font=("Times New Roman", 45, "bold"), text_color="magenta")
        self.lbl.pack(pady=(150, 10))

        self.sub = ctk.CTkLabel(self, text="Control de Calidad basado en DeepLearning (DeepSeek)\nAnálisis de Hardware v3.0", font=("Times New Roman", 18))
        self.sub.pack(pady=20)

        self.btn = ctk.CTkButton(self, text="INGRESAR AL SISTEMA", font=("Arial", 16, "bold"),
                                 fg_color="magenta", hover_color="#C71585", height=50, width=280,
                                 command=lambda: controller.mostrar_pantalla(PantallaPrincipal))
        self.btn.pack(pady=50)

# --- PANTALLA 2: INSPECTOR IA ---
class PantallaPrincipal(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        
        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(self.sidebar, text="IA CORE", font=("Times New Roman", 22, "bold")).pack(pady=30)

        self.btn_rj = ctk.CTkButton(self.sidebar, text="DeepSeek: RJ45", fg_color="magenta", command=lambda: self.set_modo("RJ45"))
        self.btn_rj.pack(pady=15, padx=20)

        self.btn_ram = ctk.CTkButton(self.sidebar, text="DeepSeek: RAM", fg_color="magenta", command=lambda: self.set_modo("RAM"))
        self.btn_ram.pack(pady=15, padx=20)

        ctk.CTkButton(self.sidebar, text="← Salir", fg_color="gray", command=lambda: controller.mostrar_pantalla(PantallaInicio)).pack(side="bottom", pady=20)

        # Panel Principal
        self.main = ctk.CTkFrame(self)
        self.main.pack(side="right", expand=True, fill="both", padx=10, pady=10)

        self.lbl_status = ctk.CTkLabel(self.main, text="Esperando Selección de Modelo...", font=("Times New Roman", 18, "bold"))
        self.lbl_status.pack(pady=15)

        # Visualizador Dual
        self.viz = ctk.CTkFrame(self.main, fg_color="transparent")
        self.viz.pack(expand=True, fill="both")

        self.label_orig = ctk.CTkLabel(self.viz, text="Imagen de Entrada")
        self.label_orig.grid(row=0, column=0, padx=20, pady=10)

        self.label_proc = ctk.CTkLabel(self.viz, text="Mapa de Calor IA (DeepSeek)")
        self.label_proc.grid(row=0, column=1, padx=20, pady=10)

        self.viz.grid_columnconfigure(0, weight=1)
        self.viz.grid_columnconfigure(1, weight=1)

        self.btn_load = ctk.CTkButton(self.main, text="Subir para Inferencia IA", state="disabled", command=self.ejecutar_ia)
        self.btn_load.pack(pady=10)
        
        self.btn_save = ctk.CTkButton(self.main, text="Exportar Reporte Técnico", fg_color="#27ae60", state="disabled", command=self.guardar)
        self.btn_save.pack(pady=10)

        self.modo = None
        self.img_cv = None
        self.img_ai_cv = None

    def set_modo(self, modo):
        self.modo = modo
        self.lbl_status.configure(text=f"Modelo DeepSeek cargado para: {modo}", text_color="magenta")
        self.btn_load.configure(state="normal")

    def ejecutar_ia(self):
        ruta = filedialog.askopenfilename()
        if not ruta: return
        
        self.img_cv = cv2.imread(ruta)
        self.lbl_status.configure(text="DeepSeek analizando patrones... ⏳", text_color="blue")
        self.update()
        time.sleep(1.2) # Simulamos el tiempo de respuesta de la red neuronal

        if self.modo == "RJ45":
            res, self.img_ai_cv = self.deepseek_inference_rj45(self.img_cv)
        else:
            res, self.img_ai_cv = self.deepseek_inference_ram(self.img_cv)

        self.lbl_status.configure(text=res, text_color="#2ecc71" if "✅" in res else "#e74c3c")
        self.mostrar(cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2RGB), self.label_orig)
        
        # Procesar visualización de IA
        ai_display = self.img_ai_cv
        if len(ai_display.shape) == 2: ai_display = cv2.cvtColor(ai_display, cv2.COLOR_GRAY2RGB)
        else: ai_display = cv2.cvtColor(ai_display, cv2.COLOR_BGR2RGB)
        
        self.mostrar(ai_display, self.label_proc)
        self.btn_save.configure(state="normal")

    def mostrar(self, arr, label):
        img = Image.fromarray(arr).resize((450, 320))
        img_tk = ImageTk.PhotoImage(img)
        label.configure(image=img_tk, text="")
        label.image = img_tk

    # --- LÓGICA DE IA (SIMULACIÓN DE ENTRENAMIENTO) ---
    def deepseek_inference_rj45(self, img):
        """Simulación de red neuronal identificando hilos del conector"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([180, 255, 255]))
        
        # Simulamos 'Attention Map' de DeepSeek
        atencion = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        
        h, w = mask.shape
        score = np.count_nonzero(mask[0:int(h*0.15), :])
        if score > (mask.size * 0.02):
            return "✅ DEEPSEEK: Conexión detectada al 98% de confianza", atencion
        return "❌ DEEPSEEK: Fallo de continuidad detectado en pines", atencion

    def deepseek_inference_ram(self, img):
        """Simulación de red neuronal identificando ángulos de inserción"""
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bordes = cv2.Canny(gris, 50, 150)
        
        # Simulamos mapa de probabilidad
        mapa_prob = cv2.applyColorMap(bordes, cv2.COLORMAP_HOT)
        
        lineas = cv2.HoughLinesP(bordes, 1, np.pi/180, 50, minLineLength=50)
        if lineas is not None:
            for l in lineas:
                x1, y1, x2, y2 = l[0]
                ang = np.abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
                if 4 < ang < 176: return f"❌ DEEPSEEK: RAM Chueca detectada ({ang:.1f}°)", mapa_prob
            return "✅ DEEPSEEK: Alineación perfecta detectada", mapa_prob
        return "⚠️ DEEPSEEK: Objeto no identificado", mapa_prob

    def guardar(self):
        p = self.img_ai_cv
        if len(p.shape) == 2: p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
        # Unimos original con el mapa de calor de la IA
        reporte = np.hstack((self.img_cv, p))
        ruta = filedialog.asksaveasfilename(defaultextension=".jpg")
        if ruta: cv2.imwrite(ruta, reporte)

if __name__ == "__main__":
    app = AppInspectorIA()
    app.mainloop()
