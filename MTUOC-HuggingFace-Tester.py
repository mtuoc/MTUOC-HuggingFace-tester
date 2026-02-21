import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import threading
from hf_engine import HFModelEngine

class MTUOCTesterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MTUOC HuggingFace Tester")
        self.root.geometry("1500x1200") # Mida inicial més compacta
        
        # Inicialitzem el motor de Hugging Face
        self.engine = HFModelEngine("config.yaml")
        if not self.engine.config:
            messagebox.showerror("Error Crític", "No s'ha trobat el fitxer config.yaml o és invàlid.")
            self.root.destroy()
            return

        # Configurem la interfície amb scroll
        self.setup_scrollable_ui()
        
        # Iniciem la càrrega del model en un fil separat
        threading.Thread(target=self.engine.load_model, 
                         args=(self.update_button_status,), 
                         daemon=True).start()

    def setup_scrollable_ui(self):
        """Crea una base amb scrollbar per a la interfície de Hugging Face."""
        # 1. Contenidor principal
        self.main_container = tk.Frame(self.root)
        self.main_container.pack(fill="both", expand=True)

        # 2. Canvas i Scrollbar
        self.canvas = tk.Canvas(self.main_container, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=self.canvas.yview)
        
        # 3. El frame intern que contindrà els widgets
        self.scrollable_frame = tk.Frame(self.canvas, padx=25, pady=20)

        # Configuració de la regió de scroll
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Creem la finestra dins del canvas
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Ajust de l'amplada automàtic
        self.canvas.bind('<Configure>', self.on_canvas_configure)

        # Empaquetar el conjunt
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # 4. Construïm els widgets de HF dins del frame amb scroll
        self.build_hf_widgets(self.scrollable_frame)

    def on_canvas_configure(self, event):
        """Assegura que el contingut s'estira horitzontalment."""
        self.canvas.itemconfig(self.canvas_frame, width=event.width)

    def build_hf_widgets(self, parent):
        """Reconstrueix la interfície original dins del contenidor amb scroll."""
        # 1. Indicador del Model
        model_name = self.engine.config['model_settings']['name']
        tk.Label(parent, text=f"Model: {model_name}", fg="#666", font=("Arial", 9)).pack(anchor="e")

        # 2. Entrada de text (Prompt)
        tk.Label(parent, text="INPUT PROMPT", font=("Arial", 10, "bold")).pack(anchor="w")
        self.input_txt = scrolledtext.ScrolledText(parent, height=10, font=("Consolas", 11))
        self.input_txt.pack(fill="both", pady=(5, 15))

        # 3. Botó d'acció
        self.btn_gen = tk.Button(
            parent, 
            text="LOADING MODEL...", 
            bg="#9E9E9E", 
            fg="white", 
            state="disabled",
            font=("Arial", 11, "bold"), 
            pady=10,
            command=self.on_generate
        )
        self.btn_gen.pack(fill="x", pady=5)

        # 4. Resposta Bruta (Raw)
        tk.Label(parent, text="FULL MODEL RESPONSE", font=("Arial", 10, "bold")).pack(anchor="w", pady=(15, 0))
        self.raw_out = scrolledtext.ScrolledText(parent, height=8, font=("Consolas", 10), bg="#F5F5F5")
        self.raw_out.pack(fill="both", pady=5)

        # 5. Configuració de filtratge (Regex)
        regex_frame = tk.LabelFrame(parent, text=" Filter Settings (JSON + Regex) ", padx=15, pady=10)
        regex_frame.pack(fill="x", pady=15)

        tk.Label(regex_frame, text="Regex Pattern:").pack(side="left")
        self.reg_entry = tk.Entry(regex_frame, font=("Consolas", 11))
        self.reg_entry.pack(side="left", fill="x", expand=True, padx=10)
        
        def_reg = self.engine.config.get('prompt_settings', {}).get('regex_pattern', "")
        if def_reg and def_reg != "None":
            self.reg_entry.insert(0, def_reg)

        # 6. Resultat Final (Filtrat)
        tk.Label(parent, text="FINAL FILTERED RESULT", font=("Arial", 10, "bold"), fg="#2E7D32").pack(anchor="w")
        self.final_out = scrolledtext.ScrolledText(parent, height=6, font=("Consolas", 12, "bold"), bg="#F1F8E9")
        self.final_out.pack(fill="both", pady=5)

    def update_button_status(self, status):
        """Actualitza el botó segons l'estat del motor."""
        if status == "READY":
            self.btn_gen.config(state="normal", text="GENERATE TEXT", bg="#4CAF50")
        elif "ERROR" in status:
            self.btn_gen.config(state="disabled", text="LOAD ERROR", bg="#F44336")
            messagebox.showerror("Error de càrrega", status)
        else:
            self.btn_gen.config(state="disabled", text=status, bg="#9E9E9E")

    def on_generate(self):
        """Gestiona la petició de generació."""
        prompt = self.input_txt.get("1.0", "end").strip()
        regex = self.reg_entry.get().strip()
        
        if not prompt: return

        def run_inference():
            self.btn_gen.config(state="disabled", text="GENERATING...", bg="#FF9800")
            try:
                raw, final = self.engine.generate(prompt, override_regex=regex)
                self.raw_out.delete("1.0", "end")
                self.raw_out.insert("end", raw)
                self.final_out.delete("1.0", "end")
                self.final_out.insert("end", final)
            except Exception as e:
                messagebox.showerror("Error de Generació", str(e))
            finally:
                self.update_button_status("READY")

        threading.Thread(target=run_inference, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = MTUOCTesterGUI(root)
    root.mainloop()
