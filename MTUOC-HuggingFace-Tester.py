import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import threading
from hf_engine import HFModelEngine

class MTUOCTesterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MTUOC HuggingFace Tester")
        self.root.geometry("1000x1200")
        
        # Inicialitzem el motor de Hugging Face
        self.engine = HFModelEngine("config.yaml")
        if not self.engine.config:
            messagebox.showerror("Error Crític", "No s'ha trobat el fitxer config_tester.yaml o és invàlid.")
            self.root.destroy()
            return

        self.setup_ui()
        
        # Iniciem la càrrega del model en un fil separat
        # Passem 'self.update_button_status' com a callback per actualitzar el botó
        threading.Thread(target=self.engine.load_model, 
                         args=(self.update_button_status,), 
                         daemon=True).start()

    def setup_ui(self):
        """Configura la interfície gràfica."""
        main_frame = tk.Frame(self.root, padx=25, pady=20)
        main_frame.pack(fill="both", expand=True)

        # 1. Indicador del Model
        model_name = self.engine.config['model_settings']['name']
        tk.Label(main_frame, text=f"Model: {model_name}", fg="#666", font=("Arial", 9)).pack(anchor="e")

        # 2. Entrada de text (Prompt)
        tk.Label(main_frame, text="INPUT PROMPT", font=("Arial", 10, "bold")).pack(anchor="w")
        self.input_txt = scrolledtext.ScrolledText(main_frame, height=10, font=("Consolas", 11))
        self.input_txt.pack(fill="both", expand=True, pady=(5, 15))

        # 3. Botó d'acció (amb estats dinàmics)
        # Inicialment en gris (LOADING)
        self.btn_gen = tk.Button(
            main_frame, 
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
        tk.Label(main_frame, text="FULL MODEL RESPONSE", font=("Arial", 10, "bold")).pack(anchor="w", pady=(15, 0))
        self.raw_out = scrolledtext.ScrolledText(main_frame, height=8, font=("Consolas", 10), bg="#F5F5F5")
        self.raw_out.pack(fill="both", expand=True, pady=5)

        # 5. Configuració de filtratge (Regex)
        regex_frame = tk.LabelFrame(main_frame, text=" Filter Settings (JSON + Regex) ", padx=15, pady=10)
        regex_frame.pack(fill="x", pady=15)

        tk.Label(regex_frame, text="Regex Pattern:").pack(side="left")
        self.reg_entry = tk.Entry(regex_frame, font=("Consolas", 11))
        self.reg_entry.pack(side="left", fill="x", expand=True, padx=10)
        
        # Carreguem el regex per defecte del YAML
        def_reg = self.engine.config.get('prompt_settings', {}).get('regex_pattern', "")
        if def_reg and def_reg != "None":
            self.reg_entry.insert(0, def_reg)

        # 6. Resultat Final (Filtrat)
        tk.Label(main_frame, text="FINAL FILTERED RESULT", font=("Arial", 10, "bold"), fg="#2E7D32").pack(anchor="w")
        self.final_out = scrolledtext.ScrolledText(main_frame, height=6, font=("Consolas", 12, "bold"), bg="#F1F8E9")
        self.final_out.pack(fill="both", expand=True, pady=5)

    def update_button_status(self, status):
        """Actualitza el botó segons l'estat del motor (thread-safe)."""
        if status == "READY":
            self.btn_gen.config(state="normal", text="GENERATE TEXT", bg="#4CAF50") # Verd
        elif "ERROR" in status:
            self.btn_gen.config(state="disabled", text=f"LOAD ERROR", bg="#F44336") # Vermell
            messagebox.showerror("Error de càrrega", status)
        else:
            # LOADING o GENERATING
            self.btn_gen.config(state="disabled", text=status, bg="#9E9E9E")

    def on_generate(self):
        """Gestiona la petició de generació."""
        prompt = self.input_txt.get("1.0", "end").strip()
        regex = self.reg_entry.get().strip()
        
        if not prompt:
            return

        def run_inference():
            # Bloquegem el botó mentre genera
            self.btn_gen.config(state="disabled", text="GENERATING...", bg="#FF9800") # Taronja
            
            try:
                # Cridem al motor (retorna raw i processed)
                raw, final = self.engine.generate(prompt, override_regex=regex)
                
                # Actualitzem la interfície (usant delete/insert)
                self.raw_out.delete("1.0", "end")
                self.raw_out.insert("end", raw)
                
                self.final_out.delete("1.0", "end")
                self.final_out.insert("end", final)
                
            except Exception as e:
                messagebox.showerror("Error de Generació", str(e))
            finally:
                # Tornem el botó a l'estat READY
                self.update_button_status("READY")

        # Executem la inferència en un fil per no congelar la GUI
        threading.Thread(target=run_inference, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = MTUOCTesterGUI(root)
    root.mainloop()
