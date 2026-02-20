import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from tkinter.scrolledtext import ScrolledText
import threading
import re
from huggingface_hub import scan_cache_dir
from transformers import pipeline
import torch

class HFApp:
    def __init__(self, master):
        self.master = master
        self.master.title("MTUOC HuggingFace Tester Pro")
        self.master.geometry("1700x1200")
        
        self.pipe = None
        self.model_name = tk.StringVar(value="gpt2")
        self.do_sample = tk.BooleanVar(value=False)
        self.early_stopping = tk.BooleanVar(value=True)

        self.style = ttk.Style()
        self.master.option_add('*TCombobox*Listbox.font', ('Segoe UI', 10))
        
        self.setup_ui()
        self.refresh_local_models() 

    def setup_ui(self):
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(3, weight=1) # Donem pes a la fila de la Resposta

        # ──── 1. Selecció de Model ────
        model_frame = tk.LabelFrame(self.master, text="Model Selection", padx=20, pady=5)
        model_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=5)
        model_frame.columnconfigure(0, weight=1)
        
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_name, font=('Segoe UI', 10))
        self.model_combo.grid(row=0, column=0, columnspan=3, sticky="ew", pady=5)
        
        btn_row_frame = tk.Frame(model_frame)
        btn_row_frame.grid(row=1, column=0, columnspan=3, sticky="w")
        tk.Button(btn_row_frame, text=" 📁 Browse ", command=self.browse_model_folder, font=('Segoe UI', 9)).pack(side="left", padx=5)
        tk.Button(btn_row_frame, text=" 🔄 Refresh ", command=self.refresh_local_models, font=('Segoe UI', 9)).pack(side="left", padx=5)
        tk.Button(btn_row_frame, text=" LOAD MODEL ", command=self.load_model, bg="#2196F3", fg="white", font=('Segoe UI', 9, 'bold'), padx=15).pack(side="left", padx=20)

        # ──── 2. Paràmetres ────
        param_frame = tk.LabelFrame(self.master, text="Parameters", padx=15, pady=5)
        param_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=5)
        self.params = {
            "max_new_tokens": tk.IntVar(value=100),
            "num_beams": tk.IntVar(value=5),
            "repetition_penalty": tk.DoubleVar(value=1.1),
            "no_repeat_ngram_size": tk.IntVar(value=3),
            "temperature": tk.DoubleVar(value=0.9),
            "top_p": tk.DoubleVar(value=0.95),
        }
        for i, (k, v) in enumerate(self.params.items()):
            r, c = divmod(i, 4) 
            tk.Label(param_frame, text=f"{k}:", font=('Segoe UI', 8)).grid(row=r, column=c*2, sticky="w", padx=(10, 2))
            tk.Entry(param_frame, textvariable=v, width=8, font=('Segoe UI', 9)).grid(row=r, column=c*2+1, sticky="w")
        tk.Checkbutton(param_frame, text="do_sample", variable=self.do_sample, font=('Segoe UI', 9, 'bold')).grid(row=0, column=8, padx=20)

        # ──── 3. Prompt (Compacte) ────
        prompt_frame = tk.LabelFrame(self.master, text="Input Prompt", padx=15, pady=5)
        prompt_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=5)
        self.prompt_text = ScrolledText(prompt_frame, height=4, font=('Consolas', 10))
        self.prompt_text.pack(fill="both", expand=True)

        # ──── 4. Resposta del Model (Més gran: 15 línies) ────
        response_frame = tk.LabelFrame(self.master, text="Model Response", padx=15, pady=5)
        response_frame.grid(row=3, column=0, sticky="nsew", padx=15, pady=5)
        self.response_text = ScrolledText(response_frame, height=15, font=('Consolas', 10), bg="#fcfcfc")
        self.response_text.pack(fill="both", expand=True)

        # ──── 5. Botons d'Acció ────
        btn_frame = tk.Frame(self.master)
        btn_frame.grid(row=4, column=0, pady=5)
        self.gen_button = tk.Button(btn_frame, text="GENERATE TEXT", command=self.send_prompt, bg="#4CAF50", fg="white", font=('Segoe UI', 10, 'bold'), padx=25, pady=8)
        self.gen_button.pack(side="left", padx=10)
        tk.Button(btn_frame, text="Clear Everything", command=self.clear_all, font=('Segoe UI', 9), padx=15).pack(side="left", padx=10)

        # ──── 6. Regex (Reduït: 3 línies) ────
        regex_frame = tk.LabelFrame(self.master, text="Regex Filter", padx=15, pady=5)
        regex_frame.grid(row=5, column=0, sticky="ew", padx=15, pady=5)
        tk.Label(regex_frame, text="Pattern:", font=('Segoe UI', 9, 'bold')).pack(anchor="w")
        self.regexp_entry = tk.Entry(regex_frame, font=('Consolas', 10))
        self.regexp_entry.pack(fill="x", pady=2)
        
        self.regexp_result = ScrolledText(regex_frame, height=3, bg="#f3f3f3", font=('Consolas', 10))
        self.regexp_result.pack(fill="x", pady=5)
        tk.Button(regex_frame, text="Apply Filter", command=self.apply_regexp, font=('Segoe UI', 9)).pack(pady=2)

    def refresh_local_models(self):
        try:
            cache_info = scan_cache_dir()
            models = sorted([repo.repo_id for repo in cache_info.repos if repo.repo_type == "model"])
            self.model_combo['values'] = models
            if models: self.model_combo.set(models[0])
        except Exception: self.model_combo.set("gpt2")

    def browse_model_folder(self):
        folder = filedialog.askdirectory()
        if folder: self.model_name.set(folder)

    def load_model(self):
        def _load():
            m = self.model_name.get().strip()
            try:
                self.pipe = pipeline("text-generation", model=m, device=0 if torch.cuda.is_available() else -1)
                messagebox.showinfo("Success", f"Model loaded: {m}")
            except Exception as e: messagebox.showerror("Error", str(e))
        threading.Thread(target=_load, daemon=True).start()

    def send_prompt(self):
        if not self.pipe:
            messagebox.showwarning("Warning", "Load a model first.")
            return
        prompt = self.prompt_text.get("1.0", "end").strip()
        if not prompt: return

        self.gen_button.config(state="disabled", text="GENERATING...", bg="#9e9e9e")

        gen_kwargs = {
            "max_new_tokens": self.params["max_new_tokens"].get(),
            "repetition_penalty": self.params["repetition_penalty"].get(),
            "no_repeat_ngram_size": self.params["no_repeat_ngram_size"].get(),
            "do_sample": self.do_sample.get(),
            "pad_token_id": self.pipe.tokenizer.eos_token_id or 0
        }
        if self.do_sample.get():
            gen_kwargs["temperature"] = self.params["temperature"].get()
            gen_kwargs["top_p"] = self.params["top_p"].get()
        else:
            gen_kwargs["num_beams"] = self.params["num_beams"].get()

        def _generate():
            try:
                res = self.pipe(prompt, **gen_kwargs)
                self.response_text.delete("1.0", "end")
                self.response_text.insert("end", res[0]['generated_text'])
            except Exception as e: messagebox.showerror("Error", str(e))
            finally: self.gen_button.config(state="normal", text="GENERATE TEXT", bg="#4CAF50")

        threading.Thread(target=_generate, daemon=True).start()

    def clear_all(self):
        for w in [self.prompt_text, self.response_text, self.regexp_result]: w.delete("1.0", "end")
        self.regexp_entry.delete(0, "end")

    def apply_regexp(self):
        pat = self.regexp_entry.get().strip()
        txt = self.response_text.get("1.0", "end").strip()
        self.regexp_result.delete("1.0", "end")
        if not pat: return
        try:
            matches = re.findall(pat, txt, re.MULTILINE)
            out = "\n".join([" | ".join(map(str, m)) if isinstance(m, tuple) else str(m) for m in matches])
            self.regexp_result.insert("end", out or "[No matches]")
        except Exception as e: self.regexp_result.insert("end", f"Error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    HFApp(root)
    root.mainloop()
