import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import json
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Check if zipvoice is available
try:
    from zipvoice.luxvoice import LuxTTS
    LUXTTS_AVAILABLE = True
except ImportError:
    LUXTTS_AVAILABLE = False
    print("Warning: LuxTTS not installed. Install with: pip install zipvoice")

class ModernTheme:
    """Modern dark theme styling"""
    BG_COLOR = "#1e1e2e"
    FG_COLOR = "#cdd6f4"
    ACCENT_COLOR = "#89b4fa"
    SUCCESS_COLOR = "#a6e3a1"
    ERROR_COLOR = "#f38ba8"
    WARNING_COLOR = "#fab387"
    SURFACE_COLOR = "#313244"
    HOVER_COLOR = "#45475a"
    
    @classmethod
    def apply(cls, root):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure("TFrame", background=cls.BG_COLOR)
        style.configure("TLabel", background=cls.BG_COLOR, foreground=cls.FG_COLOR, font=('Segoe UI', 10))
        style.configure("TButton", background=cls.SURFACE_COLOR, foreground=cls.FG_COLOR, 
                       font=('Segoe UI', 10, 'bold'), padding=10)
        style.map("TButton", background=[('active', cls.HOVER_COLOR), ('pressed', cls.ACCENT_COLOR)])
        
        style.configure("Accent.TButton", background=cls.ACCENT_COLOR, foreground=cls.BG_COLOR)
        style.map("Accent.TButton", background=[('active', '#b4befe')])
        
        style.configure("TEntry", fieldbackground=cls.SURFACE_COLOR, foreground=cls.FG_COLOR, 
                       insertcolor=cls.FG_COLOR, padding=5)
        style.configure("TCombobox", fieldbackground=cls.SURFACE_COLOR, foreground=cls.FG_COLOR)
        
        style.configure("Horizontal.TProgressbar", background=cls.ACCENT_COLOR, troughcolor=cls.SURFACE_COLOR)
        style.configure("TNotebook", background=cls.BG_COLOR, tabmargins=[2, 5, 2, 0])
        style.configure("TNotebook.Tab", background=cls.SURFACE_COLOR, foreground=cls.FG_COLOR, 
                       padding=[10, 5], font=('Segoe UI', 10))
        style.map("TNotebook.Tab", background=[("selected", cls.ACCENT_COLOR)], 
                 foreground=[("selected", cls.BG_COLOR)])
        
        root.configure(bg=cls.BG_COLOR)

class LuxTTSGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LuxTTS Studio - Professional Voice Cloning")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Apply modern theme
        ModernTheme.apply(root)
        
        # Initialize model
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threads = 4
        
        # Processing queue
        self.processing_queue = queue.Queue()
        self.is_processing = False
        self.stop_requested = False
        
        # Settings
        self.settings_file = Path.home() / ".luxtts_studio.json"
        self.load_settings()
        
        # Build UI
        self.create_menu()
        self.create_main_layout()
        
        # Status bar
        self.create_status_bar()
        
        # Start queue processor
        self.process_thread = threading.Thread(target=self.process_queue, daemon=True)
        self.process_thread.start()
        
        # Check model availability
        if not LUXTTS_AVAILABLE:
            self.log("ERROR: LuxTTS not installed. Run: pip install zipvoice", "error")
        else:
            self.log("Ready. Load a model to begin.", "info")
    
    def create_menu(self):
        menubar = tk.Menu(self.root, bg=ModernTheme.SURFACE_COLOR, fg=ModernTheme.FG_COLOR,
                         activebackground=ModernTheme.ACCENT_COLOR, activeforeground=ModernTheme.BG_COLOR)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0, bg=ModernTheme.SURFACE_COLOR, fg=ModernTheme.FG_COLOR)
        file_menu.add_command(label="Load Model", command=self.load_model_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Export Settings", command=self.export_settings)
        file_menu.add_command(label="Import Settings", command=self.import_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0, bg=ModernTheme.SURFACE_COLOR, fg=ModernTheme.FG_COLOR)
        tools_menu.add_command(label="Clear Cache", command=self.clear_cache)
        tools_menu.add_command(label="Optimize GPU", command=self.optimize_gpu)
        menubar.add_cascade(label="Tools", menu=tools_menu)
    
    def create_main_layout(self):
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)  # Left panel
        main_frame.columnconfigure(1, weight=1)  # Right panel
        main_frame.rowconfigure(0, weight=1)
        
        # Left Panel - Controls
        left_panel = self.create_left_panel(main_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Right Panel - Queue & Log
        right_panel = self.create_right_panel(main_frame)
        right_panel.grid(row=0, column=1, sticky="nsew")
    
    def create_left_panel(self, parent):
        panel = ttk.Frame(parent)
        panel.columnconfigure(0, weight=1)
        
        # Notebook for tabs
        notebook = ttk.Notebook(panel)
        notebook.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        panel.rowconfigure(0, weight=1)
        
        # Single Generation Tab
        single_tab = ttk.Frame(notebook, padding=10)
        notebook.add(single_tab, text="Single Generation")
        self.setup_single_tab(single_tab)
        
        # Batch Processing Tab
        batch_tab = ttk.Frame(notebook, padding=10)
        notebook.add(batch_tab, text="Batch Processing")
        self.setup_batch_tab(batch_tab)
        
        # Settings Tab
        settings_tab = ttk.Frame(notebook, padding=10)
        notebook.add(settings_tab, text="Model Settings")
        self.setup_settings_tab(settings_tab)
        
        return panel
    
    def setup_single_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        
        # Reference Audio Section
        ref_frame = tk.LabelFrame(parent, text="Reference Audio (Voice to Clone)", 
                                 bg=ModernTheme.SURFACE_COLOR, fg=ModernTheme.FG_COLOR,
                                 font=('Segoe UI', 11, 'bold'), padx=10, pady=10)
        ref_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ref_frame.columnconfigure(0, weight=1)
        
        self.ref_path_var = tk.StringVar()
        ref_entry = ttk.Entry(ref_frame, textvariable=self.ref_path_var, state="readonly")
        ref_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        ref_btn = ttk.Button(ref_frame, text="Browse", command=self.browse_reference)
        ref_btn.grid(row=0, column=1)
        
        # Drag and drop hint
        hint_label = tk.Label(ref_frame, text="🎵 Drag & drop audio files here", 
                             bg=ModernTheme.SURFACE_COLOR, fg=ModernTheme.ACCENT_COLOR,
                             font=('Segoe UI', 9, 'italic'))
        hint_label.grid(row=1, column=0, columnspan=2, pady=(5, 0))
        
        # Enable drag and drop
        ref_frame.drop_target_register(tk.DND_FILES)
        ref_frame.dnd_bind('<<Drop>>', self.on_drop_reference)
        
        # Text Input Section
        text_frame = tk.LabelFrame(parent, text="Text to Synthesize", 
                                  bg=ModernTheme.SURFACE_COLOR, fg=ModernTheme.FG_COLOR,
                                  font=('Segoe UI', 11, 'bold'), padx=10, pady=10)
        text_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        
        self.text_input = scrolledtext.ScrolledText(
            text_frame, wrap=tk.WORD, height=8,
            bg=ModernTheme.SURFACE_COLOR, fg=ModernTheme.FG_COLOR,
            insertbackground=ModernTheme.FG_COLOR, font=('Consolas', 11),
            relief=tk.FLAT, padx=5, pady=5
        )
        self.text_input.grid(row=0, column=0, sticky="nsew")
        
        # Quick actions
        btn_frame = ttk.Frame(text_frame)
        btn_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        
        ttk.Button(btn_frame, text="Clear", command=lambda: self.text_input.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Load from File", command=self.load_text_file).pack(side=tk.LEFT)
        
        # Output Section
        out_frame = tk.LabelFrame(parent, text="Output Settings", 
                                 bg=ModernTheme.SURFACE_COLOR, fg=ModernTheme.FG_COLOR,
                                 font=('Segoe UI', 11, 'bold'), padx=10, pady=10)
        out_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        out_frame.columnconfigure(1, weight=1)
        
        ttk.Label(out_frame, text="Filename:").grid(row=0, column=0, sticky="w")
        self.output_name_var = tk.StringVar(value="output.wav")
        ttk.Entry(out_frame, textvariable=self.output_name_var).grid(row=0, column=1, sticky="ew", padx=5)
        
        ttk.Label(out_frame, text="Output Dir:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.output_dir_var = tk.StringVar(value=str(Path.home() / "LuxTTS_Output"))
        ttk.Entry(out_frame, textvariable=self.output_dir_var, state="readonly").grid(row=1, column=1, sticky="ew", padx=5, pady=(5, 0))
        ttk.Button(out_frame, text="Browse", command=self.browse_output_dir).grid(row=1, column=2, pady=(5, 0))
        
        # Generate Button
        self.gen_btn = ttk.Button(parent, text="🎙️ Generate Speech", command=self.queue_single_generation, 
                                 style="Accent.TButton")
        self.gen_btn.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        
        # Parameters Frame
        params_frame = tk.LabelFrame(parent, text="Generation Parameters", 
                                    bg=ModernTheme.SURFACE_COLOR, fg=ModernTheme.FG_COLOR,
                                    font=('Segoe UI', 11, 'bold'), padx=10, pady=10)
        params_frame.grid(row=4, column=0, sticky="ew")
        
        self.create_parameter_controls(params_frame)
    
    def setup_batch_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        
        # Batch Controls
        ctrl_frame = ttk.Frame(parent)
        ctrl_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Button(ctrl_frame, text="📁 Add Files", command=self.add_batch_files).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(ctrl_frame, text="📂 Add Folder", command=self.add_batch_folder).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(ctrl_frame, text="🗑️ Clear All", command=self.clear_batch).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(ctrl_frame, text="▶️ Start Batch", command=self.start_batch, style="Accent.TButton").pack(side=tk.RIGHT)
        
        # Batch List
        list_frame = tk.LabelFrame(parent, text="Batch Queue", 
                                  bg=ModernTheme.SURFACE_COLOR, fg=ModernTheme.FG_COLOR,
                                  font=('Segoe UI', 11, 'bold'))
        list_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Treeview for batch items
        columns = ("status", "text_file", "ref_audio", "output", "params")
        self.batch_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=10)
        
        self.batch_tree.heading("status", text="Status")
        self.batch_tree.heading("text_file", text="Text File")
        self.batch_tree.heading("ref_audio", text="Reference Audio")
        self.batch_tree.heading("output", text="Output")
        self.batch_tree.heading("params", text="Parameters")
        
        self.batch_tree.column("status", width=80, anchor="center")
        self.batch_tree.column("text_file", width=200)
        self.batch_tree.column("ref_audio", width=200)
        self.batch_tree.column("output", width=150)
        self.batch_tree.column("params", width=100)
        
        # Scrollbars
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self.batch_tree.yview)
        hsb = ttk.Scrollbar(list_frame, orient="horizontal", command=self.batch_tree.xview)
        self.batch_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.batch_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        # Context menu
        self.batch_menu = tk.Menu(self.root, tearoff=0, bg=ModernTheme.SURFACE_COLOR, fg=ModernTheme.FG_COLOR)
        self.batch_menu.add_command(label="Remove Selected", command=self.remove_batch_item)
        self.batch_menu.add_command(label="Edit Parameters", command=self.edit_batch_params)
        self.batch_tree.bind("<Button-3>", self.show_batch_menu)
        
        # Batch Settings
        settings_frame = tk.LabelFrame(parent, text="Batch Settings", 
                                      bg=ModernTheme.SURFACE_COLOR, fg=ModernTheme.FG_COLOR,
                                      font=('Segoe UI', 11, 'bold'), padx=10, pady=10)
        settings_frame.grid(row=2, column=0, sticky="ew")
        
        ttk.Label(settings_frame, text="Output Pattern:").grid(row=0, column=0, sticky="w")
        self.batch_pattern_var = tk.StringVar(value="{text_name}_{ref_name}.wav")
        ttk.Entry(settings_frame, textvariable=self.batch_pattern_var, width=40).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(settings_frame, text="Use {text_name}, {ref_name}, {timestamp}", 
                 font=('Segoe UI', 8)).grid(row=0, column=2, sticky="w")
        
        ttk.Label(settings_frame, text="Parallel Jobs:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.parallel_var = tk.IntVar(value=1)
        ttk.Spinbox(settings_frame, from_=1, to=4, textvariable=self.parallel_var, width=5).grid(row=1, column=1, sticky="w", padx=5, pady=(5, 0))
    
    def setup_settings_tab(self, parent):
        parent.columnconfigure(1, weight=1)
        
        # Model Settings
        model_frame = tk.LabelFrame(parent, text="Model Configuration", 
                                   bg=ModernTheme.SURFACE_COLOR, fg=ModernTheme.FG_COLOR,
                                   font=('Segoe UI', 11, 'bold'), padx=10, pady=10)
        model_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)
        
        ttk.Label(model_frame, text="Model ID:").grid(row=0, column=0, sticky="w")
        self.model_id_var = tk.StringVar(value="YatharthS/LuxTTS")
        ttk.Entry(model_frame, textvariable=self.model_id_var).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(model_frame, text="Load", command=self.load_model_dialog).grid(row=0, column=2)
        
        ttk.Label(model_frame, text="Device:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.device_var = tk.StringVar(value=self.device)
        device_combo = ttk.Combobox(model_frame, textvariable=self.device_var, 
                                   values=["cuda", "cpu", "mps"], state="readonly", width=10)
        device_combo.grid(row=1, column=1, sticky="w", padx=5, pady=(5, 0))
        device_combo.bind("<<ComboboxSelected>>", self.on_device_change)
        
        ttk.Label(model_frame, text="CPU Threads:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.threads_var = tk.IntVar(value=self.threads)
        ttk.Spinbox(model_frame, from_=1, to=16, textvariable=self.threads_var, width=5).grid(row=2, column=1, sticky="w", padx=5, pady=(5, 0))
        
        # Optimization Settings
        opt_frame = tk.LabelFrame(parent, text="Optimizations", 
                                 bg=ModernTheme.SURFACE_COLOR, fg=ModernTheme.FG_COLOR,
                                 font=('Segoe UI', 11, 'bold'), padx=10, pady=10)
        opt_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        self.fp16_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text="Use FP16 (Half Precision) - Faster, less VRAM", 
                       variable=self.fp16_var).grid(row=0, column=0, sticky="w")
        
        self.compile_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_frame, text="Compile Model (PyTorch 2.0+) - Much faster after warmup", 
                       variable=self.compile_var).grid(row=1, column=0, sticky="w", pady=(5, 0))
        
        self.cache_prompts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="Cache Encoded Prompts - Speed up repeated voices", 
                       variable=self.cache_prompts_var).grid(row=2, column=0, sticky="w", pady=(5, 0))
        
        # Advanced Parameters
        adv_frame = tk.LabelFrame(parent, text="Advanced Parameters", 
                                 bg=ModernTheme.SURFACE_COLOR, fg=ModernTheme.FG_COLOR,
                                 font=('Segoe UI', 11, 'bold"), padx=10, pady=10)
        adv_frame.grid(row=2, column=0, sticky="ew")
        
        # RMS
        ttk.Label(adv_frame, text="RMS (Volume):").grid(row=0, column=0, sticky="w")
        self.rms_var = tk.DoubleVar(value=0.01)
        ttk.Scale(adv_frame, from_=0.001, to=0.1, variable=self.rms_var, orient="horizontal").grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Label(adv_frame, textvariable=self.rms_var, width=6).grid(row=0, column=2)
        
        # Ref Duration
        ttk.Label(adv_frame, text="Ref Duration (s):").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.ref_dur_var = tk.IntVar(value=5)
        ttk.Spinbox(adv_frame, from_=1, to=30, textvariable=self.ref_dur_var, width=5).grid(row=1, column=1, sticky="w", padx=5, pady=(5, 0))
    
    def create_parameter_controls(self, parent):
        # Create sliders for parameters
        params = [
            ("num_steps", "Steps (Quality):", 1, 10, 4),
            ("t_shift", "Temperature:", 0.1, 2.0, 0.9),
            ("speed", "Speed:", 0.5, 2.0, 1.0),
        ]
        
        self.param_vars = {}
        for i, (key, label, min_val, max_val, default) in enumerate(params):
            ttk.Label(parent, text=label).grid(row=i, column=0, sticky="w")
            var = tk.DoubleVar(value=default)
            self.param_vars[key] = var
            
            scale = ttk.Scale(parent, from_=min_val, to=max_val, variable=var, 
                             orient="horizontal", length=200)
            scale.grid(row=i, column=1, sticky="ew", padx=5)
            ttk.Label(parent, textvariable=var, width=6).grid(row=i, column=2)
        
        # Checkboxes
        self.return_smooth_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Return Smooth (reduce metallic sound)", 
                       variable=self.return_smooth_var).grid(row=len(params), column=0, columnspan=3, sticky="w", pady=(5, 0))
    
    def create_right_panel(self, parent):
        panel = ttk.Frame(parent)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=2)  # Log takes more space
        panel.rowconfigure(0, weight=1)  # Queue takes less
        
        # Queue Status
        queue_frame = tk.LabelFrame(panel, text="Processing Queue", 
                                   bg=ModernTheme.SURFACE_COLOR, fg=ModernTheme.FG_COLOR,
                                   font=('Segoe UI', 11, 'bold'))
        queue_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        queue_frame.columnconfigure(0, weight=1)
        queue_frame.rowconfigure(1, weight=1)
        
        # Progress
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(queue_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        self.status_label = tk.Label(queue_frame, text="Idle", bg=ModernTheme.SURFACE_COLOR, 
                                    fg=ModernTheme.ACCENT_COLOR, font=('Segoe UI', 10, 'bold'))
        self.status_label.grid(row=1, column=0, pady=5)
        
        # Current job info
        self.current_job_label = tk.Label(queue_frame, text="No active job", 
                                         bg=ModernTheme.SURFACE_COLOR, fg=ModernTheme.FG_COLOR,
                                         wraplength=300)
        self.current_job_label.grid(row=2, column=0, pady=5)
        
        # Control buttons
        btn_frame = ttk.Frame(queue_frame)
        btn_frame.grid(row=3, column=0, pady=5)
        ttk.Button(btn_frame, text="⏸ Pause", command=self.pause_processing).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="⏹ Stop", command=self.stop_processing).pack(side=tk.LEFT, padx=2)
        
        # Log Console
        log_frame = tk.LabelFrame(panel, text="Activity Log", 
                                 bg=ModernTheme.SURFACE_COLOR, fg=ModernTheme.FG_COLOR,
                                 font=('Segoe UI', 11, 'bold'))
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, state="disabled",
            bg=ModernTheme.BG_COLOR, fg=ModernTheme.FG_COLOR,
            font=('Consolas', 9), padx=5, pady=5
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        
        # Tag colors
        self.log_text.tag_configure("info", foreground=ModernTheme.FG_COLOR)
        self.log_text.tag_configure("success", foreground=ModernTheme.SUCCESS_COLOR)
        self.log_text.tag_configure("error", foreground=ModernTheme.ERROR_COLOR)
        self.log_text.tag_configure("warning", foreground=ModernTheme.WARNING_COLOR)
        
        return panel
    
    def create_status_bar(self):
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W,
                                  bg=ModernTheme.SURFACE_COLOR, fg=ModernTheme.FG_COLOR)
        self.status_bar.grid(row=1, column=0, sticky="ew")
        
        # GPU Info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.status_bar.config(text=f"GPU: {gpu_name} | VRAM: {vram:.1f}GB | Model: Not Loaded")
    
    def log(self, message, level="info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n", level)
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
    
    def load_model_dialog(self):
        if not LUXTTS_AVAILABLE:
            messagebox.showerror("Error", "LuxTTS not installed!\nRun: pip install zipvoice")
            return
        
        threading.Thread(target=self._load_model, daemon=True).start()
    
    def _load_model(self):
        try:
            self.log("Loading model... This may take a moment.", "info")
            self.update_status("Loading Model...", 0)
            
            device = self.device_var.get()
            threads = self.threads_var.get()
            
            kwargs = {"device": device}
            if device == "cpu":
                kwargs["threads"] = threads
            
            self.model = LuxTTS(self.model_id_var.get(), **kwargs)
            
            # Apply optimizations
            if self.fp16_var.get() and device == "cuda":
                self.model = self.model.half()
                self.log("Enabled FP16 mode", "success")
            
            if self.compile_var.get() and hasattr(torch, 'compile'):
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self.log("Model compiled (PyTorch 2.0+)", "success")
            
            self.log(f"Model loaded successfully on {device.upper()}", "success")
            self.update_status("Ready", 0)
            
            # Update status bar
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / 1e9
                self.status_bar.config(text=f"GPU: {torch.cuda.get_device_name(0)} | VRAM Used: {vram_used:.1f}GB | Model: Loaded")
        
        except Exception as e:
            self.log(f"Error loading model: {str(e)}", "error")
            self.update_status("Error", 0)
            messagebox.showerror("Model Error", str(e))
    
    def browse_reference(self):
        filename = filedialog.askopenfilename(
            title="Select Reference Audio",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a"), ("All Files", "*.*")]
        )
        if filename:
            self.ref_path_var.set(filename)
            self.log(f"Selected reference: {Path(filename).name}", "info")
    
    def on_drop_reference(self, event):
        files = self.root.tk.splitlist(event.data)
        if files:
            self.ref_path_var.set(files[0])
            self.log(f"Dropped reference: {Path(files[0]).name}", "info")
    
    def browse_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_text_file(self):
        filename = filedialog.askopenfilename(
            title="Select Text File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if filename:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            self.text_input.delete(1.0, tk.END)
            self.text_input.insert(1.0, content)
            self.log(f"Loaded text from {Path(filename).name}", "info")
    
    def queue_single_generation(self):
        if not self.model:
            messagebox.showwarning("Warning", "Please load the model first!")
            return
        
        text = self.text_input.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter text to synthesize!")
            return
        
        ref_audio = self.ref_path_var.get()
        if not ref_audio or not Path(ref_audio).exists():
            messagebox.showwarning("Warning", "Please select a valid reference audio file!")
            return
        
        # Create job
        job = {
            "type": "single",
            "text": text,
            "ref_audio": ref_audio,
            "output": Path(self.output_dir_var.get()) / self.output_name_var.get(),
            "params": self.get_generation_params()
        }
        
        self.processing_queue.put(job)
        self.log(f"Queued: {job['output'].name}", "info")
        self.update_queue_display()
    
    def get_generation_params(self):
        return {
            "num_steps": int(self.param_vars["num_steps"].get()),
            "t_shift": self.param_vars["t_shift"].get(),
            "speed": self.param_vars["speed"].get(),
            "return_smooth": self.return_smooth_var.get(),
            "rms": self.rms_var.get(),
            "ref_duration": self.ref_dur_var.get()
        }
    
    def add_batch_files(self):
        files = filedialog.askopenfilenames(
            title="Select Text Files",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        for f in files:
            self.add_batch_item(f)
    
    def add_batch_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with Text Files")
        if folder:
            for txt_file in Path(folder).glob("*.txt"):
                self.add_batch_item(str(txt_file))
    
    def add_batch_item(self, text_file):
        # Auto-find matching reference audio or use default
        ref_audio = self.ref_path_var.get() or "default.wav"
        
        output_name = self.batch_pattern_var.get().format(
            text_name=Path(text_file).stem,
            ref_name=Path(ref_audio).stem,
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        
        item_id = self.batch_tree.insert("", "end", values=(
            "Queued", Path(text_file).name, Path(ref_audio).name, 
            output_name, "Default"
        ))
        
        job = {
            "type": "batch",
            "id": item_id,
            "text_file": text_file,
            "ref_audio": ref_audio,
            "output": Path(self.output_dir_var.get()) / output_name,
            "params": self.get_generation_params()
        }
        self.processing_queue.put(job)
        self.update_queue_display()
    
    def clear_batch(self):
        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)
        # Clear queue
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                break
        self.update_queue_display()
    
    def remove_batch_item(self):
        selected = self.batch_tree.selection()
        if selected:
            self.batch_tree.delete(selected[0])
    
    def edit_batch_params(self):
        # Simplified - would open dialog in full implementation
        pass
    
    def show_batch_menu(self, event):
        item = self.batch_tree.identify_row(event.y)
        if item:
            self.batch_tree.selection_set(item)
            self.batch_menu.post(event.x_root, event.y_root)
    
    def start_batch(self):
        if not self.model:
            messagebox.showwarning("Warning", "Please load the model first!")
            return
        self.log("Starting batch processing...", "info")
    
    def process_queue(self):
        """Background thread that processes the queue"""
        while True:
            try:
                job = self.processing_queue.get(timeout=1)
                if job is None or self.stop_requested:
                    continue
                
                self.is_processing = True
                self.root.after(0, lambda: self.update_status("Processing...", 50))
                
                if job["type"] == "single":
                    self.process_single(job)
                elif job["type"] == "batch":
                    self.process_batch_item(job)
                
                self.root.after(0, self.update_queue_display)
                
            except queue.Empty:
                if self.is_processing:
                    self.is_processing = False
                    self.root.after(0, lambda: self.update_status("Ready", 0))
                continue
            except Exception as e:
                self.root.after(0, lambda: self.log(f"Processing error: {e}", "error"))
    
    def process_single(self, job):
        try:
            self.root.after(0, lambda: self.current_job_label.config(
                text=f"Generating: {job['output'].name}"
            ))
            
            # Encode prompt (with caching)
            cache_key = f"{job['ref_audio']}_{job['params']['rms']}_{job['params']['ref_duration']}"
            if hasattr(self, '_prompt_cache') and self.cache_prompts_var.get():
                encoded_prompt = self._prompt_cache.get(cache_key)
            else:
                self._prompt_cache = {}
                encoded_prompt = None
            
            if encoded_prompt is None:
                encoded_prompt = self.model.encode_prompt(
                    job['ref_audio'],
                    duration=job['params']['ref_duration'],
                    rms=job['params']['rms']
                )
                if self.cache_prompts_var.get():
                    self._prompt_cache[cache_key] = encoded_prompt
            
            # Generate
            final_wav = self.model.generate_speech(
                job['text'],
                encoded_prompt,
                num_steps=job['params']['num_steps'],
                t_shift=job['params']['t_shift'],
                speed=job['params']['speed'],
                return_smooth=job['params']['return_smooth']
            )
            
            # Save
            final_wav = final_wav.numpy().squeeze()
            job['output'].parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(job['output']), final_wav, 48000)
            
            self.root.after(0, lambda: self.log(f"Saved: {job['output']}", "success"))
            
        except Exception as e:
            self.root.after(0, lambda: self.log(f"Error: {str(e)}", "error"))
    
    def process_batch_item(self, job):
        try:
            # Read text file
            with open(job['text_file'], 'r', encoding='utf-8') as f:
                text = f.read()
            
            self.root.after(0, lambda: self.batch_tree.set(job['id'], "status", "Processing"))
            
            # Process
            encoded_prompt = self.model.encode_prompt(
                job['ref_audio'],
                duration=job['params']['ref_duration'],
                rms=job['params']['rms']
            )
            
            final_wav = self.model.generate_speech(
                text, encoded_prompt,
                num_steps=job['params']['num_steps'],
                t_shift=job['params']['t_shift'],
                speed=job['params']['speed'],
                return_smooth=job['params']['return_smooth']
            )
            
            final_wav = final_wav.numpy().squeeze()
            job['output'].parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(job['output']), final_wav, 48000)
            
            self.root.after(0, lambda: self.batch_tree.set(job['id'], "status", "Done"))
            self.root.after(0, lambda: self.log(f"Batch complete: {job['output'].name}", "success"))
            
        except Exception as e:
            self.root.after(0, lambda: self.batch_tree.set(job['id'], "status", "Error"))
            self.root.after(0, lambda: self.log(f"Batch error: {str(e)}", "error"))
    
    def update_status(self, text, progress):
        self.status_label.config(text=text)
        self.progress_var.set(progress)
        self.status_bar.config(text=text)
    
    def update_queue_display(self):
        size = self.processing_queue.qsize()
        if size > 0:
            self.status_label.config(text=f"Queue: {size} items")
    
    def pause_processing(self):
        self.log("Pause requested (implemented via queue delay)", "warning")
    
    def stop_processing(self):
        self.stop_requested = True
        self.log("Stop requested - finishing current job...", "warning")
        self.root.after(1000, self.reset_stop_flag)
    
    def reset_stop_flag(self):
        self.stop_requested = False
    
    def on_device_change(self, event=None):
        device = self.device_var.get()
        if device == "cuda" and not torch.cuda.is_available():
            messagebox.showwarning("Warning", "CUDA not available on this system!")
            self.device_var.set("cpu")
    
    def optimize_gpu(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.log("GPU cache cleared", "success")
    
    def clear_cache(self):
        if hasattr(self, '_prompt_cache'):
            self._prompt_cache.clear()
        self.log("Prompt cache cleared", "info")
    
    def load_settings(self):
        if self.settings_file.exists():
            try:
                with open(self.settings_file) as f:
                    settings = json.load(f)
                # Apply settings
                if 'model_id' in settings:
                    self.model_id_var = tk.StringVar(value=settings['model_id'])
                if 'output_dir' in settings:
                    self.output_dir_var = tk.StringVar(value=settings['output_dir'])
            except:
                pass
    
    def save_settings(self):
        settings = {
            'model_id': getattr(self, 'model_id_var', tk.StringVar(value="YatharthS/LuxTTS")).get(),
            'output_dir': getattr(self, 'output_dir_var', tk.StringVar(value=str(Path.home() / "LuxTTS_Output"))).get(),
            'device': self.device_var.get() if hasattr(self, 'device_var') else 'cpu'
        }
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f)
    
    def export_settings(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")]
        )
        if filename:
            self.save_settings()
            # Copy to destination
            import shutil
            shutil.copy(self.settings_file, filename)
            self.log(f"Settings exported to {filename}", "success")
    
    def import_settings(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if filename:
            with open(filename) as f:
                settings = json.load(f)
            # Apply settings
            self.log("Settings imported", "success")
    
    def on_closing(self):
        self.save_settings()
        self.stop_requested = True
        self.root.destroy()

def main():
    # Check for tkinterdnd2 for drag and drop support
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except ImportError:
        root = tk.Tk()
        print("Note: Install tkinterdnd2 for drag-and-drop support: pip install tkinterdnd2")
    
    app = LuxTTSGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()