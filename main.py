import tkinter as tk
from tkinter import filedialog, PhotoImage, ttk
import customtkinter
from utils import metadata, analyze, plot, simulate
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from PIL import Image, ImageTk
import cProfile

factor=0.6

class Application(tk.Tk):
    """
    Main application
    """
    def __init__(self):
        super().__init__()
        self.title("MARIPoSA")
        self.after(200, lambda: self.iconphoto(False, PhotoImage(file="other/MARIPoSA_icon.png")))
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        factor = 0.6
        self.width = int(screen_width * factor)
        self.height = int(screen_height * factor)
        x = (screen_width - self.width) // 2
        y = (screen_height - self.height) // 2
        self.geometry(f'{self.width}x{self.height}+{x}+{y}')

        # color pallete
        bg_color = '#FDFDFF'
        bg_color2 = '#EBEBFF' #radio hover
        text_color = '#393D3F'
        button_bg_color0 = '#CBCDE2'
        button_bg_color2 = '#B1B4D3' #hover
        button_bg_color1 = '#FFEF9F' #press
        button_text_color = '#161827'

        self.configure(bg=bg_color)  # slightly lighter than your current

        style = ttk.Style(self)
        style.theme_use('clam')

        style.configure('TLabel',
                        font=('Helvetica', 14),
                        background=bg_color,
                        foreground=text_color
                        )

        style.configure('TFrame',
                        font=('Helvetica', 14),
                        background=bg_color,
                        foreground=text_color
                        )

        style.configure('TButton',
                        font=('Helvetica', 14),
                        padding=8,
                        foreground=button_text_color,
                        background=button_bg_color0  # modern blue
                        )
        style.map('TButton',
                  foreground=[('pressed', button_text_color), ('active', button_text_color)],
                  background=[('pressed', button_bg_color1), ('active', button_bg_color2)]
                  )

        style.configure('TRadiobutton',
                        font=('Helvetica', 14),
                        padding=8,
                        background=bg_color,
                        foreground=text_color
                        )
        style.map('TRadiobutton',
                  foreground=[('active', text_color)],
                  background=[('active', bg_color2)]
                  )
        style.map('TCheckbutton',
                  foreground=[('active', text_color)],
                  background=[('active', bg_color2)]
                  )

        style.configure('TRadiobutton',
                        font=('Helvetica', 14),
                        padding=8,
                        background=bg_color,
                        foreground=text_color
                        )
        
        self.projectstart_choice = tk.StringVar(value="New project")
        self.datatype = tk.StringVar(value="B-SOiD")
        self.config = None
        self.config_path = None
        self.project_name = None
        self.plot_window = None
        self.plots_generated = 0
        self.option_5a = tk.StringVar(value="")
        self.option_5a_plot = tk.StringVar(value="")
        self.option_5b = tk.StringVar(value="")
        self.option_5c = tk.StringVar(value="")
        self.option_6a = tk.StringVar(value="")
        self.option_simmode = tk.StringVar(value="multivariate_normal")
        self.window1_start()
        self.log_text = ""
        self.log_window_ref = None
        self.log_text_widget = None

    def try_int(self, value):
        """
        Int, but error window if not possible
        :return:
        """
        try:
            return int(value)
        except Exception as e:
            self.error_window(f"A field which should be an integer numeric had non-numeric value: {value}")

    def try_float(self, value):
        """
        Int, but error window if not possible
        :return:
        """
        try:
            return float(value)
        except Exception as e:
            self.error_window(f"A field which should be a float numeric had non-numeric value: {value}")

    def clear_window(self):
        """
        Clear window
        :return:
        """
        for widget in self.winfo_children():
            if not isinstance(widget, tk.Toplevel):
                widget.destroy()

    def error_window(self, error_text):
        bg_color = '#D6D6FF'
        text_color = '#393D3F'

        error_win = tk.Toplevel()
        error_win.title("Error")
        error_win.configure(bg=bg_color)
        error_win.resizable(False, False)

        # Container frame
        container = ttk.Frame(error_win, padding=10)
        container.pack(fill="both", expand=True)

        label = ttk.Label(container, text="Error:", foreground="red", style="Error.TLabel")
        label.pack(pady=(0, 5))

        error_message = tk.Text(
            container,
            height=4,
            wrap="word",
            bg=bg_color,
            fg=text_color,
            font=('Helvetica', 14),
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            insertbackground=text_color
        )
        error_message.insert("1.0", error_text)
        error_message.config(state="disabled")
        error_message.pack(fill="both", expand=True, pady=(0, 10))

        ok_button = ttk.Button(container, text="OK", command=error_win.destroy)
        ok_button.pack()

        # Set a fixed width (e.g., 250px) but keep the height dynamic
        custom_width = 300  # Adjust this value to make it skinnier or wider
        error_win.update_idletasks()
        height = error_win.winfo_height()  # Get the current height of the window
        x = (error_win.winfo_screenwidth() // 2) - (custom_width // 2)  # Center horizontally
        y = (error_win.winfo_screenheight() // 2) - (height // 2)  # Center vertically
        error_win.geometry(f"{custom_width}x{height}+{x}+{y}")

        error_win.transient()
        error_win.grab_set()
        error_win.focus()


    def append_log(self, event_log):
        self.log_text += event_log + "\n\n"
        if self.log_text_widget:
            self.log_text_widget.config(state="normal")
            self.log_text_widget.delete("1.0", "end")
            self.log_text_widget.insert("1.0", self.log_text)
            self.log_text_widget.config(state="disabled")

    def log_window(self):
        # Avoid creating multiple windows
        if self.log_window_ref and self.log_window_ref.winfo_exists():
            self.log_window_ref.lift()
            return

        bg_color = '#ADADFF'
        text_color = '#203A18'

        log_win = tk.Toplevel(self)
        log_win.title("Command Log")
        log_win.configure(bg=bg_color)
        log_win.resizable(True, True)
        self.log_window_ref = log_win  # Keep reference

        container = ttk.Frame(log_win, padding=10)
        container.pack(fill="both", expand=True)

        log = tk.Text(
            container,
            wrap="word",
            bg=bg_color,
            fg=text_color,
            font=('Courier', 14),
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            insertbackground=text_color,
            state="disabled"
        )
        log.pack(fill="both", expand=True, pady=(0, 10))

        self.log_text_widget = log  # Save widget for updates
        self.append_log("")  # Trigger initial update

        save_button = ttk.Button(container, text="Save log.txt", command=self.save_log)
        save_button.pack()

    def save_log(self):
        file_path = filedialog.asksaveasfilename(defaultextension='.txt',
                                                 filetypes=[("Text files", "*.txt"),
                                                            ("All Files", "*.*")])
        if file_path:
            with open(file_path, "w") as f:
                f.write(self.log_text)

    def window_browse(self, item_path_entry, type="file"):
        """
        Function to browse files
        :param item_path_entry:
        :param type:
        :return:
        """
        if type == "file":
            filename = filedialog.askopenfilename()
            item_path_entry.delete(0, tk.END)
            item_path_entry.insert(0, filename)
        elif type == "directory":
            dirname = filedialog.askdirectory()
            item_path_entry.delete(0, tk.END)
            item_path_entry.insert(0, dirname)

    def window1_start(self):
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        pil_img = Image.open('other/MARIPoSA_icon.png').resize((130, 130))
        self.logo_img = ImageTk.PhotoImage(pil_img)
        self.projectstart_choice = tk.StringVar(value="New project")
        ttk.Label(self, text="MARIPoSA",
                               font=('Helvetica', 32, "bold")).place(x=int(self.width*0.5), y=int(self.height*0.1),anchor=tk.CENTER)
        ttk.Label(self, image=self.logo_img).place(x=int(self.width*0.5), y=int(self.height*0.25),anchor=tk.CENTER)
        ttk.Label(self, text="Would you like to start a new project or load a previous project?",
                               ).place(x=int(self.width*0.5), y=int(self.height*0.4),anchor=tk.CENTER)

        # Radio buttons for starting new project or loading old project
        projectstart_options = ["New project","Load previous"]
        grid_last = 0
        for option in projectstart_options:
            ttk.Radiobutton(self, text=option, variable=self.projectstart_choice, value=option,
                                         ).place(x=int(self.width*0.2), y=int(self.height*(0.5+0.1*grid_last)))
            grid_last = grid_last + 1

        # File path entry and Browse button
        config_path_entry = ttk.Entry(self)
        config_path_entry.insert(0, "/path/to/config.yaml")
        config_path_entry.place(x=int(self.width*0.4), y=int(self.height*0.6))
        ttk.Button(self,
                                text="Browse",
                                command=lambda: self.window_browse(config_path_entry, type="file")).place(x=int(self.width*0.6), y=int(self.height*0.6))

        ttk.Button(self, text="▶", command=lambda: self.make_or_load_project(config_path_entry.get())
                   ).place(x=int(self.width*0.8), y=int(self.height*0.9))

    def make_or_load_project(self, config_path):
        if type(self.projectstart_choice)!=str:
            self.projectstart_choice = self.projectstart_choice.get()
        self.config_path = config_path
        if self.projectstart_choice == "New project":
            self.window1b_projecttype()
        elif self.projectstart_choice == "Load previous":
            if os.path.exists(config_path):
                self.load_project()
            else:
                self.error_window("That path does not exist, please choose a different one.")
        else:
            print(self.projectstart_choice)

    def window1b_projecttype(self):
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        self.projectstart_choice = tk.StringVar(value="New project")
        ttk.Label(self, text="Create a new project",
                               font=('Helvetica', 32, "bold")).place(x=int(self.width*0.5), y=int(self.height*0.1),anchor=tk.CENTER)
        ttk.Label(self, text="Will the data type for this new dataset be pose estimation or pose segmentation?",
                               ).place(x=int(self.width*0.5), y=int(self.height*0.3),anchor=tk.CENTER)

        ttk.Button(self, text="Pose estimation\n\n(DeepLabCut, SLEAP, or OpenFace)",
                                command=lambda: self.window2_makeproject("pose_estimation"),
                                ).place(x=int(self.width*0.3), y=int(self.height*0.6),height=int(self.height*0.25),width=int(self.width*0.35),anchor=tk.CENTER)
        ttk.Button(self, text="Pose segmentation\n\n(B-SOiD, VAME, or Keypoint-MoSeq)",
                                command=lambda: self.window2_makeproject("pose_segmentation"),
                                ).place(x=int(self.width*0.7), y=int(self.height*0.6),height=int(self.height*0.25),width=int(self.width*0.35),anchor=tk.CENTER)

        ttk.Button(self, text="◀", command=self.window1_start,
                                ).place(x=int(self.width * 0.1), y=int(self.height * 0.9))

    def window2_makeproject(self,type):
        if type=="pose_estimation":
            self.clear_window()
            ttk.Button(self, text="Log", command=self.log_window,
                                    ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)

            ttk.Label(self, text="Create a new pose estimation project",
                                   font=('Helvetica', 32, "bold")).place(x=int(self.width*0.5), y=int(self.height*0.08),anchor=tk.CENTER)

            ttk.Label(self, text="Project name",
                                   ).place(x=int(self.width*0.2), y=int(self.height*0.2))
            ttk.Entry(self).place(x=int(self.width*0.5), y=int(self.height*0.2))

            # Enter data directory
            ttk.Label(self, text="Path to data directory",
                                   ).place(x=int(self.width*0.2), y=int(self.height*0.3))
            data_path_entry = ttk.Entry(self)
            data_path_entry.place(x=int(self.width*0.5), y=int(self.height*0.3))
            browse_button = ttk.Button(self, text="Browse",
                                                    command=lambda: self.window_browse(data_path_entry, type="directory"))
            browse_button.place(x=int(self.width*0.7), y=int(self.height*0.3))

            #Info about data
            ttk.Label(self, text="Source of data for new project",
                                   ).place(x=int(self.width*0.2), y=int(self.height*0.4))

            # Radio buttons for starting new project or loading old project
            datatype_options = ["DeepLabCut", "SLEAP", "OpenFace"]
            for r, option in enumerate(datatype_options):
                radio_btn = ttk.Radiobutton(self, text=option, variable=self.datatype, value=option,
                                                         )
                radio_btn.place(x=int(self.width*0.5), y=int(self.height*(0.4+r*0.1)))

            ttk.Label(self, text="Frames per second",
                                   ).place(x=int(self.width*0.2), y=int(self.height*0.7))
            fps = ttk.Entry(self)
            fps.place(x=int(self.width*0.5), y=int(self.height*0.7))

            # Enter project directory
            ttk.Label(self, text="Destination path for MARIPoSA output",
                                   ).place(x=int(self.width*0.2), y=int(self.height*0.8))
            project_path_entry = ttk.Entry(self)
            project_path_entry.place(x=int(self.width*0.5), y=int(self.height*0.8))
            browse_button = ttk.Button(self, text="Browse",
                                                    command=lambda: self.window_browse(project_path_entry,
                                                                                       type="directory"))
            browse_button.place(x=int(self.width*0.7), y=int(self.height*0.8))

            # Back to the initial view
            ttk.Button(self, text="◀", command=self.window1_start,
                                    ).place(x=int(self.width*0.1), y=int(self.height*0.9))
            ttk.Button(self, text="▶",
                                    command=lambda: self.create_PE_project(
                                        project_name.get(), data_path_entry.get(), self.datatype.get(),
                                        project_path_entry.get(),fps.get()),
                                    ).place(x=int(self.width*0.8), y=int(self.height*0.9))
        if type == "pose_segmentation":
            self.clear_window()
            ttk.Button(self, text="Log", command=self.log_window,
                       ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)

            ttk.Label(self, text="Create a new pose segmentation project",
                                   font=('Helvetica', 32, "bold")).place(x=int(self.width * 0.5),
                                                                         y=int(self.height * 0.08), anchor=tk.CENTER)

            ttk.Label(self, text="Project name",
                                   ).place(x=int(self.width * 0.2), y=int(self.height * 0.2))
            project_name = ttk.Entry(self)
            project_name.place(x=int(self.width * 0.5), y=int(self.height * 0.2))

            # Enter data directory
            ttk.Label(self, text="Path to data directory",
                                   ).place(x=int(self.width * 0.2), y=int(self.height * 0.28))
            data_path_entry = ttk.Entry(self)
            data_path_entry.place(x=int(self.width * 0.5), y=int(self.height * 0.28))
            browse_button = ttk.Button(self, text="Browse",
                                                    command=lambda: self.window_browse(data_path_entry,
                                                                                       type="directory"))
            browse_button.place(x=int(self.width * 0.7), y=int(self.height * 0.28))

            # Info about data
            ttk.Label(self, text="Source of data for new project",
                                   ).place(x=int(self.width * 0.2), y=int(self.height * 0.36))

            # Radio buttons for starting new project or loading old project
            datatype_options = ["B-SOiD", "VAME", "Keypoint-MoSeq"]
            for r, option in enumerate(datatype_options):
                ttk.Radiobutton(self, text=option, variable=self.datatype, value=option,
                                                         ).place(x=int(self.width * 0.5), y=int(self.height * (0.36 + r * 0.08)))

            ttk.Label(self, text="Frames per second",
                                   ).place(x=int(self.width * 0.2), y=int(self.height * 0.6))
            fps = ttk.Entry(self)
            fps.place(x=int(self.width * 0.5), y=int(self.height * 0.6))

            ttk.Label(self, text="Number of modules",
                                   ).place(x=int(self.width * 0.2), y=int(self.height * 0.68))
            n_modules = ttk.Entry(self)
            n_modules.place(x=int(self.width * 0.5), y=int(self.height * 0.68))

            # Enter project directory
            ttk.Label(self, text="Destination path for MARIPoSA output",
                                   ).place(x=int(self.width * 0.2), y=int(self.height * 0.75))
            project_path_entry = ttk.Entry(self)
            project_path_entry.place(x=int(self.width * 0.5), y=int(self.height * 0.75))
            browse_button = ttk.Button(self, text="Browse",
                                                    command=lambda: self.window_browse(project_path_entry,
                                                                                       type="directory"))
            browse_button.place(x=int(self.width * 0.7), y=int(self.height * 0.75))

            # Back to the initial view
            ttk.Button(self, text="◀", command=self.window1_start,
                                    ).place(x=int(self.width * 0.1), y=int(self.height * 0.9))
            ttk.Button(self, text="▶",
                                    command=lambda: self.create_PS_project(
                                        project_name.get(), data_path_entry.get(), self.datatype.get(),
                                        project_path_entry.get(), fps.get(), n_modules.get()),
                                    ).place(x=int(self.width * 0.8), y=int(self.height * 0.9))

    def create_PE_project(self, project_name, data_directory, data_source, output_directory,fps):
        metadata.create_PE_project(project_name, data_directory, data_source, output_directory,fps)
        self.append_log(f"metadata.create_PE_project('{project_name}', '{data_directory}', '{data_source}', '{output_directory}',{fps})")
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        self.project_name = datetime.now().strftime('%y%m%d_') + project_name
        self.config_path = output_directory + "/" + self.project_name + "/config_PE.yaml"
        config = metadata.load_project(self.config_path)
        self.config = config
        self.window3a_PE_menu()

    def create_PS_project(self, project_name, data_directory, data_source, output_directory,fps,n_modules):
        metadata.create_PS_project(project_name, data_directory, data_source, output_directory,fps,n_modules)
        self.append_log(f"metadata.create_PS_project('{project_name}', '{data_directory}', '{data_source}', '{output_directory}',{fps},{n_modules})")
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        self.project_name = datetime.now().strftime('%y%m%d_') + project_name
        self.config_path = output_directory + "/" + self.project_name + "/config_PS.yaml"
        config = metadata.load_project(self.config_path)
        self.config = config
        self.window3b_PS_menu()

    def load_project(self):
        try:
            config = metadata.load_project(self.config_path)
            self.config = config
            self.append_log(f"config = metadata.load_project('{self.config_path}')")
            if self.config["data_type"]=="Pose estimation":
                self.clear_window()
                self.window3a_PE_menu()
            elif self.config["data_type"]=="Pose segmentation":
                self.clear_window()
                self.window3b_PS_menu()
        except Exception as e:
            self.error_window("Problem loading config. Did you choose a wrong file? Or rename from config_<PS/PE>.yaml?")

    def load_project_BORIS(self):
        self.clear_window()
        config = metadata.load_project(self.config_path)
        self.append_log(f"config = metadata.load_project('{self.config_path}')")
        self.config = config
        self.window4b3_pose_vs_BORIS()

    def load_project_mainmenu(self):
        self.clear_window()
        config = metadata.load_project(self.config_path)
        self.append_log(f"config = metadata.load_project('{self.config_path}')")
        self.config = config
        if self.config["data_type"]=="Pose estimation":
            self.window3a_PE_menu()
        elif self.config["data_type"]=="Pose segmentation":
            self.window3b_PS_menu()

    def create_sidebar_widget(self):

        bar_width=self.width*0.25
        bar_width=int(bar_width)
        frame = tk.Frame(self, width=bar_width, height=self.height*0.95)
        frame.place(x=int(self.width * 0.01), y=int(self.height * 0.02))

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=0)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=0)

        # Canvas
        canvas = tk.Canvas(frame, width=bar_width,height=self.height*0.95)
        canvas.grid(row=0, column=0, sticky="nsew")

        # Vertical scrollbar
        v_scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=v_scrollbar.set)

        # Horizontal scrollbar
        h_scrollbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        canvas.configure(xscrollcommand=h_scrollbar.set)

        list_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=list_frame, anchor=tk.NW)

        ttk.Label(list_frame, text="Config path",  font=('Helvetica', 16, "bold"),anchor="w",width=bar_width).pack(side=tk.TOP, fill=tk.X,padx=5)
        ttk.Label(list_frame, text=self.config["project_directory"],anchor="w",width=bar_width).pack(side=tk.TOP, fill=tk.X,padx=5)
        ttk.Label(list_frame, text="Data source",  font=('Helvetica', 16, "bold"),anchor="w",width=bar_width).pack(side=tk.TOP, fill=tk.X,padx=5)
        ttk.Label(list_frame, text=self.config["data_source"],anchor="w",width=bar_width).pack(side=tk.TOP, fill=tk.X,padx=5)
        ttk.Label(list_frame, text="Subgroups (" + str(len(self.config["subgroups"].keys())) + ")",
                                font=('Helvetica', 16, "bold"),anchor="w",width=bar_width).pack(side=tk.TOP, fill=tk.X,padx=5)
        for key in list(self.config["subgroups"].keys()):
            label = ttk.Label(list_frame, text=str(key+" ("+str(len(self.config["subgroups"][key]))+")"),anchor="w",width=bar_width)
            label.pack(anchor=tk.W,padx=5)
        ttk.Label(list_frame, text="Project Files (" + str(len(self.config["project_files"])) + ")",
                                font=('Helvetica', 16, "bold"),anchor="w",width=bar_width).pack(side=tk.TOP, fill=tk.X,padx=5)
        for item in self.config["project_files"]:
            label = ttk.Label(list_frame, text=item, anchor="w", width=bar_width)
            label.pack(anchor=tk.W,padx=5)

        # Update scrollregion of the canvas
        def update_scrollregion(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        list_frame.bind("<Configure>", update_scrollregion)

        return frame

    def create_header(self, header_title, header_path=None):
        if header_path!=None:
            ttk.Label(self, text=header_path,
                                   font=('Helvetica', 12)).place(x=int(self.width*0.3), y=int(self.height*0.02))
        ttk.Label(self, text=header_title,
                               font=('Helvetica', 32, "bold")).place(x=int(self.width*0.65), y=int(self.height*0.09),anchor=tk.CENTER)
        ttk.Label(self, text="Project "+self.config["project_name"],
                               font=('Helvetica', 20, "bold")).place(x=int(self.width*0.65), y=int(self.height*0.16),anchor=tk.CENTER)

    def window3a_PE_menu(self):
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        self.create_sidebar_widget()
        self.create_header("Analysis Menu")

        menu_item_width=int(self.width*0.2)
        button_width=int(self.width*0.45)
        button_height=int(self.height*0.06)
        ttk.Label(self, text="Further configure project:",
                               ).place(x=int(self.width * 0.3), y=int(self.height * 0.2),width=menu_item_width,height=button_height)
        ttk.Button(self, text="Define subgroups within data",
                                command=self.window4a_define_subgroups,
                                ).place(x=int(self.width*0.5), y=int(self.height*0.2),width=button_width,height=button_height)

        ttk.Label(self, text="Analyze and visualize:",
                               ).place(x=int(self.width * 0.3), y=int(self.height * 0.44),width=menu_item_width,height=button_height)
        ttk.Button(self, text="Measure keypoint travel (e.g., locomotion)",
                                command=self.window5d_keypoint_displacement_menu,
                                ).place(x=int(self.width * 0.5), y=int(self.height * 0.44),width=button_width,height=button_height)
        if self.config["data_source"]=="OpenFace":
            last_menu_item_pos=0.51
            ttk.Button(self, text="Measure action units",
                       command=self.window5e_action_units_menu,
                       ).place(x=int(self.width * 0.5), y=int(self.height * last_menu_item_pos), width=button_width,
                               height=button_height)
        else:
            last_menu_item_pos=0.44
        ttk.Button(self, text="Embed and/or measure distance between groups",
                                command=self.window5b_embed_distance_menu,
                                ).place(x=int(self.width*0.5), y=int(self.height*(last_menu_item_pos+0.07)),width=button_width,height=button_height)
        ttk.Button(self, text="Classify and/or regress conditions",
                                command=self.window5c_classify_regress_menu,
                                ).place(x=int(self.width*0.5), y=int(self.height*(last_menu_item_pos+0.14)),width=button_width,height=button_height)

        # ttk.Label(self, text="Model and simulate:",
        #                        ).place(x=int(self.width * 0.3), y=int(self.height * 0.75),width=menu_item_width,height=button_height)
        # ttk.Button(self, text="Simulate module labels, usage, or transitions",
        #                         command=self.window6a_simulate_menu,
        #                         ).place(x=int(self.width*0.5), y=int(self.height*0.75),width=button_width,height=button_height)
        # ttk.Button(self, text="Fit curve to within-session pose data",
        #                         command=self.window6b_fit_curve_menu,
        #                         ).place(x=int(self.width*0.5), y=int(self.height*0.82),width=button_width,height=button_height)
        # ttk.Button(self, text="Get cumulative distribution function from real or simulated behavior",
        #                         command=self.window6c_cdf_menu,
        #                         ).place(x=int(self.width*0.5), y=int(self.height*0.89),width=button_width,height=button_height)

        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window3b_PS_menu(self):
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        self.create_sidebar_widget()
        self.create_header("Analysis Menu")

        menu_item_width=int(self.width*0.2)
        button_width=int(self.width*0.45)
        button_height=int(self.height*0.06)
        ttk.Label(self, text="Further configure project:",
                               ).place(x=int(self.width * 0.3), y=int(self.height * 0.2),width=menu_item_width,height=button_height)
        ttk.Button(self, text="Define subgroups within data",
                                command=self.window4a_define_subgroups,
                                ).place(x=int(self.width*0.5), y=int(self.height*0.2),width=button_width,height=button_height)
        ttk.Button(self, text="Compare modules to manual scoring",
                                command=self.window4b_pose_vs_BORIS,
                                ).place(x=int(self.width*0.5), y=int(self.height*0.27),width=button_width,height=button_height)
        ttk.Button(self, text="Manually combine pose modules",
                                command=self.window4c_manual_combine,
                                ).place(x=int(self.width*0.5), y=int(self.height*0.34),width=button_width,height=button_height)

        ttk.Label(self, text="Analyze and visualize:",
                               ).place(x=int(self.width * 0.3), y=int(self.height * 0.44),width=menu_item_width,height=button_height)
        ttk.Button(self, text="Get streamlined pose module label sequence file",
                                command=self.window5d_get_modlabels,
                                ).place(x=int(self.width*0.5), y=int(self.height*0.44),width=button_width,height=button_height)
        ttk.Button(self, text="Measure pose module usage and transitions",
                                command=self.window5a_usage_transitions_menu,
                                ).place(x=int(self.width*0.5), y=int(self.height*0.51),width=button_width,height=button_height)
        ttk.Button(self, text="Embed and/or measure distance between groups",
                                command=self.window5b_embed_distance_menu,
                                ).place(x=int(self.width*0.5), y=int(self.height*0.58),width=button_width,height=button_height)
        ttk.Button(self, text="Classify and/or regress conditions",
                                command=self.window5c_classify_regress_menu,
                                ).place(x=int(self.width*0.5), y=int(self.height*0.65),width=button_width,height=button_height)

        ttk.Label(self, text="Model and simulate:",
                               ).place(x=int(self.width * 0.3), y=int(self.height * 0.75),width=menu_item_width,height=button_height)
        ttk.Button(self, text="Simulate module labels, usage, or transitions",
                                command=self.window6a_simulate_menu,
                                ).place(x=int(self.width*0.5), y=int(self.height*0.75),width=button_width,height=button_height)
        # ttk.Button(self, text="Fit curve to within-session pose data",
        #                         command=self.window6b_fit_curve_menu,
        #                         ).place(x=int(self.width*0.5), y=int(self.height*0.82),width=button_width,height=button_height)
        # ttk.Button(self, text="Get cumulative distribution function from real or simulated behavior",
        #                         command=self.window6c_cdf_menu,
        #                         ).place(x=int(self.width*0.5), y=int(self.height*0.89),width=button_width,height=button_height)

        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window4a_define_subgroups(self):
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        self.create_sidebar_widget()
        self.create_header("Define Subgroups",header_path="Configuration ▶ Subgroup Definition")
        instruction_text = """For this step, you will need to manually edit the config file, which you should be able to access by pressing the 'Edit config.yaml' button below. In the file, there is a section called subgroups with all your files listed as such:

        subgroups:
          group1:
            - 'subj1_conditionA.csv'
            - 'subj2_conditionB.csv'

        and you can edit it to be as follows:

        subgroups:
          conditionA:
            - 'subj1_conditionA.csv'
          conditionB:
            - 'subj2_conditionB.csv'

        You can do it! You're doing great."""
        instruction_block = ttk.Label(self,text=instruction_text, wraplength=int(self.width * 0.65),justify=tk.LEFT)

        # Place the label in the window
        instruction_block.place(x=int(self.width * 0.3),y=int(self.height * 0.22),anchor=tk.NW)
        ttk.Button(self,
                                text="Edit config.yaml",
                                command=lambda: metadata.edit_config(self.config_path)).place(x=int(self.width*0.6), y=int(self.height*0.75))
        # Bottom back buttons
        ttk.Button(self, text="◀ back to analysis menu", command=self.load_project_mainmenu).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window4b_pose_vs_BORIS(self):
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        self.create_sidebar_widget()
        self.create_header("Compare Pose Data to Manual Scoring",header_path="Configure ▶ Manual scoring comparison")

        ttk.Button(self, text="Update config file with manual scoring info from BORIS",
                                command=self.window4b2_boris_config,
                                ).place(x=int(self.width*0.65), y=int(self.height*0.4,),height=int(self.width*0.1),width=int(self.width*0.6),anchor=tk.CENTER)
        ttk.Button(self, text="Get and plot pose module to BORIS comparison matrix",
                                command=self.window4b3_pose_vs_BORIS,
                                ).place(x=int(self.width*0.65), y=int(self.height*0.6),height=int(self.width*0.1),width=int(self.width*0.6),anchor=tk.CENTER)
        # Bottom back buttons
        ttk.Button(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window4b2_boris_config(self):
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        self.create_sidebar_widget()
        self.create_header("Compare Pose Data to Manual Scoring",header_path="Configure ▶ Manual scoring comparison ▶ Update config")

        instruction_text = """For this step, you will need to manually edit the config file, which you should be able to access by pressing the 'Edit config.yaml' button below.
        You'll have to edit the boris_directory and boris_to_pose_pairings."""
        instruction_block = ttk.Label(self,text=instruction_text, wraplength=int(self.width * 0.65),justify=tk.LEFT)

        instruction_block.place(x=int(self.width * 0.3),y=int(self.height * 0.3),anchor=tk.NW)
        ttk.Button(self,
                                text="Edit config.yaml",
                                command=lambda: metadata.edit_config(self.config["project_directory"] + "/config_PS.yaml")).place(x=int(self.width * 0.65),y=int(self.height * 0.6),anchor=tk.CENTER)
        # Bottom back buttons
        ttk.Button(self, text="◀ update config and go back to BORIS menu", command=self.window4b_pose_vs_BORIS,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.76))
        ttk.Button(self, text="◀ update config and go back to analysis menu", command=self.load_project_mainmenu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window4b3_pose_vs_BORIS(self):
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        self.create_sidebar_widget()
        self.create_header("Compare Pose Data to Manual Scoring",header_path="Configure ▶ Manual scoring comparison ▶ Plot comparison matrix")

        ttk.Label(self, text="Get a matrix showing overlap between pose modules and behaviors manually scored in BORIS.",
                               ).place(x=int(self.width*0.65), y=int(self.height*0.4),anchor=tk.CENTER)

        ttk.Button(self, text="Compare!",
                                command=lambda: self.plot_pose_vs_BORIS(),
                                ).place(x=int(self.width*0.85), y=int(self.height*0.9))
        # Bottom back buttons
        ttk.Button(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window4c_manual_combine(self):
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        self.create_sidebar_widget()
        self.create_header("Manually Combine Pose Modules",header_path="Configure ▶ Manually combine pose modules")

        instruction_text = """For this step, you will need to manually edit the config file, which you should be able to access by pressing the 'Edit config.yaml' button below.
        You'll have to edit the remappings section to contain, for each remapping, a list of modules to be remapped and what the combined class should be. Formatting should be as follows:
        remappings:
        - - [0,1,2,3]
          - 'walking'
        - - [4,5,6,7]
          - 'resting'
        - - [8]
          - 'grooming'"""
        instruction_block = ttk.Label(self,text=instruction_text, wraplength=int(self.width * 0.65),justify=tk.LEFT)

        instruction_block.place(x=int(self.width * 0.3),y=int(self.height * 0.3),anchor=tk.NW)
        ttk.Button(self,
                                text="Edit config.yaml",
                                command=lambda: metadata.edit_config(self.config["project_directory"] + "/config_PS.yaml")).place(x=int(self.width * 0.65),y=int(self.height * 0.7),anchor=tk.CENTER)
        # Bottom back buttons
        ttk.Button(self, text="◀ update config and go back to analysis menu", command=self.load_project_mainmenu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window5d_get_modlabels(self):
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        self.create_sidebar_widget()
        self.create_header("Get Streamlined Pose Module Labels File",header_path="Viz & Analyze ▶ Get Labels File")

        ttk.Label(self, text="Enter info about your analysis.",
                               ).place(x=int(self.width*0.3), y=int(self.height*0.3))

        ttk.Label(self, text="Start time (seconds)",
                               ).place(x=int(self.width*0.3), y=int(self.height*0.38))
        start = ttk.Entry(self)
        start.place(x=int(self.width*0.3), y=int(self.height*0.43))
        ttk.Label(self, text="End time (seconds)",
                               ).place(x=int(self.width*0.3), y=int(self.height*0.51))
        end = ttk.Entry(self)
        end.place(x=int(self.width*0.3), y=int(self.height*0.56))
        ttk.Label(self, text="Data to include",
                               ).place(x=int(self.width*0.65), y=int(self.height*0.38))
        dropdown = MultiDropDown(self,options=["all combined"]+list(self.config["subgroups"].keys()),
                                 x=int(self.width*0.65), y=int(self.height*0.43))
        dropdown.place(x=int(self.width*0.65), y=int(self.height*0.43))

        ttk.Button(self, text="Analyze & Save",
                                command=lambda: self.make_save_labels_df(self.try_int(start.get()),
                                                                            self.try_int(end.get()),
                                                                            dropdown.get_selected_options())
                                ).place(x=int(self.width*0.8), y=int(self.height*0.9))

        ttk.Button(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window5a_usage_transitions_menu(self):
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        self.create_sidebar_widget()
        self.create_header("Pose Usage and Transition Analysis Menu",header_path="Viz & Analyze ▶ Usage & Transitions")
        self.option_5a = tk.StringVar(value="")

        ttk.Radiobutton(self, text="Get pose usage and transitions",
                        command=lambda: self.window5a_usage_transitions_menu_update(),
                        variable=self.option_5a, value="Get usage tx",
                        ).place(x=int(self.width*0.3), y=int(self.height*0.2))
        ttk.Radiobutton(self, text="Plot pose usage and/or transitions",
                        command=lambda: self.window5a_usage_transitions_menu_update(),
                        variable=self.option_5a, value="Plot usage tx",
                        ).place(x=int(self.width*0.65), y=int(self.height*0.2))

        # Frame that will contain the dynamic content
        self.frame_5a = ttk.Frame(self)
        self.frame_5a.place(x=0, y=0, relwidth=1, relheight=1)
        self.frame_5a.lower()

        # Initial content
        self.window5a_usage_transitions_menu_update()

        ttk.Button(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))


    def window5a_usage_transitions_menu_update(self,pickle_path_prefill = "/path/to/usage.pickle",
                                               tx_pickle_path_prefill="/path/to/transitions.pickle",
                                               plot_option=None):

        # Clear existing content in the frame
        for widget in self.frame_5a.winfo_children():
            widget.destroy()

        # Load content based on the selected option
        selected = self.option_5a.get()

        if selected=="Get usage tx":
            ttk.Label(self.frame_5a, text="Enter info about your analysis.",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.3))

            ttk.Label(self.frame_5a, text="Start time (seconds)",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.38))
            start = ttk.Entry(self.frame_5a)
            start.place(x=int(self.width*0.3), y=int(self.height*0.43))
            ttk.Label(self.frame_5a, text="End time (seconds)",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.51))
            end = ttk.Entry(self.frame_5a)
            end.place(x=int(self.width*0.3), y=int(self.height*0.56))
            ttk.Label(self.frame_5a, text="Binsize (seconds; blank for no binning)",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.64))
            binsize = ttk.Entry(self.frame_5a)
            binsize.place(x=int(self.width*0.3), y=int(self.height*0.69))
            ttk.Label(self.frame_5a, text="Data to include",
                                   ).place(x=int(self.width*0.65), y=int(self.height*0.38))
            dropdown = MultiDropDown(self.frame_5a,options=["all combined"]+list(self.config["subgroups"].keys()),
                                     x=int(self.width*0.65), y=int(self.height*0.43))
            dropdown.place(x=int(self.width*0.65), y=int(self.height*0.43))
            save_option = tk.StringVar(value="scatter")
            save_options = ["pickle (for further use in MARIPoSA)",
                             "csv (for external use)",
                             "pickle AND csv"]
            style_vars = ["pickle", "csv", "both"]
            ttk.Label(self.frame_5a, text="How should the data be saved?",
                                   ).place(x=int(self.width*0.65), y=int(self.height*0.6))
            for s in range(len(style_vars)):
                radio_btn = ttk.Radiobutton(self.frame_5a, text=save_options[s],
                                                         variable=save_option, value=style_vars[s],
                                                         )
                radio_btn.place(x=int(self.width*0.65), y=int(self.height*(0.67+0.05*s)))

            ttk.Button(self.frame_5a, text="Analyze & Save",
                                    command=lambda: self.save_usage_transitions(self.try_int(start.get()),
                                                                                self.try_int(end.get()),
                                                                                dropdown.get_selected_options(),
                                                                                binsize.get(),
                                                                                save_to=save_option.get()),
                                    ).place(x=int(self.width*0.8), y=int(self.height*0.9))

        elif selected=="Plot usage tx":
            ttk.Label(self.frame_5a, text="What type of plot do you want to make?",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.27))
            ttk.Radiobutton(self.frame_5a, text="Usage comparison",
                                    command=lambda: self.window5a_usage_transitions_menu_update(plot_option="usage"),
                                    variable=self.option_5a_plot, value="usage",
                                    ).place(x=int(self.width*0.3), y=int(self.height*0.32))
            ttk.Radiobutton(self.frame_5a, text="Pairwise network\ncomparison",
                                    command=lambda: self.window5a_usage_transitions_menu_update(plot_option="network"),
                                    variable=self.option_5a_plot, value="network",
                                    ).place(x=int(self.width*0.55), y=int(self.height*0.32))
            ttk.Radiobutton(self.frame_5a, text="Within-session\ntime dynamic usage",
                                    command=lambda: self.window5a_usage_transitions_menu_update(plot_option="sandplot"),
                                    variable=self.option_5a_plot, value="sandplot",
                                    ).place(x=int(self.width*0.8), y=int(self.height*0.32))
            if plot_option=="usage":
                ttk.Label(self.frame_5a, text="Path to module usage .pickle:",
                                       ).place(x=int(self.width*0.3), y=int(self.height*0.42))
                pickle_path = ttk.Entry(self.frame_5a)
                pickle_path.insert(0, pickle_path_prefill)
                pickle_path.place(x=int(self.width*0.6), y=int(self.height*0.42))
                ttk.Button(self.frame_5a,
                                        text="Browse",
                                        command=lambda: self.window_browse(pickle_path, type="file")).place(x=int(self.width*0.8), y=int(self.height*0.42))
                ttk.Label(self.frame_5a, text="Colormap",
                                       ).place(x=int(self.width*0.65), y=int(self.height*0.5))
                color = ttk.Combobox(self.frame_5a, values=["jet", "cividis", "viridis", "magma"])
                color.place(x=int(self.width*0.75), y=int(self.height*0.5))
                color.set("jet")
                legend_bool = tk.BooleanVar()
                legend_button = ttk.Checkbutton(self.frame_5a, text="Legend", variable=legend_bool)
                legend_button.place(x=int(self.width*0.65), y=int(self.height*0.58))
                legend_bool.set(True)
                ttk.Label(self.frame_5a, text="Figure dimensions",
                                       ).place(x=int(self.width*0.65), y=int(self.height*0.66))
                ttk.Label(self.frame_5a, text="H",
                                       ).place(x=int(self.width*0.65), y=int(self.height*0.72))
                figH = ttk.Entry(self.frame_5a)
                figH.insert(0, "3")
                figH.place(x=int(self.width*0.67), y=int(self.height*0.72), relwidth=0.05)
                ttk.Label(self.frame_5a, text="W",
                                       ).place(x=int(self.width*0.8), y=int(self.height*0.72))
                figW = ttk.Entry(self.frame_5a)
                figW.insert(0, "6")
                figW.place(x=int(self.width*0.82), y=int(self.height*0.72), relwidth=0.05)
                # Choose style for plot
                ttk.Label(self.frame_5a, text="What should the usage plot style be?",
                                       ).place(x=int(self.width*0.3), y=int(self.height*0.5))
                style = tk.StringVar(value="scatter")
                style_options = ["Bar with scattered individual points",
                                 "Bar with standard error of the mean",
                                 "Points with standard error of the mean",
                                 "Stacked means"]
                style_vars = ["bar_scatter", "bar_error", "points", "stacked"]
                for s in range(len(style_vars)):
                    radio_btn = ttk.Radiobutton(self.frame_5a, text=style_options[s],
                                                             variable=style, value=style_vars[s],
                                                             )
                    radio_btn.place(x=int(self.width*0.3), y=int(self.height*(0.56+0.05*s)))
                ttk.Button(self.frame_5a, text="Plot it!",
                                        command=lambda: self.plot_usage(pickle_path.get(), style.get(), color.get(), legend_bool.get(), self.try_float(figH.get()), self.try_float(figW.get())),
                                        ).place(x=int(self.width*0.8), y=int(self.height*0.9))

            elif plot_option == "network":
                ttk.Label(self.frame_5a, text="Path to module usage .pickle:",
                                       ).place(x=int(self.width*0.3), y=int(self.height*0.42))
                pickle_path = ttk.Entry(self.frame_5a)
                pickle_path.insert(0, pickle_path_prefill)
                pickle_path.place(x=int(self.width*0.6), y=int(self.height*0.42))
                ttk.Button(self.frame_5a,
                                        text="Browse",
                                        command=lambda: self.window_browse(pickle_path, type="file")).place(x=int(self.width*0.8), y=int(self.height*0.42))
                ttk.Label(self.frame_5a, text="Path to module transitions .pickle:",
                                       ).place(x=int(self.width*0.3), y=int(self.height*0.5))
                tx_pickle_path = ttk.Entry(self.frame_5a)
                tx_pickle_path.insert(0, tx_pickle_path_prefill)
                tx_pickle_path.place(x=int(self.width*0.6), y=int(self.height*0.5))
                ttk.Button(self.frame_5a,
                                        text="Browse",
                                        command=lambda: self.window_browse(tx_pickle_path, type="file")).place(x=int(self.width*0.8), y=int(self.height*0.5))
                ttk.Label(self.frame_5a, text="Colormap",
                                       ).place(x=int(self.width*0.65), y=int(self.height*0.6))
                color = ttk.Combobox(self.frame_5a, values=["bwr", "seismic", "PiYG", "BrBG", "PRGn"])
                color.place(x=int(self.width*0.75), y=int(self.height*0.6))
                color.set("bwr")
                ttk.Button(self.frame_5a, text="Plot it!",
                                        command=lambda: self.plot_network(pickle_path.get(), tx_pickle_path.get(), color.get()),
                                        ).place(x=int(self.width*0.8), y=int(self.height*0.9))
            elif plot_option=="sandplot":
                ttk.Label(self.frame_5a, text="Path to module usage .pickle:",
                                       ).place(x=int(self.width*0.3), y=int(self.height*0.42))
                pickle_path = ttk.Entry(self.frame_5a)
                pickle_path.insert(0, pickle_path_prefill)
                pickle_path.place(x=int(self.width*0.6), y=int(self.height*0.42))
                ttk.Button(self.frame_5a,
                                        text="Browse",
                                        command=lambda: self.window_browse(pickle_path, type="file")).place(x=int(self.width*0.8), y=int(self.height*0.42))
                ttk.Button(self.frame_5a, text="Plot it!",
                                        command=lambda: self.plot_sandplot(pickle_path.get()),
                                        ).place(x=int(self.width*0.8), y=int(self.height*0.9))
            # Bottom back buttons
        ttk.Button(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))


    def window5b_embed_distance_menu(self,pickle_path_prefill = "/path/to/file.pickle"):
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        self.create_sidebar_widget()
        self.create_header("Embed and Distance Menu",header_path="Viz & Analyze ▶ Embed")

        if self.config["data_type"]=="Pose segmentation":
            pickle_text = "Path to module feature object (transitions or usage) .pickle:"
        elif self.config["data_type"]=="Pose estimation":
            pickle_text = "Path to keypoint travel object .pickle:"

        ttk.Label(self, text=pickle_text,
                               ).place(x=int(self.width*0.3), y=int(self.height*0.25))
        self.pickle_path_5b = ttk.Entry(self)
        self.pickle_path_5b.insert(0, pickle_path_prefill)
        self.pickle_path_5b.place(x=int(self.width*0.65), y=int(self.height*0.25))
        ttk.Button(self,
                                text="Browse",
                                command=lambda: self.window_browse(self.pickle_path_5b, type="file")).place(x=int(self.width*0.85), y=int(self.height*0.25))

        ttk.Label(self, text="Embed or measure distance between groups?",
                               ).place(x=int(self.width*0.3), y=int(self.height*0.4))
        ttk.Radiobutton(self, text="Embed",
                        command=lambda: self.window5b_embed_distance_menu_udpate(),
                        variable=self.option_5b, value="embed",
                        ).place(x=int(self.width * 0.3), y=int(self.height * 0.45))
        ttk.Radiobutton(self, text="Distance",
                        command=lambda: self.window5b_embed_distance_menu_udpate(),
                        variable=self.option_5b, value="distance",
                        ).place(x=int(self.width * 0.55), y=int(self.height * 0.45))

        # Frame that will contain the dynamic content
        self.frame_5b = ttk.Frame(self)
        self.frame_5b.place(x=0, y=0, relwidth=1, relheight=1)
        self.frame_5b.lower()

        # Initial content
        self.window5b_embed_distance_menu_udpate()

        if self.config["data_type"]=="Pose segmentation":
            destination = self.window3b_PS_menu
        elif self.config["data_type"]=="Pose estimation":
            destination = self.window3a_PE_menu
        ttk.Button(self, text="◀ back to analysis menu", command=destination).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window5b_embed_distance_menu_udpate(self):

        # Clear existing content in the frame
        for widget in self.frame_5b.winfo_children():
            widget.destroy()

        # Load content based on the selected option
        selected = self.option_5b.get()

        if selected=="embed":
            ttk.Label(self.frame_5b, text="Dimensionality reduction method",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.53))
            embedding_type = ttk.Combobox(self.frame_5b, values=["pca", "lda"])
            embedding_type.place(x=int(self.width*0.3), y=int(self.height*0.59))
            embedding_type.set("pca")
            ttk.Label(self.frame_5b, text="Colormap",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.65))
            color = ttk.Combobox(self.frame_5b, values=["jet", "cividis", "viridis", "magma"])
            color.place(x=int(self.width*0.3), y=int(self.height*0.71))
            color.set("jet")
            legend_bool = tk.BooleanVar()
            legend_button = ttk.Checkbutton(self.frame_5b, text="Legend", variable=legend_bool)
            legend_button.place(x=int(self.width*0.65), y=int(self.height*0.58))
            legend_bool.set(True)
            ttk.Label(self.frame_5b, text="Figure dimensions",
                                   ).place(x=int(self.width*0.65), y=int(self.height*0.66))
            ttk.Label(self.frame_5b, text="H",
                                   ).place(x=int(self.width*0.65), y=int(self.height*0.72))
            figH = ttk.Entry(self.frame_5b)
            figH.insert(0, "3")
            figH.place(x=int(self.width*0.67), y=int(self.height*0.72), relwidth=0.05)
            ttk.Label(self.frame_5b, text="W",
                                   ).place(x=int(self.width*0.8), y=int(self.height*0.72))
            figW = ttk.Entry(self.frame_5b)
            figW.insert(0, "6")
            figW.place(x=int(self.width*0.82), y=int(self.height*0.72), relwidth=0.05)
            ttk.Button(self.frame_5b, text="Plot embeddings",
                                    command=lambda: self.embed_plot(self.pickle_path_5b.get(), embedding_type.get(), color.get(), legend_bool.get(), self.try_float(figH.get()), self.try_float(figW.get())),
                                    ).place(x=int(self.width*0.8), y=int(self.height*0.9))
        elif selected=="distance":
            ttk.Label(self.frame_5b, text="Distance metric to use",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.6))
            dist_metric = ttk.Combobox(self.frame_5b, values=["euclidean","cityblock","correlation"])
            dist_metric.place(x=int(self.width*0.45), y=int(self.height*0.6))
            dist_metric.set("euclidean")
            ttk.Label(self.frame_5b, text="Pairwise or centroid",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.7))
            pairwise_centroid_opt = ttk.Combobox(self.frame_5b, values=["pairwise", "centroid"])
            pairwise_centroid_opt.place(x=int(self.width*0.45), y=int(self.height*0.7))
            pairwise_centroid_opt.set("centroid")
            ttk.Label(self.frame_5b, text="Plot type",
                                   ).place(x=int(self.width*0.7), y=int(self.height*0.6))
            plot_type = ttk.Combobox(self.frame_5b, values=["boxplot", "heatmap"])
            plot_type.place(x=int(self.width*0.8), y=int(self.height*0.6))
            plot_type.set("heatmap")
            ttk.Button(self.frame_5b, text="Plot distance",
                                    command=lambda: self.distance_plot(self.pickle_path_5b.get(), dist_metric.get(),
                                                                       pairwise_centroid_opt.get(),
                                                                       plot_type.get()),
                                    ).place(x=int(self.width*0.8), y=int(self.height*0.9))


    def window5c_classify_regress_menu(self,pickle_path_prefill="/path/to/file.pickle"):
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        self.create_sidebar_widget()
        self.create_header("Classify and Embed Menu",header_path="Viz & Analyze ▶ Classify/Regress")

        if self.config["data_type"]=="Pose segmentation":
            pickle_text = "Path to module feature object (transitions or usage) .pickle:"
        elif self.config["data_type"]=="Pose estimation":
            pickle_text = "Path to keypoint travel object .pickle:"

        ttk.Label(self, text=pickle_text,
                               ).place(x=int(self.width*0.3), y=int(self.height*0.25))
        self.pickle_path_5c = ttk.Entry(self)
        self.pickle_path_5c.insert(0, pickle_path_prefill)
        self.pickle_path_5c.place(x=int(self.width*0.65), y=int(self.height*0.25))
        ttk.Button(self,
                                text="Browse",
                                command=lambda: self.window_browse(self.pickle_path_5c, type="file")).place(x=int(self.width*0.85), y=int(self.height*0.25))
        ttk.Label(self, text="Would you like to predict independent variable associated with subgroups using classification or regression?",
                               ).place(x=int(self.width*0.3), y=int(self.height*0.35))
        ttk.Radiobutton(self, text="Classify",
                        command=lambda: self.window5c_classify_regress_menu_update(),
                        variable=self.option_5c, value="classify",
                        ).place(x=int(self.width * 0.3), y=int(self.height * 0.4))
        ttk.Radiobutton(self, text="Regress",
                        command=lambda: self.window5c_classify_regress_menu_update(),
                        variable=self.option_5c, value="regress",
                        ).place(x=int(self.width * 0.55), y=int(self.height * 0.4))

        # Frame that will contain the dynamic content
        self.frame_5c = ttk.Frame(self)
        self.frame_5c.place(x=0, y=0, relwidth=1, relheight=1)
        self.frame_5c.lower()

        if self.config["data_type"]=="Pose segmentation":
            destination = self.window3b_PS_menu
        elif self.config["data_type"]=="Pose estimation":
            destination = self.window3a_PE_menu
        ttk.Button(self, text="◀ back to analysis menu", command=destination).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,).place(x=int(self.width*0.3), y=int(self.height*0.9))

        # Initial content
        self.window5c_classify_regress_menu_update()

    def window5c_classify_regress_menu_update(self):

        # Clear existing content in the frame
        for widget in self.frame_5c.winfo_children():
            widget.destroy()

        # Load content based on the selected option
        selected = self.option_5c.get()

        if selected=="classify":
            ttk.Label(self.frame_5c, text="Classification method to use",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.6))
            method = ttk.Combobox(self.frame_5c, values=["LogisticRegression","LDA","MLP","NaiveBayes","KNN","RandomForest"])
            method.place(x=int(self.width*0.5), y=int(self.height*0.6))
            method.set("LogisticRegression")
            ttk.Button(self.frame_5c, text="Leave-one-out-cross-validation (LOOCV)",
                                    command=lambda: self.classify(self.pickle_path_5c.get(), method.get(), "loocv"),
                                    ).place(x=int(self.width*0.65), y=int(self.height*0.8))
            ttk.Button(self.frame_5c, text="Fit and save classifier",
                                    command=lambda: self.classify(self.pickle_path_5c.get(), method.get(), "fullfit"),
                                    ).place(x=int(self.width*0.65), y=int(self.height*0.9))
        elif selected=="regress":
            try:
                module_feature_object_path = self.pickle_path_5c.get()
                module_feature_object = analyze.load_module_feature_object(module_feature_object_path)
                group_dict = module_feature_object.group_dict
                groups = list(group_dict.keys())
            except Exception:
                module_feature_object_path = self.pickle_path_5c.get()
                self.error_window(f"Issue loading module feature object - is the pickle path valid? {module_feature_object_path}")
            ttk.Label(self.frame_5c, text="Regression method to use",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.48))
            method = ttk.Combobox(self.frame_5c, values=["LinearRegression","Ridge","Lasso"])
            method.place(x=int(self.width*0.3), y=int(self.height*0.53))
            method.set("LinearRegression")
            ttk.Label(self.frame_5c, text="Alpha (for Ridge or Lasso only)",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.61))
            alpha = ttk.Entry(self.frame_5c)
            alpha.place(x=int(self.width*0.3), y=int(self.height*0.66))

            ttk.Label(self.frame_5c, text="What numeric value (dose, stim) \nis associated with each class?",
                                   ).place(x=int(self.width*0.65), y=int(self.height*0.48))
            scrollbox = ScrollableInput(self.frame_5c,groups)
            scrollbox.place(x=int(self.width*0.65), y=int(self.height*0.53), relwidth=0.25, relheight=0.2)

            ttk.Button(self.frame_5c, text="Leave-one-out-cross-validation (LOOCV)",
                                    command=lambda: self.regress(self.pickle_path_5c.get(), method.get(), scrollbox.get_values(), self.try_float(alpha.get()), "loocv"),
                                    ).place(x=int(self.width*0.65), y=int(self.height*0.8))
            ttk.Button(self.frame_5c, text="Fit and save regression model",
                                    command=lambda: self.regress(self.pickle_path_5c.get(), method.get(), scrollbox.get_values(), self.try_float(alpha.get()), "fullfit"),
                                    ).place(x=int(self.width*0.65), y=int(self.height*0.9))

    def window6a_simulate_menu(self):
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        self.create_sidebar_widget()
        self.create_header("Simulate Pose Data",header_path="Model & Simulate ▶ Simulate Data")

        self.option_6a = tk.StringVar(value="")

        ttk.Radiobutton(self, text="Simulate pose module \nsequence", variable=self.option_6a, value="Simulate sequence",
                        command=self.window6a_simulate_menu_update).place(x=int(self.width*0.3), y=int(self.height*0.25))
        ttk.Radiobutton(self, text="Simulate data with no \nsubgroup label", variable=self.option_6a, value="Simulate usage with no labels",
                        command=self.window6a_simulate_menu_update).place(x=int(self.width*0.55), y=int(self.height*0.25))
        ttk.Radiobutton(self, text="Simulate data with \nsubgroup label", variable=self.option_6a, value="Simulate usage with labels",
                        command=self.window6a_simulate_menu_update).place(x=int(self.width*0.8), y=int(self.height*0.25))

        # Frame that will contain the dynamic content
        self.frame_6a = ttk.Frame(self)
        self.frame_6a.place(x=0, y=0, relwidth=1, relheight=1)
        self.frame_6a.lower()

        # Initial content
        self.window6a_simulate_menu_update()

        # Bottom back buttons
        ttk.Button(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window5d_keypoint_displacement_menu(self):
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        self.create_sidebar_widget()
        self.create_header("Keypoint Displacment Menu",header_path="Viz & Analyze ▶ Keypoint Displacement")
        self.option_5d = tk.StringVar(value="")

        ttk.Radiobutton(self, text="Get keypoint travel",
                        command=lambda: self.window5d_keypoint_displacement_menu_update(),
                        variable=self.option_5d, value="Get displacement",
                        ).place(x=int(self.width*0.3), y=int(self.height*0.2))
        ttk.Radiobutton(self, text="Plot keypoint travel",
                        command=lambda: self.window5d_keypoint_displacement_menu_update(),
                        variable=self.option_5d, value="Plot displacement",
                        ).place(x=int(self.width*0.65), y=int(self.height*0.2))

        # Frame that will contain the dynamic content
        self.frame_5d = ttk.Frame(self)
        self.frame_5d.place(x=0, y=0, relwidth=1, relheight=1)
        self.frame_5d.lower()

        # Initial content
        self.window5d_keypoint_displacement_menu_update()

        ttk.Button(self, text="◀ back to analysis menu", command=self.window3a_PE_menu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))


    def window5d_keypoint_displacement_menu_update(self,pickle_path_prefill = "/path/to/usage.pickle",
                                               tx_pickle_path_prefill="/path/to/transitions.pickle",
                                               plot_option=None):

        # Clear existing content in the frame
        for widget in self.frame_5d.winfo_children():
            widget.destroy()

        # Load content based on the selected option
        selected = self.option_5d.get()

        if selected=="Get displacement":
            ttk.Label(self.frame_5d, text="Enter info about your analysis.",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.3))

            ttk.Label(self.frame_5d, text="Start time (seconds)",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.38))
            start = ttk.Entry(self.frame_5d)
            start.place(x=int(self.width*0.3), y=int(self.height*0.43))
            ttk.Label(self.frame_5d, text="End time (seconds)",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.51))
            end = ttk.Entry(self.frame_5d)
            end.place(x=int(self.width*0.3), y=int(self.height*0.56))
            ttk.Label(self.frame_5d, text="Binsize (seconds; blank for no binning)",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.64))
            binsize = ttk.Entry(self.frame_5d)
            binsize.place(x=int(self.width*0.3), y=int(self.height*0.69))
            ttk.Label(self.frame_5d, text="Data to include",
                                   ).place(x=int(self.width*0.65), y=int(self.height*0.38))
            dropdown = MultiDropDown(self.frame_5d,options=["all combined"]+list(self.config["subgroups"].keys()),
                                     x=int(self.width*0.65), y=int(self.height*0.43))
            dropdown.place(x=int(self.width*0.65), y=int(self.height*0.43))

            save_option = tk.StringVar(value="scatter")
            save_options = ["pickle (for further use in MARIPoSA)",
                             "csv (for external use)",
                             "pickle AND csv"]
            style_vars = ["pickle", "csv", "both"]
            ttk.Label(self.frame_5d, text="How should the data be saved?",
                                   ).place(x=int(self.width*0.65), y=int(self.height*0.6))
            for s in range(len(style_vars)):
                radio_btn = ttk.Radiobutton(self.frame_5d, text=save_options[s],
                                                         variable=save_option, value=style_vars[s],
                                                         )
                radio_btn.place(x=int(self.width*0.65), y=int(self.height*(0.67+0.05*s)))

            keypoint="tailbase"
            thresh=70

            ttk.Button(self.frame_5d, text="Analyze & Save",
                                    command=lambda: self.save_keypoint_travel(keypoint,
                                                                              self.try_int(start.get()),
                                                                              self.try_int(end.get()),
                                                                              dropdown.get_selected_options(),
                                                                              binsize.get(),
                                                                              self.try_int(thresh),
                                                                              save_to=save_option.get()),
                                    ).place(x=int(self.width*0.8), y=int(self.height*0.9))

        elif selected=="Plot displacement":
            ttk.Label(self.frame_5d, text="Path to keypoint displacement .pickle:",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.42))
            pickle_path = ttk.Entry(self.frame_5d)
            pickle_path.insert(0, pickle_path_prefill)
            pickle_path.place(x=int(self.width*0.6), y=int(self.height*0.42))
            ttk.Button(self.frame_5d,
                                    text="Browse",
                                    command=lambda: self.window_browse(pickle_path, type="file")).place(x=int(self.width*0.8), y=int(self.height*0.42))
            ttk.Label(self.frame_5d, text="Colormap",
                                   ).place(x=int(self.width*0.65), y=int(self.height*0.5))
            color = ttk.Combobox(self.frame_5d, values=["jet", "cividis", "viridis", "magma"])
            color.place(x=int(self.width*0.75), y=int(self.height*0.5))
            color.set("jet")
            # Choose style for plot
            ttk.Label(self.frame_5d, text="What should the usage plot style be?",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.5))
            style = tk.StringVar(value="scatter")
            style_options = ["Bar (for no timebins)",
                             "Trace with error bar",
                             "Trace with error band"]
            style_vars = ["bar","errorbar","band"]
            for s in range(len(style_vars)):
                radio_btn = ttk.Radiobutton(self.frame_5d, text=style_options[s],
                                                         variable=style, value=style_vars[s],
                                                         )
                radio_btn.place(x=int(self.width*0.3), y=int(self.height*(0.56+0.06*s)))
            ttk.Button(self.frame_5d, text="Plot it!",
                       command=lambda: self.plot_keypoint_travel(pickle_path.get(), color.get(), style.get()),
                       ).place(x=int(self.width*0.8), y=int(self.height*0.9))


        ttk.Button(self, text="◀ back to analysis menu", command=self.window3a_PE_menu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window5e_action_units_menu(self):
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        self.create_sidebar_widget()
        self.create_header("Action Units Menu",header_path="Viz & Analyze ▶ Action Units")
        self.option_5e = tk.StringVar(value="")

        ttk.Radiobutton(self, text="Get action units",
                        command=lambda: self.window5e_action_units_menu_update(),
                        variable=self.option_5e, value="Get AU",
                        ).place(x=int(self.width*0.3), y=int(self.height*0.2))
        ttk.Radiobutton(self, text="Plot action units",
                        command=lambda: self.window5e_action_units_menu_update(),
                        variable=self.option_5e, value="Plot AU",
                        ).place(x=int(self.width*0.65), y=int(self.height*0.2))

        # Frame that will contain the dynamic content
        self.frame_5e = ttk.Frame(self)
        self.frame_5e.place(x=0, y=0, relwidth=1, relheight=1)
        self.frame_5e.lower()

        # Initial content
        self.window5e_action_units_menu_update()

        ttk.Button(self, text="◀ back to analysis menu", command=self.window3a_PE_menu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))


    def window5e_action_units_menu_update(self,pickle_path_prefill = "/path/to/usage.pickle",
                                               tx_pickle_path_prefill="/path/to/transitions.pickle",
                                               plot_option=None):

        # Clear existing content in the frame
        for widget in self.frame_5e.winfo_children():
            widget.destroy()

        # Load content based on the selected option
        selected = self.option_5e.get()

        if selected=="Get AU":
            ttk.Label(self.frame_5e, text="Enter info about your analysis.",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.3))

            ttk.Label(self.frame_5e, text="Start time (seconds)",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.38))
            start = ttk.Entry(self.frame_5e)
            start.place(x=int(self.width*0.3), y=int(self.height*0.43))
            ttk.Label(self.frame_5e, text="End time (seconds)",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.51))
            end = ttk.Entry(self.frame_5e)
            end.place(x=int(self.width*0.3), y=int(self.height*0.56))
            ttk.Label(self.frame_5e, text="Binsize (seconds; blank for no binning)",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.64))
            binsize = ttk.Entry(self.frame_5e)
            binsize.place(x=int(self.width*0.3), y=int(self.height*0.69))
            ttk.Label(self.frame_5e, text="Data to include",
                                   ).place(x=int(self.width*0.65), y=int(self.height*0.38))
            dropdown = MultiDropDown(self.frame_5e,options=["all combined"]+list(self.config["subgroups"].keys()),
                                     x=int(self.width*0.65), y=int(self.height*0.43))
            dropdown.place(x=int(self.width*0.65), y=int(self.height*0.43))

            save_option = tk.StringVar(value="scatter")
            save_options = ["pickle (for further use in MARIPoSA)",
                             "csv (for external use)",
                             "pickle AND csv"]
            style_vars = ["pickle", "csv", "both"]
            ttk.Label(self.frame_5e, text="How should the data be saved?",
                                   ).place(x=int(self.width*0.65), y=int(self.height*0.6))
            for s in range(len(style_vars)):
                radio_btn = ttk.Radiobutton(self.frame_5e, text=save_options[s],
                                                         variable=save_option, value=style_vars[s],
                                                         )
                radio_btn.place(x=int(self.width*0.65), y=int(self.height*(0.67+0.05*s)))

            ttk.Button(self.frame_5e, text="Analyze & Save",
                                    command=lambda: self.save_action_units(self.try_int(start.get()),
                                                                              self.try_int(end.get()),
                                                                              dropdown.get_selected_options(),
                                                                              binsize.get(),
                                                                              save_to=save_option.get()),
                                    ).place(x=int(self.width*0.8), y=int(self.height*0.9))

        elif selected=="Plot AU":
            ttk.Label(self.frame_5e, text="Path to keypoint displacement .pickle:",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.42))
            pickle_path = ttk.Entry(self.frame_5e)
            pickle_path.insert(0, pickle_path_prefill)
            pickle_path.place(x=int(self.width*0.6), y=int(self.height*0.42))
            ttk.Button(self.frame_5e,
                                    text="Browse",
                                    command=lambda: self.window_browse(pickle_path, type="file")).place(x=int(self.width*0.8), y=int(self.height*0.42))
            ttk.Label(self.frame_5e, text="Colormap",
                                   ).place(x=int(self.width*0.65), y=int(self.height*0.5))
            color = ttk.Combobox(self.frame_5e, values=["jet", "cividis", "viridis", "magma"])
            color.place(x=int(self.width*0.75), y=int(self.height*0.5))
            color.set("jet")
            # Choose style for plot
            ttk.Label(self.frame_5e, text="What should the usage plot style be?",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.5))
            style = tk.StringVar(value="scatter")
            style_options = ["Bar (for no timebins)",
                             "Trace with error bar",
                             "Trace with error band"]
            style_vars = ["bar","errorbar","band"]
            for s in range(len(style_vars)):
                radio_btn = ttk.Radiobutton(self.frame_5e, text=style_options[s],
                                                         variable=style, value=style_vars[s],
                                                         )
                radio_btn.place(x=int(self.width*0.3), y=int(self.height*(0.5+0.06*s)))
            ttk.Button(self.frame_5e, text="Plot it!",
                       command=lambda: self.plot_action_units(pickle_path.get(), color.get(), style.get()),
                       ).place(x=int(self.width*0.8), y=int(self.height*0.9))


        ttk.Button(self, text="◀ back to analysis menu", command=self.window3a_PE_menu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window6a_simulate_menu_update(self,pickle_path_prefill="/path/to/file.pickle",labels_path_prefill="/path/to/file.csv"):
        # Clear existing content in the frame
        for widget in self.frame_6a.winfo_children():
            widget.destroy()

        # Load content based on the selected option
        selected = self.option_6a.get()
        if selected == "Simulate sequence":

            ttk.Label(self.frame_6a, text="Path to labels .csv:",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.4))
            labels_df_path = ttk.Entry(self.frame_6a)
            labels_df_path.insert(0, labels_path_prefill)
            labels_df_path.place(x=int(self.width*0.6), y=int(self.height*0.4))
            ttk.Button(self.frame_6a,
                                    text="Browse",
                                    command=lambda: self.window_browse(labels_df_path, type="file")).place(x=int(self.width*0.85), y=int(self.height*0.4))

            ttk.Label(self.frame_6a, text="Number of samples to generate:").place(x=int(self.width*0.3), y=int(self.height*0.5))
            n_samples = ttk.Entry(self.frame_6a)
            n_samples.insert(0, "10")
            n_samples.place(x=int(self.width*0.3), y=int(self.height*0.55))

            ttk.Label(self.frame_6a, text="Length of simulated sequences (in seconds):").place(x=int(self.width*0.3), y=int(self.height*0.63))
            T = ttk.Entry(self.frame_6a)
            T.insert(0, "")
            T.place(x=int(self.width*0.3), y=int(self.height*0.68))

            ttk.Label(self.frame_6a, text="Random state:").place(x=int(self.width*0.7), y=int(self.height*0.5))
            rs = ttk.Entry(self.frame_6a)
            rs.insert(0, "42")
            rs.place(x=int(self.width*0.7), y=int(self.height*0.55))

            ttk.Button(self.frame_6a, text="Simulate!",
                                    command=lambda: self.simulate_sequence(labels_df_path.get(),
                                                                           self.try_int(T.get()),
                                                                           self.try_int(n_samples.get()),
                                                                           self.try_int(rs.get())),
                                    ).place(x=int(self.width*0.8), y=int(self.height*0.9))

        elif selected == "Simulate usage with no labels":

            ttk.Label(self.frame_6a, text="Path to module usage .pickle:",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.4))
            pickle_path = ttk.Entry(self.frame_6a)
            pickle_path.insert(0, pickle_path_prefill)
            pickle_path.place(x=int(self.width*0.6), y=int(self.height*0.4))
            ttk.Button(self.frame_6a,
                                    text="Browse",
                                    command=lambda: self.window_browse(pickle_path, type="file")).place(x=int(self.width*0.85), y=int(self.height*0.4))

            ttk.Label(self.frame_6a, text="Simulation mode:",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.5))
            ttk.Radiobutton(self.frame_6a, text="multivariate normal (recommended)", variable=self.option_simmode,
                            value="multivariate_normal").place(x=int(self.width * 0.3),y=int(self.height * 0.55))
            ttk.Radiobutton(self.frame_6a, text="log normal", variable=self.option_simmode,
                            value="log-normal").place(x=int(self.width * 0.3), y=int(self.height * 0.6))

            ttk.Label(self.frame_6a, text="Number of samples to generate:").place(x=int(self.width*0.6), y=int(self.height*0.5))
            n_samples = ttk.Entry(self.frame_6a)
            n_samples.insert(0, "10")
            n_samples.place(x=int(self.width*0.8), y=int(self.height*0.5))

            ttk.Label(self.frame_6a, text="Random state:").place(x=int(self.width*0.6), y=int(self.height*0.6))
            rs = ttk.Entry(self.frame_6a)
            rs.insert(0, "42")
            rs.place(x=int(self.width*0.8), y=int(self.height*0.6))

            ttk.Button(self.frame_6a, text="Simulate!",
                                    command=lambda: self.simulate_usage_unlabeled(pickle_path.get(), self.option_simmode.get(), n_samples.get(), rs.get()),
                                    ).place(x=int(self.width*0.8), y=int(self.height*0.9))



        elif selected == "Simulate usage with labels":

            ttk.Label(self.frame_6a, text="Path to module usage .pickle:",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.4))
            pickle_path = ttk.Entry(self.frame_6a)
            pickle_path.insert(0, pickle_path_prefill)
            pickle_path.place(x=int(self.width*0.6), y=int(self.height*0.4))
            ttk.Button(self.frame_6a,
                                    text="Browse",
                                    command=lambda: self.window_browse(pickle_path, type="file")).place(x=int(self.width*0.85), y=int(self.height*0.4))

            ttk.Label(self.frame_6a, text="Path to regression model .pickle:",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.5))
            reg_path = ttk.Entry(self.frame_6a)
            reg_path.insert(0, pickle_path_prefill)
            reg_path.place(x=int(self.width*0.6), y=int(self.height*0.5))
            ttk.Button(self.frame_6a,
                                    text="Browse",
                                    command=lambda: self.window_browse(reg_path, type="file")).place(x=int(self.width*0.85), y=int(self.height*0.5))

            ttk.Label(self.frame_6a, text="Simulation mode:",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.6))
            ttk.Radiobutton(self.frame_6a, text="multivariate normal (recommended)", variable=self.option_simmode,
                            value="multivariate_normal").place(x=int(self.width * 0.3),y=int(self.height * 0.65))
            ttk.Radiobutton(self.frame_6a, text="log normal", variable=self.option_simmode,
                            value="log-normal").place(x=int(self.width * 0.3), y=int(self.height * 0.7))

            ttk.Label(self.frame_6a, text="Number of samples to generate:").place(x=int(self.width*0.6), y=int(self.height*0.6))
            n_samples = ttk.Entry(self.frame_6a)
            n_samples.insert(0, "10")
            n_samples.place(x=int(self.width*0.8), y=int(self.height*0.6))

            ttk.Label(self.frame_6a, text="Random state:").place(x=int(self.width*0.6), y=int(self.height*0.7))
            rs = ttk.Entry(self.frame_6a)
            rs.insert(0, "42")
            rs.place(x=int(self.width*0.8), y=int(self.height*0.7))

            ttk.Button(self.frame_6a, text="Simulate!",
                                    command=lambda: self.simulate_usage_labeled(pickle_path.get(), self.option_simmode.get(), n_samples.get(), rs.get()),
                                    ).place(x=int(self.width*0.8), y=int(self.height*0.9))

    def window6b_fit_curve_menu(self):
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        self.create_sidebar_widget()
        self.create_header("Fit a Curve",header_path="Model & Simulate ▶ Curve Fitting")

        # Bottom back buttons
        ttk.Button(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window6c_cdf_menu(self):
        self.clear_window()
        ttk.Button(self, text="Log", command=self.log_window,
                                ).place(x=int(self.width * 0.94), y=int(self.height * 0.02),relwidth=0.05)
        self.create_sidebar_widget()
        self.create_header("Fit Cumulative Distribution Function",
                           header_path="Model & Simulate ▶ Cumulative Distribution Function")

        # Bottom back buttons
        ttk.Button(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                ).place(x=int(self.width * 0.3), y=int(self.height * 0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width * 0.3), y=int(self.height * 0.9))
#

    def plot_pose_vs_BORIS(self):
        BORIS_to_pose_mat, BORIS_to_pose_mat_normalized, loss = analyze.BORIS_to_pose(self.config) # this line is messing up the loaded config
        fig = plot.BORIS_to_pose_matrix_plot(self.config, BORIS_to_pose_mat_normalized)
        self.plots_generated = self.plots_generated + 1
        self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
        self.plot_window.mainloop()

    def make_save_labels_df(self, start, end, subgroups):
        try:
            if subgroups == ["all combined"]:
                labels_df = analyze.get_module_labels(self.config, start, end)
                file_path = filedialog.asksaveasfilename(defaultextension='.csv',
                                                         filetypes=[("csvs", "*.csv"),
                                                                    ("All Files", "*.*")])
                labels_df.to_csv(file_path)
                self.append_log(f"labels_df = analyze.get_module_labels(config, {start}, {end}")
                self.append_log(f"labels_df.to_csv('{file_path}')")
            else:
                selected_subgroups=[i for i in subgroups if (i!="all combined")]
                labels_df = analyze.get_module_labels(self.config, start, end, subgroups=selected_subgroups)
                file_path = filedialog.asksaveasfilename(defaultextension='.csv',
                                                         filetypes=[("csvs", "*.csv"),
                                                                    ("All Files", "*.*")])
                labels_df.to_csv(file_path)
                self.append_log(f"labels_df = analyze.get_module_labels(config, {start}, {end}, subgroups={selected_subgroups})")
                self.append_log(f"labels_df.to_csv('{file_path}')")
        except Exception as e:
            self.error_window(e)

    def save_usage_transitions(self, start, end, subgroups, binsize, save_to):
        try:
            if len(binsize)>0:
                bin = True
                binsize = float(binsize)
            else:
                bin = False
            if subgroups == ["all combined"]:
                labels_df = analyze.get_module_labels(self.config, start, end)
                self.append_log(f"labels_df = analyze.get_module_labels(config, {start}, {end})")
                if bin:
                    module_usage = analyze.get_module_usage(self.config, labels_df, binsize=binsize)
                    self.append_log(f"module_usage = analyze.get_module_usage(config, labels_df, binsize={binsize})")
                else:
                    module_usage = analyze.get_module_usage(self.config, labels_df)
                module_transitions = analyze.get_module_transitions(self.config, labels_df)
                file_path = filedialog.asksaveasfilename(defaultextension='.pickle',
                                                         filetypes=[("pickle files", "*.pickle"),
                                                                    ("All Files", "*.*")])
                if save_to=="pickle":
                    module_usage.save(file_path+"_USAGE.pickle")
                    module_transitions.save(file_path+"_TRANSITIONS.pickle")
                    self.append_log(f"module_usage.save('{file_path}_USAGE.pickle')")
                    self.append_log(f"module_transitions.save('{file_path}_TRANSITIONS.pickle')")
                elif save_to=="csv":
                    usage_df = module_usage.to_df()
                    usage_df.to_csv(file_path+"_USAGE.csv")
                    transitions_df = module_transitions.to_df()
                    transitions_df.to_csv(file_path+"_TRANSITIONS.csv")
                    self.append_log(f"usage_df = module_usage.to_df()")
                    self.append_log(f"usage_df.to_csv('{file_path}_USAGE.csv')")
                    self.append_log(f"transitions_df = module_transitions.to_df()")
                    self.append_log(f"transitions_df.to_csv('{file_path}_TRANSITIONS.csv')")
                elif save_to=="both":
                    module_usage.save(file_path+"_USAGE.pickle")
                    module_transitions.save(file_path+"_TRANSITIONS.pickle")
                    self.append_log(f"module_usage.save('{file_path}_USAGE.pickle')")
                    self.append_log(f"module_transitions.save('{file_path}_TRANSITIONS.pickle')")
                    usage_df = module_usage.to_df()
                    usage_df.to_csv(file_path+"_USAGE.csv")
                    self.append_log(f"usage_df = module_usage.to_df()")
                    self.append_log(f"usage_df.to_csv('{file_path}_USAGE.csv')")
                    transitions_df = module_transitions.to_df()
                    transitions_df.to_csv(file_path+"_TRANSITIONS.csv")
                    self.append_log(f"transitions_df = module_transitions.to_df()")
                    self.append_log(f"transitions_df.to_csv('{file_path}_TRANSITIONS.csv')")


            else:
                selected_subgroups=[i for i in subgroups if (i!="all combined")]
                labels_df = analyze.get_module_labels(self.config, start, end, subgroups=selected_subgroups)
                self.append_log(f"labels_df = analyze.get_module_labels(config, {start}, {end}, subgroups={selected_subgroups})")
                if bin:
                    module_usage = analyze.get_module_usage(self.config, labels_df, binsize=binsize)
                    self.append_log(f"module_usage = analyze.get_module_usage(config, labels_df, binsize={binsize})")
                else:
                    module_usage = analyze.get_module_usage(self.config, labels_df)
                    self.append_log(f"module_usage = analyze.get_module_usage(config, labels_df)")
                module_transitions = analyze.get_module_transitions(self.config, labels_df)
                self.append_log(f"module_transitions = analyze.get_module_transitions(config, labels_df)")
                file_path = filedialog.asksaveasfilename(defaultextension='.pickle',
                                                         filetypes=[("pickle files", "*.pickle"),
                                                                    ("All Files", "*.*")])
                if save_to=="pickle":
                    module_usage.save(file_path+"_USAGE.pickle")
                    module_transitions.save(file_path+"_TRANSITIONS.pickle")
                    self.append_log(f"module_usage.save('{file_path}_USAGE.pickle')")
                    self.append_log(f"module_transitions.save('{file_path}_TRANSITIONS.pickle')")
                elif save_to=="csv":
                    usage_df = module_usage.to_df()
                    usage_df.to_csv(file_path+"_USAGE.csv")
                    self.append_log(f"usage_df = module_usage.to_df()")
                    self.append_log(f"usage_df.to_csv('{file_path}_USAGE.csv')")
                    transitions_df = module_transitions.to_df()
                    transitions_df.to_csv(file_path+"_TRANSITIONS.csv")
                    self.append_log(f"transitions_df = module_transitions.to_df()")
                    self.append_log(f"transitions_df.to_csv('{file_path}_TRANSITIONS.csv')")
                elif save_to=="both":
                    module_usage.save(file_path+"_USAGE.pickle")
                    module_transitions.save(file_path+"_TRANSITIONS.pickle")
                    self.append_log(f"module_usage.save('{file_path}_USAGE.pickle')")
                    self.append_log(f"module_transitions.save('{file_path}_TRANSITIONS.pickle')")
                    usage_df = module_usage.to_df()
                    usage_df.to_csv(file_path+"_USAGE.csv")
                    self.append_log(f"usage_df = module_usage.to_df()")
                    self.append_log(f"usage_df.to_csv('{file_path}_USAGE.csv')")
                    transitions_df = module_transitions.to_df()
                    transitions_df.to_csv(file_path+"_TRANSITIONS.csv")
                    self.append_log(f"transitions_df = module_transitions.to_df()")
                    self.append_log(f"transitions_df.to_csv('{file_path}_TRANSITIONS.csv')")
        except Exception as e:
            self.error_window(e)

    def save_keypoint_travel(self, keypoint, start, end, subgroups, binsize, thresh, save_to):
        if len(binsize)>0:
            bin = True
            binsize = int(binsize)
        else:
            bin = False
        if subgroups == ["all combined"]:
            if bin:
                keypoint_travel = analyze.get_keypoint_travel(self.config,keypoint,start,end,binsize=binsize,thresh=thresh,return_as_df=False)
                self.append_log(f"analyze.get_keypoint_travel(config,'{keypoint}',{start},{end},binsize={binsize},thresh={thresh},return_as_df=False)")
            else:
                keypoint_travel = analyze.get_keypoint_travel(self.config,keypoint,start,end,thresh=thresh,return_as_df=False)
                self.append_log(f"analyze.get_keypoint_travel(config,'{keypoint}',{start},{end},thresh={thresh},return_as_df=False)")
        else:
            selected_subgroups=[i for i in subgroups if (i!="all combined")]
            if bin:
                keypoint_travel = analyze.get_keypoint_travel(self.config,keypoint,start,end,binsize=binsize,selected_subgroups=selected_subgroups,thresh=thresh,return_as_df=False)
                self.append_log(f"analyze.get_keypoint_travel(config,'{keypoint}',{start},{end},binsize={binsize},selected_subgroups={selected_subgroups},thresh={thresh},return_as_df=False)")
            else:
                keypoint_travel = analyze.get_keypoint_travel(self.config,keypoint,start,end,selected_subgroups=selected_subgroups,thresh=thresh,return_as_df=False)
                self.append_log(f"analyze.get_keypoint_travel(config,'{keypoint}',{start},{end},selected_subgroups={selected_subgroups},thresh={thresh},return_as_df=False)")
            file_path = filedialog.asksaveasfilename(defaultextension='.pickle',
                                                     filetypes=[("pickle files", "*.pickle"),
                                                                ("All Files", "*.*")])
            if save_to=="pickle":
                keypoint_travel.save(file_path+"_KEYPOINT_TRAVEL.pickle")
                self.append_log(f"keypoint_travel.save('{file_path}_KEYPOINT_TRAVEL.pickle')")
            elif save_to=="csv":
                kf_df = keypoint_travel.to_df()
                kf_df.to_csv(file_path+"_KEYPOINT_TRAVEL.csv")
                self.append_log(f"kf_df = keypoint_travel.to_df()")
                self.append_log(f"kf_df.to_csv('{file_path}_KEYPOINT_TRAVEL.csv')")
            elif save_to=="both":
                keypoint_travel.save(file_path+"_KEYPOINT_TRAVEL.pickle")
                self.append_log(f"keypoint_travel.save('{file_path}_KEYPOINT_TRAVEL.pickle')")
                kf_df = keypoint_travel.to_df()
                kf_df.to_csv(file_path+"_KEYPOINT_TRAVEL.csv")
                self.append_log(f"kf_df = keypoint_travel.to_df()")
                self.append_log(f"kf_df.to_csv('{file_path}_KEYPOINT_TRAVEL.csv')")

    def save_action_units(self, start, end, subgroups, binsize, save_to):
        if len(binsize)>0:
            bin = True
            binsize = int(binsize)
        else:
            bin = False
        if subgroups == ["all combined"]:
            if bin:
                au = analyze.get_action_units(self.config, start, end, binsize=binsize)
                self.append_log(f"au = analyze.get_action_units(config, {start}, {end}, binsize={binsize})")
            else:
                au = analyze.get_action_units(self.config, start, end)
                self.append_log(f"au = analyze.get_action_units(config, {start}, {end})")
        else:
            selected_subgroups=[i for i in subgroups if (i!="all combined")]
            if bin:
                au = analyze.get_action_units(self.config, start, end, binsize=binsize,selected_subgroups=selected_subgroups)
                self.append_log(f"au = analyze.get_action_units(config, {start}, {end}, binsize={binsize},selected_subgroups={selected_subgroups})")
            else:
                au = analyze.get_action_units(self.config, start, end, selected_subgroups=selected_subgroups)
                self.append_log(f"au = analyze.get_action_units(config, {start}, {end}, selected_subgroups={selected_subgroups})")
            file_path = filedialog.asksaveasfilename(defaultextension='.pickle',
                                                     filetypes=[("pickle files", "*.pickle"),
                                                                ("All Files", "*.*")])
            if save_to=="pickle":
                au.save(file_path+"_ACTION_UNITS.pickle")
                self.append_log(f"au.save('{file_path}_ACTION_UNITS.pickle')")
            elif save_to=="csv":
                au_df = au.to_df()
                au_df.to_csv(file_path+"_ACTION_UNITS.csv")
                self.append_log(f"au_df = au.to_df()")
                self.append_log(f"au_df.to_csv('{file_path}_ACTION_UNITS.csv')")
            elif save_to=="both":
                au.save(file_path+"_ACTION_UNITS.pickle")
                self.append_log(f"au.save('{file_path}_ACTION_UNITS.pickle')")
                au_df = au.to_df()
                au_df.to_csv(file_path+"_ACTION_UNITS.csv")
                self.append_log(f"au_df = au.to_df()")
                self.append_log(f"au_df.to_csv('{file_path}_ACTION_UNITS.csv')")

    def plot_usage(self, pickle_path, style, color, legend_TF, figH, figW):
        try:
            module_usage = analyze.load_module_feature_object(pickle_path)
            fig = plot.plot_module_usage(self.config, module_usage, style=style, cmap=color, legend=legend_TF, figH=figH, figW=figW)
            self.plots_generated = self.plots_generated + 1
            self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
            self.plot_window.mainloop()
        except Exception as e:
            self.error_window(e)

    def plot_network(self, pickle_path, tx_pickle_path, color):
        try:
            module_usage = analyze.load_module_feature_object(pickle_path)
            module_transitions = analyze.load_module_feature_object(tx_pickle_path)
            fig = plot.network_plot(self.config, module_usage = module_usage, module_transitions = module_transitions, cmap=color)
            self.append_log(f"module_usage = analyze.load_module_feature_object('{pickle_path}')")
            self.append_log(f"module_transitions = analyze.load_module_feature_object('{tx_pickle_path}')")
            self.append_log(f"fig = plot.network_plot(config, module_usage = module_usage, module_transitions = module_transitions, cmap='{color}')")
            self.plots_generated = self.plots_generated + 1
            self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
            self.plot_window.mainloop()
        except Exception as e:
            self.error_window(e)

    def plot_sandplot(self, pickle_path):
        try:
            module_usage = analyze.load_module_feature_object(pickle_path)
            fig = plot.module_usage_sandplot(self.config, module_usage)
            self.append_log(f"module_usage = analyze.load_module_feature_object('{pickle_path}')")
            self.append_log(f"fig = plot.module_usage_sandplot(config, module_usage)")
            self.plots_generated = self.plots_generated + 1
            self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
            self.plot_window.mainloop()
        except Exception as e:
            self.error_window(e)

    def plot_keypoint_travel(self,kpf_pickle_path,color,plottype):
        try:
            keypoint_feature = analyze.load_module_feature_object(kpf_pickle_path)
            fig = plot.plot_keypoint_travel(keypoint_feature, cmap=color, plottype=plottype)
            self.append_log(f"keypoint_feature = analyze.load_module_feature_object('{kpf_pickle_path}')")
            self.append_log(f"fig = plot.plot_keypoint_travel(keypoint_feature, cmap='{color}', plottype='{plottype}')")
            self.plots_generated = self.plots_generated + 1
            self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
            self.plot_window.mainloop()
        except Exception as e:
            self.error_window(e)

    def embed_plot(self, pickle_path, embedding_type, color, legend_TF, figH, figW):
        try:
            module_feature_object = analyze.load_module_feature_object(pickle_path)
            emb = analyze.embed(module_feature_object, method=embedding_type, n_components=2)
            fig = plot.plot_embeddings(module_feature_object, emb, cmap=color, legend=legend_TF, figH=figH, figW=figW)
            self.append_log(f"module_feature_object = analyze.load_module_feature_object('{pickle_path}')")
            self.append_log(f"emb = analyze.embed(module_feature_object, method='{embedding_type}', n_components=2)")
            self.append_log(f"fig = plot.plot_embeddings(module_feature_object, emb, cmap='{color}')")
            self.plots_generated = self.plots_generated + 1
            self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
            self.plot_window.mainloop()
        except Exception as e:
            self.error_window(e)

    def distance_plot(self, pickle_path, dist_metric, pairwise_centroid_opt, plot_type):
        try:
            module_feature_object = analyze.load_module_feature_object(pickle_path)
            dist_mat = analyze.get_distance(module_feature_object, method=dist_metric)
            self.append_log(f"module_feature_object = analyze.load_module_feature_object('{pickle_path}')")
            self.append_log(f"dist_mat = analyze.get_distance(module_feature_object, method='{dist_metric}')")
            if plot_type == "heatmap":
                fig = plot.plot_distance_matrix(module_feature_object, dist_mat)
                self.append_log(f"fig = plot.plot_distance_matrix(module_feature_object, dist_mat)")
            elif plot_type == "boxplot":
                fig = plot.plot_distance_box(module_feature_object, dist_mat)
                self.append_log(f"fig = plot.plot_distance_box(module_feature_object, dist_mat)")
            self.plots_generated = self.plots_generated + 1
            self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
            self.plot_window.mainloop()
        except Exception as e:
            self.error_window(e)

    def classify(self, pickle_path, method, fullfit_or_loocv):
        try:
            if fullfit_or_loocv == "fullfit":
                module_feature_object = analyze.load_module_feature_object(pickle_path)
                clf = analyze.classify(module_feature_object, method=method)
                self.append_log(f"module_feature_object = analyze.load_module_feature_object('{pickle_path}')")
                self.append_log(f"clf = analyze.classify(module_feature_object, method='{method}')")
                save_path = filedialog.asksaveasfilename(defaultextension='.pickle',
                                                         filetypes=[("pickle files", "*.pickle"),
                                                                    ("All Files", "*.*")])
                analyze.pickle_dump(clf, save_path)
            elif fullfit_or_loocv == "loocv":
                module_feature_object = analyze.load_module_feature_object(pickle_path)
                accuracy, conf_mat = analyze.loocv(module_feature_object, method=method)
                self.append_log(f"module_feature_object = analyze.load_module_feature_object('{pickle_path}')")
                self.append_log(f"accuracy, conf_mat = analyze.loocv(module_feature_object, method='{method}')")
                print(f"Accuracy: {accuracy}")
                print(f"Confusion matrix: \n{conf_mat}")

        except Exception as e:
            self.error_window(e)

    def regress(self, pickle_path, method, dose_dict, alpha, fullfit_or_loocv):
        try:
            if fullfit_or_loocv == "fullfit":
                module_feature_object = analyze.load_module_feature_object(pickle_path)
                reg = analyze.regress(module_feature_object, dose_dict, method=method, alpha=alpha)
                save_path = filedialog.asksaveasfilename(defaultextension='.pickle',
                                                         filetypes=[("pickle files", "*.pickle"),
                                                                    ("All Files", "*.*")])
                analyze.pickle_dump(reg, save_path)
            elif fullfit_or_loocv == "loocv":
                module_feature_object = analyze.load_module_feature_object(pickle_path)
                y_preds, sq_err = analyze.loocv_regression(module_feature_object, dose_dict, method=method,alpha=alpha)
                self.append_log(f"module_feature_object = analyze.load_module_feature_object('{pickle_path}')")
                self.append_log(f"y_preds, sq_err = analyze.loocv_regression(module_feature_object, {dose_dict}, method='{method}',alpha={alpha})")
                print(f"Error: {np.sqrt(np.mean(sq_err))}")
                print(f"Held-out predictions (M±SD):")
                for handle, num in list(module_feature_object.group_dict.items()):
                    mask = np.array(module_feature_object.group_labels)==num
                    y_preds_h = y_preds[mask]
                    print(f"  {handle} ({dose_dict[handle]}): {np.mean(y_preds_h)} ± {np.std(y_preds_h)}")

        except Exception as e:
            self.error_window(e)

    def simulate_sequence(self, labels_df_path, T, n_samples, rs):
        labels_df = pd.read_csv(labels_df_path, index_col=0, header=[0, 1])
        labels_df.reset_index(drop=True, inplace=True)
        sim_labels_df = simulate.generate_sequence(self.config, labels_df, T, n_subjs=n_samples, random_state=rs)
        self.append_log(f"labels_df = pd.read_csv('{labels_df_path}', index_col=0, header=[0, 1])")
        self.append_log(f"labels_df.reset_index(drop=True, inplace=True)")
        self.append_log(f"sim_labels_df = simulate.generate_sequence(config, labels_df, {T}, n_subjs={n_samples}, random_state={rs})")
        save_path = filedialog.asksaveasfilename(defaultextension='.csv',
                                                 filetypes=[("csvs", "*.csv"),
                                                            ("All Files", "*.*")])
        sim_labels_df.to_csv(save_path)
        self.append_log(f"sim_labels_df.to_csv('{save_path}')")


    def simulate_usage_unlabeled(self, pickle_path, method, n_samples, rs):
        try:
            module_usage = analyze.load_module_feature_object(pickle_path)
            sim_module_usage = simulate.generate_usage(module_usage, int(n_samples), random_state=int(rs), mode=method)
            self.append_log(f"module_usage = analyze.load_module_feature_object('{pickle_path}')")
            self.append_log(f"sim_module_usage = simulate.generate_usage(module_usage, {int(n_samples)}, random_state={int(rs)}, mode='{method}')")
            save_path = filedialog.asksaveasfilename(defaultextension='.pickle',
                                                     filetypes=[("pickle files", "*.pickle"),
                                                                ("All Files", "*.*")])
            analyze.pickle_dump(sim_module_usage, save_path)
            self.append_log(f"analyze.pickle_dump(sim_module_usage, '{save_path}')")
        except Exception as e:
            self.error_window(e)

    def simulate_usage_labeled(self, pickle_path, reg_path, method, n_samples, rs):
        try:
            module_usage = analyze.load_module_feature_object(pickle_path)
            reg = analyze.load_module_feature_object(reg_path)
            bins = 0
            simulate.generate_usage_labeled(module_usage, int(n_samples), bins, reg, max_iters="default",
                                   random_state=int(rs), mode=method, scale=10, verbosity="medium")
            sim_module_usage = simulate.generate_usage(module_usage, int(n_samples), random_state=int(rs), mode=method)
            self.append_log(f"module_usage = analyze.load_module_feature_object('{pickle_path}')")
            self.append_log(f"reg = analyze.load_module_feature_object('{reg_path}')")
            self.append_log(f"bins = 0")
            self.append_log(f"simulate.generate_usage_labeled(module_usage, {int(n_samples)}, bins, reg, max_iters='default',random_state={int(rs)}, mode='{method}', scale=10, verbosity='medium')")
            save_path = filedialog.asksaveasfilename(defaultextension='.pickle',
                                                     filetypes=[("pickle files", "*.pickle"),
                                                                ("All Files", "*.*")])
            analyze.pickle_dump(sim_module_usage, save_path)
            self.append_log(f"analyze.pickle_dump(sim_module_usage, '{save_path}')")
        except Exception as e:
            self.error_window(e)

class MultiDropDown(ttk.Frame):
    def __init__(self, parent, options, x, y):
        super().__init__(parent)

        self.options = options
        self.choices = {}  # Keeps track of checkbox states
        self.menu_visible = False
        self.dropdown_win = None  # Will be created on first open

        self.menubutton = ttk.Button(self, text="Select option(s) ▼", command=self.toggle_menu, width=25)
        self.menubutton.pack(fill="x")

        # Store the position of the dropdown menu relative to the parent widget
        self.x = x
        self.y = y

        # Bind click event to close the menu when clicked outside
        self.parent = parent
        self.parent.bind("<Button-1>", self.close_menu_on_click_outside)

    def toggle_menu(self):
        """ Toggles the visibility of the dropdown menu. """
        if self.menu_visible:
            self.hide_menu()
        else:
            self.show_menu()

    def show_menu(self):
        """ Displays the dropdown menu. """
        if not self.dropdown_win:
            self.create_dropdown()
            self.dropdown_win.canvas = tk.Canvas(self.dropdown_win)

        button_x = self.menubutton.winfo_rootx()
        button_y = self.menubutton.winfo_rooty() + self.menubutton.winfo_height()

        self.dropdown_win.geometry(f"+{button_x}+{button_y}")

        self.dropdown_win.deiconify()
        self.dropdown_win.lift()
        self.dropdown_win.focus_force()

        self.menu_visible = True

    def hide_menu(self, event=None):
        """ Hides the dropdown menu. """
        if self.dropdown_win:
            self.dropdown_win.withdraw()
        self.menu_visible = False

    def create_dropdown(self):
        """ Creates the dropdown menu (Toplevel window with scrollable checkboxes). """
        self.dropdown_win = tk.Toplevel(self)
        self.dropdown_win.withdraw()
        self.dropdown_win.overrideredirect(True)  # Remove window borders
        self.dropdown_win.attributes("-topmost", True)  # Keep it on top

        # Outer frame inside Toplevel
        outer_frame = tk.Frame(self.dropdown_win)
        outer_frame.pack(fill="both", expand=True)

        # Canvas and scrollbar
        canvas = tk.Canvas(outer_frame, borderwidth=0, highlightthickness=0)
        scrollbar = tk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Internal frame inside canvas
        checkbox_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=checkbox_frame, anchor="nw")

        # Create checkboxes for each option
        for option in self.options:
            var = tk.BooleanVar(value=self.choices.get(option, False))  # Preserve state
            checkbox = ttk.Checkbutton(checkbox_frame, text=option, variable=var, style="TCheckbutton")
            checkbox.pack(anchor="w", fill="x", padx=5, pady=2)
            self.choices[option] = var  # Keep track of the variable

        # Update scroll region to fit all checkboxes
        checkbox_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        # Bind resizing event to update scroll region
        checkbox_frame.bind("<Configure>", self._update_scrollregion)
        self.dropdown_win.bind_all("<Button-1>", self._on_global_click, "+")

    def _on_global_click(self, event):
        # Check if the click is outside the dropdown
        widget = event.widget
        if not self.dropdown_win.winfo_containing(event.x_root, event.y_root):
            self.hide_menu()

    def _update_scrollregion(self, event=None):
        """ Updates the scroll region whenever the checkboxes change size. """
        self.dropdown_win.update_idletasks()  # Ensure the size is updated
        self.dropdown_win.canvas.configure(scrollregion=self.dropdown_win.canvas.bbox("all"))

    def close_menu_on_click_outside(self, event):
        """ Closes the menu if clicked outside the dropdown. """
        if self.menu_visible:
            # Check if the click happened outside the dropdown menu and button
            if not (self.menubutton.winfo_rootx() <= event.x <= self.menubutton.winfo_rootx() + self.menubutton.winfo_width() and
                    self.menubutton.winfo_rooty() <= event.y <= self.menubutton.winfo_rooty() + self.menubutton.winfo_height()):
                if not (self.dropdown_win and
                        self.dropdown_win.winfo_rootx() <= event.x <= self.dropdown_win.winfo_rootx() + self.dropdown_win.winfo_width() and
                        self.dropdown_win.winfo_rooty() <= event.y <= self.dropdown_win.winfo_rooty() + self.dropdown_win.winfo_height()):
                    self.hide_menu()

    def get_selected_options(self):
        """ Returns a list of selected options. """
        return [option for option, var in self.choices.items() if var.get()]

    def update_button_text(self):
        """ Updates the button text with selected options. """
        selected = self.get_selected_options()
        if selected:
            self.menubutton.config(text=f"Selected: {', '.join(selected)}")
        else:
            self.menubutton.config(text="Select option(s) ▼")


class ScrollableInput(tk.Frame):
    def __init__(self, parent, items):
        super().__init__(parent)

        self.items = items
        self.entries = {}

        # Create canvas + scrollbar to hold the frame with inputs
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Populate scrollable_frame with label + entry pairs
        for item in self.items:
            frame = ttk.Frame(self.scrollable_frame)
            label = ttk.Label(frame, text=item)
            entry = ttk.Entry(frame)
            label.pack(side="left", padx=(0, 10))
            entry.pack(side="left", fill="x", expand=True)
            frame.pack(fill="x", pady=2)

            self.entries[item] = entry

    def get_values(self):
        result = {}
        for item, entry in self.entries.items():
            val = entry.get()
            try:
                num = float(val)
                result[item] = num
            except ValueError:
                Application.error_window(f"Values in dictionary need to be numeric! Bad value: {val}")
        return result

class PlotWindow(tk.Tk):
    def __init__(self, fig, plot_number, master=None):
        super().__init__()
        self.title('MARIPoSA - Plot ' + str(plot_number))

        # Set window style
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        factor = 0.4
        self.width = int(screen_width * factor)
        self.height = int(screen_height * factor)
        x = (screen_width - self.width) // 2
        y = (screen_height - self.height) // 2
        self.geometry(f'{self.width}x{self.height}+{x}+{y}')
        self.fig = fig
        self.create_widgets_tkagg()

    def create_widgets_tkagg(self):
        # Create a canvas to embed the Matplotlib figure
        canvas = FigureCanvasTkAgg(self.fig, master=self)  # Use the received figure
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0)

        # Add a button to save the plot
        save_button = ttk.Button(self, text="Save Plot", command=lambda: self.save_plot(self.fig))
        save_button.grid(row=1, column=0)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def save_plot(self, fig):
        # Open a file dialog and save the figure to the specified path
        file_path = filedialog.asksaveasfilename(defaultextension='.png',
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("All Files", "*.*")])
        if file_path:
            fig.savefig(file_path, dpi=500)
#
if __name__ == "__main__":
    app = Application()
    #cProfile.run('app.mainloop()')
    app.mainloop()
