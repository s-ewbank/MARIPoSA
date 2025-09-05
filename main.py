import tkinter as tk
from tkinter import filedialog, PhotoImage, ttk
import customtkinter
from utils import metadata, analyze, plot
from datetime import datetime
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

        self.window1_start()

    def clear_window(self):
        """
        Clear window
        :return:
        """
        for widget in self.winfo_children():
            widget.destroy()

    def display_error(self, message, x,y):
        """
        Display error
        :param message:
        :param row:
        :param column:
        :param columnspan:
        :return:
        """
        print(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")+' Error: '+message)
        self.error_label = ttk.Label(self, text=message, text_color="red", bg_color="#ffe5e3",anchor=tk.CENTER)
        self.error_label.place(x=x,y=y,anchor=tk.CENTER)
        self.after(3000, lambda: self.error_label.configure(text="",bg_color="gray16"))

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
                error_message="That path does not exist. please enter an existing path."
                self.display_error(error_message,int(self.width*0.5),int(self.height*0.7))
        else:
            print(self.projectstart_choice)

    def window1b_projecttype(self):
        self.clear_window()
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
        self.clear_window()
        self.project_name = datetime.now().strftime('%y%m%d_') + project_name
        self.config_path = output_directory + "/" + self.project_name + "/config_PE.yaml"
        config = metadata.load_project(self.config_path)
        self.config = config
        self.window3a_PE_menu()

    def create_PS_project(self, project_name, data_directory, data_source, output_directory,fps,n_modules):
        metadata.create_PS_project(project_name, data_directory, data_source, output_directory,fps,n_modules)
        self.clear_window()
        self.project_name = datetime.now().strftime('%y%m%d_') + project_name
        self.config_path = output_directory + "/" + self.project_name + "/config_PS.yaml"
        config = metadata.load_project(self.config_path)
        self.config = config
        self.window3b_PS_menu()

    def load_project(self):
        self.clear_window()
        config = metadata.load_project(self.config_path)
        self.config = config
        if self.config["data_type"]=="Pose estimation":
            self.window3a_PE_menu()
        elif self.config["data_type"]=="Pose segmentation":
            self.window3b_PS_menu()

    def load_project_BORIS(self):
        self.clear_window()
        config = metadata.load_project(self.config_path)
        self.config = config
        self.window4b3_pose_vs_BORIS()

    def load_project_mainmenu(self):
        self.clear_window()
        config = metadata.load_project(self.config_path)
        self.config = config
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
        self.create_sidebar_widget()
        self.create_header("Analysis Menu")
        button_width=int(self.width*0.2)
        button_height=int(self.height*0.15)
        ttk.Label(self, text="Further configure project:",
                               ).place(x=int(self.width * 0.3), y=int(self.height * 0.2),width=button_width,height=button_height)
        # ttk.Button(self, text="Define subgroups \nwithin data",
        #                         command=self.window4a_define_subgroups, image=subgroups_img,
        #                         width=button_width,height=button_height).place(x=int(self.width*0.3), y=int(self.height*0.3))
        # ttk.Button(self, text="Compare modules to \nmanual scoring",
        #                         command=self.window4h_pose_vs_BORIS, image=remap_img,
        #                         width=button_width,height=button_height).place(x=int(self.width*0.52), y=int(self.height*0.3))
        # ttk.Button(self, text="Manually combine \npose modules",
        #                         command=self.window1_start, image=remap_img,
        #                         width=button_width,height=button_height).place(x=int(self.width*0.74), y=int(self.height*0.3))

        ttk.Label(self, text="Visualize and classify:",
                               width=button_width).place(x=int(self.width * 0.3), y=int(self.height * 0.5))
        # ttk.Button(self, text="Analyze pose module \nusage and transitions",
        #                         command=self.window4b_usage_transitions_menu, image=usage_img,
        #                         width=button_width,height=button_height).place(x=int(self.width*0.3), y=int(self.height*0.6))
        # ttk.Button(self, text="Embed and measure \ndistance between \ngroups",
        #                         command=self.window4e_embed_menu, image=embed_img,
        #                         width=button_width,height=button_height).place(x=int(self.width*0.52), y=int(self.height*0.6))
        # ttk.Button(self, text="Classify conditions",
        #                         command=self.window4g_classify_menu, image=classify_img,
        #                         width=button_width,height=button_height).place(x=int(self.width*0.74), y=int(self.height*0.6))

        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window3b_PS_menu(self):
        self.clear_window()
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
        ttk.Button(self, text="Measure pose module usage and transitions",
                                command=self.window5a_usage_transitions_menu,
                                ).place(x=int(self.width*0.5), y=int(self.height*0.44),width=button_width,height=button_height)
        ttk.Button(self, text="Embed and/or measure distance between groups",
                                command=self.window5b_embed_distance_menu,
                                ).place(x=int(self.width*0.5), y=int(self.height*0.51),width=button_width,height=button_height)
        ttk.Button(self, text="Classify and/or regress conditions",
                                command=self.window5c_classify_regress_menu,
                                ).place(x=int(self.width*0.5), y=int(self.height*0.58),width=button_width,height=button_height)

        ttk.Label(self, text="Model and simulate:",
                               ).place(x=int(self.width * 0.3), y=int(self.height * 0.68),width=menu_item_width,height=button_height)
        ttk.Button(self, text="Fit curve to within-session pose data",
                                command=self.window6a_fit_curve_menu,
                                ).place(x=int(self.width*0.5), y=int(self.height*0.68),width=button_width,height=button_height)
        ttk.Button(self, text="Simulate module labels, usage, or transitions",
                                command=self.window6b_simulate_menu,
                                ).place(x=int(self.width*0.5), y=int(self.height*0.75),width=button_width,height=button_height)
        ttk.Button(self, text="Get cumulative distribution function from real or simulated behavior",
                                command=self.window6c_cdf_menu,
                                ).place(x=int(self.width*0.5), y=int(self.height*0.82),width=button_width,height=button_height)

        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window4a_define_subgroups(self):
        self.clear_window()
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
        ttk.Button(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window4b_pose_vs_BORIS(self):
        self.clear_window()
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
        ttk.Button(self, text="◀ update config and go back to BORIS menu", command=self.load_project_BORIS,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.76))
        ttk.Button(self, text="◀ update config and go back to analysis menu", command=self.load_project_mainmenu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window4b3_pose_vs_BORIS(self):
        self.clear_window()
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
    def window5a_usage_transitions_menu(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Pose Usage and Transition Analysis Menu",header_path="Viz & Analyze ▶ Usage & Transitions")

        ttk.Label(self, text="What kind of plot would you like to do?",
                               ).place(x=int(self.width*0.65), y=int(self.height*0.3),anchor=tk.CENTER)

        ttk.Button(self, text="1. Get pose usage and transitions",
                                command=lambda: self.window5a1_get_usage_transitions(),
                                ).place(x=int(self.width*0.65), y=int(self.height*0.35),height=int(self.height*0.2),width=int(self.width*0.6),anchor=tk.CENTER)
        ttk.Button(self, text="2. Plot pose usage and/or transitions",
                                command=lambda: self.window5a2_plot_usage_transitions(),
                                ).place(x=int(self.width*0.65), y=int(self.height*0.58),height=int(self.height*0.2),width=int(self.width*0.6),anchor=tk.CENTER)

        ttk.Button(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))


    def window5a1_get_usage_transitions(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Measure Pose Usage and Transitions",header_path="Viz & Analyze ▶ Usage & Transitions ▶ Measure")

        ttk.Label(self, text="Enter info about your analysis.",
                               ).place(x=int(self.width*0.3), y=int(self.height*0.25))

        ttk.Label(self, text="Start time (seconds)",
                               ).place(x=int(self.width*0.3), y=int(self.height*0.35))
        start = ttk.Entry(self)
        start.place(x=int(self.width*0.45), y=int(self.height*0.35))
        ttk.Label(self, text="End time (seconds)",
                               ).place(x=int(self.width*0.3), y=int(self.height*0.45))
        end = ttk.Entry(self)
        end.place(x=int(self.width*0.45), y=int(self.height*0.45))
        ttk.Label(self, text="Data to include",
                               ).place(x=int(self.width*0.3), y=int(self.height*0.55))
        dropdown = MultiDropDown(self,options=["all combined"]+list(self.config["subgroups"].keys()))
        dropdown.place(x=int(self.width*0.45), y=int(self.height*0.55))
        ttk.Label(self, text="Binsize (seconds; blank for no binning)",
                               ).place(x=int(self.width*0.65), y=int(self.height*0.35))
        binsize = ttk.Entry(self)
        binsize.place(x=int(self.width*0.65), y=int(self.height*0.45))
        save_option = tk.StringVar(value="scatter")
        save_options = ["Save to pickle (for further using in MARIPoSA)",
                         "Save to csv (for external use)",
                         "Save to pickle AND csv"]
        style_vars = ["pickle", "csv", "both"]
        for s in range(len(style_vars)):
            radio_btn = ttk.Radiobutton(self, text=save_options[s],
                                                     variable=save_option, value=style_vars[s],
                                                     )
            radio_btn.place(x=int(self.width*0.65), y=int(self.height*(0.55+0.08*s)))
        ttk.Button(self, text="Analyze & Save",
                                command=lambda: self.save_usage_transitions(int(start.get()),
                                                                            int(end.get()),
                                                                            dropdown.get_selected_values(),
                                                                            binsize.get(),
                                                                            save_to=save_option.get()),
                                ).place(x=int(self.width*0.8), y=int(self.height*0.8))
        ttk.Button(self, text="Proceed to plotting",
                                command=lambda: self.window5a2_plot_usage_transitions(),
                                ).place(x=int(self.width*0.8), y=int(self.height*0.9))
        # Bottom back buttons
        ttk.Button(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window5a2_plot_usage_transitions(self,pickle_path_prefill = "/path/to/usage.pickle",
                                         tx_pickle_path_prefill="/path/to/transitions.pickle",
                                         plot_option=None):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Get & Plot Pose Usage",
                           header_path="Viz & Analyze ▶ Usage & Transitions ▶ Plot")

        ttk.Label(self, text="Path to module usage .pickle:",
                               ).place(x=int(self.width*0.3), y=int(self.height*0.25))
        pickle_path = ttk.Entry(self)
        pickle_path.insert(0, pickle_path_prefill)
        pickle_path.place(x=int(self.width*0.6), y=int(self.height*0.25))
        ttk.Button(self,
                                text="Browse",
                                command=lambda: self.window_browse(pickle_path, type="file")).place(x=int(self.width*0.8), y=int(self.height*0.25))
        ttk.Label(self, text="Path to module transitions .pickle:",
                               ).place(x=int(self.width*0.3), y=int(self.height*0.32))
        tx_pickle_path = ttk.Entry(self)
        tx_pickle_path.insert(0, tx_pickle_path_prefill)
        tx_pickle_path.place(x=int(self.width*0.6), y=int(self.height*0.32))
        ttk.Button(self,
                                text="Browse",
                                command=lambda: self.window_browse(tx_pickle_path, type="file")).place(x=int(self.width*0.8), y=int(self.height*0.32))
        if plot_option==None:
            ttk.Label(self, text="What type of plot do you want to make?",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.4))
            ttk.Button(self, text="Usage comparison",
                                    command=lambda: self.window5a2_plot_usage_transitions(pickle_path_prefill=pickle_path.get(),
                                                                                          tx_pickle_path_prefill=tx_pickle_path.get(),
                                                                                          plot_option="usage"),
                                    ).place(x=int(self.width*0.3), y=int(self.height*0.45))
            ttk.Button(self, text="Pairwise network\ncomparison",
                                    command=lambda: self.window5a2_plot_usage_transitions(pickle_path_prefill=pickle_path.get(),
                                                                                          tx_pickle_path_prefill=tx_pickle_path.get(),
                                                                                          plot_option="network"),
                                    ).place(x=int(self.width*0.55), y=int(self.height*0.45))
            ttk.Button(self, text="Within-session\ntime dynamic usage",
                                    command=lambda: self.window5a2_plot_usage_transitions(pickle_path_prefill=pickle_path.get(),
                                                                                          tx_pickle_path_prefill=tx_pickle_path.get(),
                                                                                          plot_option="sandplot"),
                                    ).place(x=int(self.width*0.8), y=int(self.height*0.45))
        elif plot_option=="usage":
            ttk.Label(self, text="Selected plot type: USAGE",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.4))
            ttk.Label(self, text="Colormap",
                                   ).place(x=int(self.width*0.65), y=int(self.height*0.5))
            color = tk.ComboBox(self, values=["jet", "cividis", "viridis", "magma"])
            color.place(x=int(self.width*0.75), y=int(self.height*0.5))
            color.set("jet")
            # Choose style for plot
            ttk.Label(self, text="What should the usage plot style be?",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.5))
            style = tk.StringVar(value="scatter")
            style_options = ["Bar with scattered individual points",
                             "Bar with standard error of the mean",
                             "Points with standard error of the mean",
                             "Stacked means"]
            style_vars = ["bar_scatter", "bar_error", "points", "stacked"]
            for s in range(len(style_vars)):
                radio_btn = ttk.Radiobutton(self, text=style_options[s],
                                                         variable=style, value=style_vars[s],
                                                         )
                radio_btn.place(x=int(self.width*0.3), y=int(self.height*(0.5+0.06*s)))
            ttk.Button(self, text="Plot it!",
                                    command=lambda: self.plot_usage(pickle_path.get(), style.get(), color.get()),
                                    ).place(x=int(self.width*0.8), y=int(self.height*0.9))

        elif plot_option == "network":
            ttk.Label(self, text="Selected plot type: NETWORK",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.4))
            ttk.Label(self, text="Colormap",
                                   ).place(x=int(self.width*0.65), y=int(self.height*0.5))
            color = tk.Combobox(self, values=["bwr", "seismic", "PiYG", "BrBG", "PRGn"])
            color.place(x=int(self.width*0.75), y=int(self.height*0.5))
            color.set("bwr")
            ttk.Button(self, text="Plot it!",
                                    command=lambda: self.plot_network(pickle_path.get(), tx_pickle_path.get(), color.get()),
                                    ).place(x=int(self.width*0.8), y=int(self.height*0.9))
        elif plot_option=="sandplot":
            ttk.Label(self, text="Selected plot type: SANDPLOT",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.4))
            ttk.Button(self, text="Plot it!",
                                    command=lambda: self.plot_sandplot(pickle_path.get()),
                                    ).place(x=int(self.width*0.8), y=int(self.height*0.9))
        # Bottom back buttons
        ttk.Button(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window5b_embed_distance_menu(self,pickle_path_prefill = "/path/to/file.pickle",
                                         emb_dist_option=None):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Embed and Distance Menu",header_path="Viz & Analyze ▶ Embed")

        ttk.Label(self, text="Path to module feature object (transitions or usage) .pickle:",
                               ).place(x=int(self.width*0.3), y=int(self.height*0.25))
        pickle_path = ttk.Entry(self)
        pickle_path.insert(0, pickle_path_prefill)
        pickle_path.place(x=int(self.width*0.7), y=int(self.height*0.25))
        ttk.Button(self,
                                text="Browse",
                                command=lambda: self.window_browse(pickle_path, type="file")).place(x=int(self.width*0.85), y=int(self.height*0.25))

        if emb_dist_option is None:
            ttk.Label(self, text="Embed or measure distance between groups?",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.4))
            ttk.Button(self, text="Embed",
                                    command=lambda: self.window5b_embed_distance_menu(
                                        pickle_path_prefill=pickle_path.get(),
                                        emb_dist_option="embed"),
                                    ).place(x=int(self.width * 0.3), y=int(self.height * 0.45))
            ttk.Button(self, text="Distance",
                                    command=lambda: self.window5b_embed_distance_menu(
                                        pickle_path_prefill=pickle_path.get(),
                                        emb_dist_option="distance"),
                                    ).place(x=int(self.width * 0.55), y=int(self.height * 0.45))
        elif emb_dist_option=="embed":
            ttk.Label(self, text="Embed or measure distance option: EMBED",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.4))
            ttk.Label(self, text="Dimensionality reduction method",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.6))
            embedding_type = ttk.Combobox(self, values=["pca", "lda"])
            embedding_type.place(x=int(self.width*0.65), y=int(self.height*0.6))
            embedding_type.set("pca")
            ttk.Label(self, text="Colormap",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.7))
            color = ttk.Combobox(self, values=["jet", "cividis", "viridis", "magma"])
            color.place(x=int(self.width*0.65), y=int(self.height*0.7))
            color.set("jet")
            ttk.Button(self, text="Plot embeddings",
                                    command=lambda: self.embed_plot(pickle_path.get(), embedding_type.get(), color.get()),
                                    ).place(x=int(self.width*0.8), y=int(self.height*0.9))
        elif emb_dist_option=="distance":
            ttk.Label(self, text="Embed or measure distance option: DISTANCE",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.4))
            ttk.Label(self, text="Distance metric to use",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.6))
            dist_metric = ttk.Combobox(self, values=["euclidean","cityblock","correlation"])
            dist_metric.place(x=int(self.width*0.5), y=int(self.height*0.6))
            dist_metric.set("euclidean")
            ttk.Label(self, text="Pairwise or centroid",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.7))
            pairwise_centroid_opt = ttk.Combobox(self, values=["pairwise", "centroid"])
            pairwise_centroid_opt.place(x=int(self.width*0.5), y=int(self.height*0.7))
            pairwise_centroid_opt.set("centroid")
            ttk.Label(self, text="Plot type",
                                   ).place(x=int(self.width*0.65), y=int(self.height*0.6))
            plot_type = ttk.Combobox(self, values=["boxplot", "heatmap"])
            plot_type.place(x=int(self.width*0.8), y=int(self.height*0.6))
            plot_type.set("heatmap")
            ttk.Button(self, text="Plot distance",
                                    command=lambda: self.distance_plot(pickle_path.get(), dist_metric.get(),
                                                                       pairwise_centroid_opt.get(),
                                                                       plot_type.get()),
                                    ).place(x=int(self.width*0.8), y=int(self.height*0.9))
        # Bottom back buttons
        ttk.Button(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))


    def window5c_classify_regress_menu(self,pickle_path_prefill="/path/to/file.pickle",classify_regress_opt=None):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Classify and Embed Menu",header_path="Viz & Analyze ▶ Classify/Regress")

        ttk.Label(self, text="Path to module feature object (transitions or usage) .pickle:",
                               ).place(x=int(self.width*0.3), y=int(self.height*0.25))
        pickle_path = ttk.Entry(self)
        pickle_path.insert(0, pickle_path_prefill)
        pickle_path.place(x=int(self.width*0.7), y=int(self.height*0.25))
        ttk.Button(self,
                                text="Browse",
                                command=lambda: self.window_browse(pickle_path, type="file")).place(x=int(self.width*0.85), y=int(self.height*0.25))
        if classify_regress_opt is None:
            ttk.Label(self, text="Would you like to predict independent variable associated with subgroups using classification or regression?",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.4))
            ttk.Button(self, text="Classify",
                                    command=lambda: self.window5c_classify_regress_menu(
                                        pickle_path_prefill=pickle_path.get(),
                                        classify_regress_opt="classify"),
                                    ).place(x=int(self.width * 0.3), y=int(self.height * 0.45))
            ttk.Button(self, text="Regress",
                                    command=lambda: self.window5c_classify_regress_menu(
                                        pickle_path_prefill=pickle_path.get(),
                                        classify_regress_opt="regress"),
                                    ).place(x=int(self.width * 0.55), y=int(self.height * 0.45))
        elif classify_regress_opt=="classify":
            ttk.Label(self, text="Classification method to use",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.6))
            method = tk.ComboBox(self, values=["LogisticRegression","LDA","MLP","NaiveBayes","KNN","RandomForest"])
            method.place(x=int(self.width*0.5), y=int(self.height*0.6))
            method.set("LogisticRegression")
            ttk.Label(self, text="Classify or regress option: CLASSIFY",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.4))
            ttk.Button(self, text="Leave-one-out-cross-validation (LOOCV)",
                                    command=lambda: self.classify(pickle_path.get(), method.get(), "loocv"),
                                    ).place(x=int(self.width*0.8), y=int(self.height*0.8))
            ttk.Button(self, text="Fit and save classifier",
                                    command=lambda: self.classify(pickle_path.get(), method.get(), "fullfit"),
                                    ).place(x=int(self.width*0.8), y=int(self.height*0.9))
        elif classify_regress_opt=="regress":
            ttk.Label(self, text="Classify or regress option: REGRESS",
                                   ).place(x=int(self.width*0.3), y=int(self.height*0.4))
        # Bottom back buttons
        ttk.Button(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window6a_fit_curve_menu(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Fit a Curve",header_path="Model & Simulate ▶ Curve Fitting")

        # Bottom back buttons
        ttk.Button(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window6b_simulate_menu(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Simulate Pose Data",header_path="Model & Simulate ▶ Simulate Data")

        # Bottom back buttons
        ttk.Button(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.83))
        ttk.Button(self, text="◀◀ back to start", command=self.window1_start,
                                ).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window6c_cdf_menu(self):
        self.clear_window()
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

    def save_usage_transitions(self, start, end, subgroups, binsize, save_to):
        if len(binsize)>0:
            bin = True
            binsize = float(binsize)
        else:
            bin = False
        if subgroups == ["all combined"]:
            labels_df = analyze.get_module_labels(self.config, start, end)
            if bin:
                module_usage = analyze.get_module_usage(self.config, labels_df, binsize=binsize)
            else:
                module_usage = analyze.get_module_usage(self.config, labels_df)
            module_transitions = analyze.get_module_transitions(self.config, labels_df)
            file_path = filedialog.asksaveasfilename(defaultextension='.pickle',
                                                     filetypes=[("pickle files", "*.pickle"),
                                                                ("All Files", "*.*")])
            if save_to=="pickle":
                module_usage.save(file_path+"_USAGE.pickle")
                module_transitions.save(file_path+"_TRANSITIONS.pickle")
            elif save_to=="csv":
                usage_df = module_usage.to_df()
                usage_df.to_csv(file_path+"_USAGE.csv")
                transitions_df = module_transitions.to_df()
                transitions_df.to_csv(file_path+"_TRANSITIONS.csv")
            elif save_to=="both":
                module_usage.save(file_path+"_USAGE.pickle")
                module_transitions.save(file_path+"_TRANSITIONS.pickle")
                usage_df = module_usage.to_df()
                usage_df.to_csv(file_path+"_USAGE.csv")
                transitions_df = module_transitions.to_df()
                transitions_df.to_csv(file_path+"_TRANSITIONS.csv")


        else:
            selected_subgroups=[i for i in subgroups if (i!="all combined")]
            labels_df = analyze.get_module_labels(self.config, start, end, subgroups=selected_subgroups)
            if bin:
                module_usage = analyze.get_module_usage(self.config, labels_df, binsize=binsize)
            else:
                module_usage = analyze.get_module_usage(self.config, labels_df)
            module_transitions = analyze.get_module_transitions(self.config, labels_df)
            file_path = filedialog.asksaveasfilename(defaultextension='.pickle',
                                                     filetypes=[("pickle files", "*.pickle"),
                                                                ("All Files", "*.*")])
            if save_to=="pickle":
                module_usage.save(file_path+"_USAGE.pickle")
                module_transitions.save(file_path+"_TRANSITIONS.pickle")
            elif save_to=="csv":
                usage_df = module_usage.to_df()
                usage_df.to_csv(file_path+"_USAGE.csv")
                transitions_df = module_transitions.to_df()
                transitions_df.to_csv(file_path+"_TRANSITIONS.csv")
            elif save_to=="both":
                module_usage.save(file_path+"_USAGE.pickle")
                module_transitions.save(file_path+"_TRANSITIONS.pickle")
                usage_df = module_usage.to_df()
                usage_df.to_csv(file_path+"_USAGE.csv")
                transitions_df = module_transitions.to_df()
                transitions_df.to_csv(file_path+"_TRANSITIONS.csv")

    def plot_usage(self, pickle_path, style, color):
        module_usage = analyze.load_module_feature_object(pickle_path)
        fig = plot.plot_module_usage(self.config, module_usage, style=style, cmap=color)
        self.plots_generated = self.plots_generated + 1
        self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
        self.plot_window.mainloop()

    def plot_network(self, pickle_path, tx_pickle_path, color):
        module_usage = analyze.load_module_feature_object(pickle_path)
        module_transitions = analyze.load_module_feature_object(tx_pickle_path)
        fig = plot.network_plot(self.config, module_usage = module_usage, module_transitions = module_transitions, cmap=color)
        self.plots_generated = self.plots_generated + 1
        self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
        self.plot_window.mainloop()

    def plot_sandplot(self, pickle_path):
        module_usage = analyze.load_module_feature_object(pickle_path)
        fig = plot.module_usage_sandplot(self.config, module_usage)
        self.plots_generated = self.plots_generated + 1
        self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
        self.plot_window.mainloop()

    def embed_plot(self, pickle_path, embedding_type, color):
        module_feature_object = analyze.load_module_feature_object(pickle_path)
        emb = analyze.embed(module_feature_object, method=embedding_type, n_components=2)
        fig = plot.plot_embeddings(module_feature_object, emb, cmap=color)
        self.plots_generated = self.plots_generated + 1
        self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
        self.plot_window.mainloop()

    def distance_plot(self, pickle_path, dist_metric, pairwise_centroid_opt, plot_type):
        module_feature_object = analyze.load_module_feature_object(pickle_path)
        dist_mat = analyze.get_distance(module_feature_object, method=dist_metric)
        if plot_type == "heatmap":
            fig = plot.plot_distance_matrix(module_feature_object, dist_mat)
        elif plot_type == "boxplot":
            fig = plot.plot_distance_box(module_feature_object, dist_mat)
        self.plots_generated = self.plots_generated + 1
        self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
        self.plot_window.mainloop()

    def classify(self, pickle_path, method, fullfit_or_loocv):
        if fullfit_or_loocv == "fullfit":
            module_feature_object = analyze.load_module_feature_object(pickle_path)
            clf = analyze.classify(module_feature_object, method=method)
            save_path = filedialog.asksaveasfilename(defaultextension='.pickle',
                                                     filetypes=[("pickle files", "*.pickle"),
                                                                ("All Files", "*.*")])
            analyze.pickle_dump(clf, save_path)
        elif fullfit_or_loocv == "loocv":
            module_feature_object = analyze.load_module_feature_object(pickle_path)
            accuracy, conf_mat = analyze.loocv(module_feature_object, method=method)
            print(f"Accuracy: {accuracy}")
            print(f"Confusion matrix: \n{conf_mat}")

    def regress(self, pickle_path, method, fullfit_or_loocv):
        if fullfit_or_loocv == "fullfit":
            module_feature_object = analyze.load_module_feature_object(pickle_path)
            clf = analyze.classify(module_feature_object, method=method)
            save_path = filedialog.asksaveasfilename(defaultextension='.pickle',
                                                     filetypes=[("pickle files", "*.pickle"),
                                                                ("All Files", "*.*")])
            analyze.pickle_dump(clf, save_path)
        elif fullfit_or_loocv == "loocv":
            module_feature_object = analyze.load_module_feature_object(pickle_path)
            accuracy, conf_mat = analyze.loocv(module_feature_object, method=method)
            print(f"Accuracy: {accuracy}")
            print(f"Confusion matrix: \n{conf_mat}")

class MultiDropDown(tk.Frame):
    def __init__(self, parent, options):
        super().__init__(parent)

        self.options = options
        self.choices = {}
        self.menu_visible = False

        self.menubutton = ttk.Button(self, text="Select option(s) ▼", command=self.toggle_menu, width=25)
        self.menubutton.pack(fill="x")

        self.dropdown_win = None  # Will be created on first open

    def toggle_menu(self):
        if self.menu_visible:
            self.hide_menu()
        else:
            self.show_menu()

    def show_menu(self):
        if self.dropdown_win is None:
            self.create_dropdown()

        # Position the dropdown below the button
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.menubutton.winfo_height()
        self.dropdown_win.geometry(f"{self.menubutton.winfo_width()}x200+{x}+{y}")
        self.dropdown_win.deiconify()
        self.dropdown_win.lift()
        self.dropdown_win.focus_force()
        self.menu_visible = True

    def hide_menu(self, event=None):
        if self.dropdown_win:
            self.dropdown_win.withdraw()
        self.menu_visible = False

    def create_dropdown(self):
        self.dropdown_win = tk.Toplevel(self)
        self.dropdown_win.withdraw()
        self.dropdown_win.overrideredirect(True)
        self.dropdown_win.attributes("-topmost", True)
        self.dropdown_win.configure(bg="gray16")

        # Outer frame inside Toplevel
        outer_frame = tk.Frame(self.dropdown_win, bg="gray16")
        outer_frame.pack(fill="both", expand=True)

        # Canvas and scrollbar
        self.canvas = tk.Canvas(outer_frame, bg="gray16", borderwidth=0, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(outer_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Create internal frame inside canvas
        self.inner_frame = tk.Frame(self.canvas, bg="gray16")
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        # Add checkboxes
        for option in self.options:
            var = tk.IntVar()
            self.choices[option] = var
            cb = tk.Checkbutton(self.inner_frame, text=option, variable=var,
                                bg="gray16", fg="white", selectcolor="gray30", anchor="w")
            cb.pack(fill="x", padx=5, pady=2)

        # Scroll region and resizing
        self.inner_frame.bind("<Configure>", self._update_scrollregion)

        # Optional: hide when clicking outside
        self.dropdown_win.bind("<FocusOut>", self.hide_menu)

    def _update_scrollregion(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def get_selected_values(self):
        return [key for key, var in self.choices.items() if var.get() == 1]

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
