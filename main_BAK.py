import tkinter as tk
from tkinter import filedialog, PhotoImage
import customtkinter
from utils import metadata, analyze, plot
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from PIL import Image
import cProfile

factor=0.6

class Application(customtkinter.CTk):
    """
    Main application
    """
    def __init__(self):
        super().__init__()
        self.title("MARIPoSA")
        self.after(200, lambda: self.iconphoto(False, PhotoImage(file="other/MARIPoSA_icon.png")))
        #customtkinter.set_appearance_mode("dark")
        #customtkinter.set_default_color_theme("blue")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        factor = 0.6
        self.width = int(screen_width * factor)
        self.height = int(screen_height * factor)
        x = (screen_width - self.width) // 2
        y = (screen_height - self.height) // 2
        self.geometry(f'{self.width}x{self.height}+{x}+{y}')
        self.projectstart_choice = customtkinter.StringVar(value="New project")
        self.datatype = customtkinter.StringVar(value="B-SOiD")
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
        self.error_label = customtkinter.CTkLabel(self, text=message, text_color="red", bg_color="#ffe5e3",anchor=tk.CENTER)
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
            item_path_entry.delete(0, customtkinter.END)
            item_path_entry.insert(0, filename)
        elif type == "directory":
            dirname = filedialog.askdirectory()
            item_path_entry.delete(0, customtkinter.END)
            item_path_entry.insert(0, dirname)

    def window1_start(self):
        self.clear_window()

        logo_img = customtkinter.CTkImage(dark_image=Image.open('other/MARIPoSA_icon.png'), size=(130, 130))
        self.projectstart_choice = customtkinter.StringVar(value="New project")
        customtkinter.CTkLabel(self, text="MARIPoSA",
                               font=('Helvetica', 32, "bold")).place(x=int(self.width*0.5), y=int(self.height*0.1),anchor=tk.CENTER)
        customtkinter.CTkLabel(self, text="",image=logo_img).place(x=int(self.width*0.5), y=int(self.height*0.25),anchor=tk.CENTER)
        customtkinter.CTkLabel(self, text="Would you like to start a new project or load a previous project?",
                               font=('Helvetica', 16)).place(x=int(self.width*0.5), y=int(self.height*0.4),anchor=tk.CENTER)

        # Radio buttons for starting new project or loading old project
        projectstart_options = ["New project","Load previous"]
        grid_last = 0
        for option in projectstart_options:
            customtkinter.CTkRadioButton(self, text=option, variable=self.projectstart_choice, value=option,
                                         font=('Helvetica', 16)).place(x=int(self.width*0.2), y=int(self.height*(0.5+0.1*grid_last)))
            grid_last = grid_last + 1

        # File path entry and Browse button
        config_path_entry = customtkinter.CTkEntry(self)
        config_path_entry.insert(0, "/path/to/config.yaml")
        config_path_entry.place(x=int(self.width*0.4), y=int(self.height*0.6))
        customtkinter.CTkButton(self,
                                text="Browse",
                                command=lambda: self.window_browse(config_path_entry, type="file")).place(x=int(self.width*0.6), y=int(self.height*0.6))

        customtkinter.CTkButton(self, text="▶", command=lambda: self.make_or_load_project(config_path_entry.get()),
                                font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.9))

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
        self.projectstart_choice = customtkinter.StringVar(value="New project")
        customtkinter.CTkLabel(self, text="Create a new project",
                               font=('Helvetica', 32, "bold")).place(x=int(self.width*0.5), y=int(self.height*0.1),anchor=tk.CENTER)
        customtkinter.CTkLabel(self, text="Will the data type for this new dataset be pose estimation or pose segmentation?",
                               font=('Helvetica', 16)).place(x=int(self.width*0.5), y=int(self.height*0.3),anchor=tk.CENTER)

        customtkinter.CTkButton(self, text="Pose estimation\n\n(DeepLabCut, SLEAP, or OpenFace)",
                                command=lambda: self.window2_makeproject("pose_estimation"),height=int(self.height*0.25),width=int(self.width*0.35),
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.6),anchor=tk.CENTER)
        customtkinter.CTkButton(self, text="Pose segmentation\n\n(B-SOiD, VAME, or Keypoint-MoSeq)",
                                command=lambda: self.window2_makeproject("pose_segmentation"),height=int(self.height*0.25),width=int(self.width*0.35),
                                font=('Helvetica', 16)).place(x=int(self.width*0.7), y=int(self.height*0.6),anchor=tk.CENTER)

        customtkinter.CTkButton(self, text="◀", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width * 0.1), y=int(self.height * 0.9))

    def window2_makeproject(self,type):
        if type=="pose_estimation":
            self.clear_window()

            customtkinter.CTkLabel(self, text="Create a new pose estimation project",
                                   font=('Helvetica', 32, "bold")).place(x=int(self.width*0.5), y=int(self.height*0.08),anchor=tk.CENTER)

            customtkinter.CTkLabel(self, text="Project name",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.2), y=int(self.height*0.2))
            customtkinter.CTkEntry(self).place(x=int(self.width*0.5), y=int(self.height*0.2))

            # Enter data directory
            customtkinter.CTkLabel(self, text="Path to data directory",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.2), y=int(self.height*0.3))
            data_path_entry = customtkinter.CTkEntry(self)
            data_path_entry.place(x=int(self.width*0.5), y=int(self.height*0.3))
            browse_button = customtkinter.CTkButton(self, text="Browse",
                                                    command=lambda: self.window_browse(data_path_entry, type="directory"))
            browse_button.place(x=int(self.width*0.7), y=int(self.height*0.3))

            #Info about data
            customtkinter.CTkLabel(self, text="Source of data for new project",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.2), y=int(self.height*0.4))

            # Radio buttons for starting new project or loading old project
            datatype_options = ["DeepLabCut", "SLEAP", "OpenFace"]
            for r, option in enumerate(datatype_options):
                radio_btn = customtkinter.CTkRadioButton(self, text=option, variable=self.datatype, value=option,
                                                         font=('Helvetica', 16))
                radio_btn.place(x=int(self.width*0.5), y=int(self.height*(0.4+r*0.1)))

            customtkinter.CTkLabel(self, text="Frames per second",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.2), y=int(self.height*0.7))
            fps = customtkinter.CTkEntry(self)
            fps.place(x=int(self.width*0.5), y=int(self.height*0.7))

            # Enter project directory
            customtkinter.CTkLabel(self, text="Destination path for MARIPoSA output",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.2), y=int(self.height*0.8))
            project_path_entry = customtkinter.CTkEntry(self)
            project_path_entry.place(x=int(self.width*0.5), y=int(self.height*0.8))
            browse_button = customtkinter.CTkButton(self, text="Browse",
                                                    command=lambda: self.window_browse(project_path_entry,
                                                                                       type="directory"))
            browse_button.place(x=int(self.width*0.7), y=int(self.height*0.8))

            # Back to the initial view
            customtkinter.CTkButton(self, text="◀", command=self.window1_start,
                                    font=('Helvetica', 16)).place(x=int(self.width*0.1), y=int(self.height*0.9))
            customtkinter.CTkButton(self, text="▶",
                                    command=lambda: self.create_PE_project(
                                        project_name.get(), data_path_entry.get(), self.datatype.get(),
                                        project_path_entry.get(),fps.get()),
                                    font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.9))
        if type == "pose_segmentation":
            self.clear_window()

            customtkinter.CTkLabel(self, text="Create a new pose segmentation project",
                                   font=('Helvetica', 32, "bold")).place(x=int(self.width * 0.5),
                                                                         y=int(self.height * 0.08), anchor=tk.CENTER)

            customtkinter.CTkLabel(self, text="Project name",
                                   font=('Helvetica', 16)).place(x=int(self.width * 0.2), y=int(self.height * 0.2))
            project_name = customtkinter.CTkEntry(self)
            project_name.place(x=int(self.width * 0.5), y=int(self.height * 0.2))

            # Enter data directory
            customtkinter.CTkLabel(self, text="Path to data directory",
                                   font=('Helvetica', 16)).place(x=int(self.width * 0.2), y=int(self.height * 0.28))
            data_path_entry = customtkinter.CTkEntry(self)
            data_path_entry.place(x=int(self.width * 0.5), y=int(self.height * 0.28))
            browse_button = customtkinter.CTkButton(self, text="Browse",
                                                    command=lambda: self.window_browse(data_path_entry,
                                                                                       type="directory"))
            browse_button.place(x=int(self.width * 0.7), y=int(self.height * 0.28))

            # Info about data
            customtkinter.CTkLabel(self, text="Source of data for new project",
                                   font=('Helvetica', 16)).place(x=int(self.width * 0.2), y=int(self.height * 0.36))

            # Radio buttons for starting new project or loading old project
            datatype_options = ["B-SOiD", "VAME", "Keypoint-MoSeq"]
            for r, option in enumerate(datatype_options):
                customtkinter.CTkRadioButton(self, text=option, variable=self.datatype, value=option,
                                                         font=('Helvetica', 16)).place(x=int(self.width * 0.5), y=int(self.height * (0.36 + r * 0.08)))

            customtkinter.CTkLabel(self, text="Frames per second",
                                   font=('Helvetica', 16)).place(x=int(self.width * 0.2), y=int(self.height * 0.6))
            fps = customtkinter.CTkEntry(self)
            fps.place(x=int(self.width * 0.5), y=int(self.height * 0.6))

            customtkinter.CTkLabel(self, text="Number of modules",
                                   font=('Helvetica', 16)).place(x=int(self.width * 0.2), y=int(self.height * 0.68))
            n_modules = customtkinter.CTkEntry(self)
            n_modules.place(x=int(self.width * 0.5), y=int(self.height * 0.68))

            # Enter project directory
            customtkinter.CTkLabel(self, text="Destination path for MARIPoSA output",
                                   font=('Helvetica', 16)).place(x=int(self.width * 0.2), y=int(self.height * 0.75))
            project_path_entry = customtkinter.CTkEntry(self)
            project_path_entry.place(x=int(self.width * 0.5), y=int(self.height * 0.75))
            browse_button = customtkinter.CTkButton(self, text="Browse",
                                                    command=lambda: self.window_browse(project_path_entry,
                                                                                       type="directory"))
            browse_button.place(x=int(self.width * 0.7), y=int(self.height * 0.75))

            # Back to the initial view
            customtkinter.CTkButton(self, text="◀", command=self.window1_start,
                                    font=('Helvetica', 16)).place(x=int(self.width * 0.1), y=int(self.height * 0.9))
            customtkinter.CTkButton(self, text="▶",
                                    command=lambda: self.create_PS_project(
                                        project_name.get(), data_path_entry.get(), self.datatype.get(),
                                        project_path_entry.get(), fps.get(), n_modules.get()),
                                    font=('Helvetica', 16)).place(x=int(self.width * 0.8), y=int(self.height * 0.9))

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

    def create_sidebar_widget(self):

        bar_width=self.width*0.25
        frame = customtkinter.CTkFrame(self, width=bar_width, height=self.height*0.95)
        frame.place(x=int(self.width * 0.01), y=int(self.height * 0.02))

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=0)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=0)

        # Canvas
        canvas = tk.Canvas(frame, width=bar_width,height=self.height*0.95)
        canvas.grid(row=0, column=0, sticky="nsew")

        # Vertical scrollbar
        v_scrollbar = customtkinter.CTkScrollbar(frame, height=self.height*0.95, orientation=tk.VERTICAL, command=canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=v_scrollbar.set)

        # Horizontal scrollbar
        h_scrollbar = customtkinter.CTkScrollbar(frame, orientation=tk.HORIZONTAL, command=canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        canvas.configure(xscrollcommand=h_scrollbar.set)

        list_frame = customtkinter.CTkFrame(canvas)
        canvas.create_window((0, 0), window=list_frame, anchor=tk.NW)

        customtkinter.CTkLabel(list_frame, text="Config path", text_color="#3a7ebf", font=('Helvetica', 16, "bold"),anchor="w",width=bar_width).pack(side=tk.TOP, fill=tk.X,padx=5)
        customtkinter.CTkLabel(list_frame, text=self.config["project_directory"], font=('Helvetica', 16),anchor="w",width=bar_width).pack(side=tk.TOP, fill=tk.X,padx=5)
        customtkinter.CTkLabel(list_frame, text="Data source", text_color="#3a7ebf", font=('Helvetica', 16, "bold"),anchor="w",width=bar_width).pack(side=tk.TOP, fill=tk.X,padx=5)
        customtkinter.CTkLabel(list_frame, text=self.config["data_source"], font=('Helvetica', 16),anchor="w",width=bar_width).pack(side=tk.TOP, fill=tk.X,padx=5)
        customtkinter.CTkLabel(list_frame, text="Subgroups (" + str(len(self.config["subgroups"].keys())) + ")",
                               text_color="#3a7ebf", font=('Helvetica', 16, "bold"),anchor="w",width=bar_width).pack(side=tk.TOP, fill=tk.X,padx=5)
        for key in list(self.config["subgroups"].keys()):
            label = customtkinter.CTkLabel(list_frame, text=str(key+" ("+str(len(self.config["subgroups"][key]))+")"),anchor="w",width=bar_width)
            label.pack(anchor=tk.W,padx=5)
        customtkinter.CTkLabel(list_frame, text="Project Files (" + str(len(self.config["project_files"])) + ")",
                               text_color="#3a7ebf", font=('Helvetica', 16, "bold"),anchor="w",width=bar_width).pack(side=tk.TOP, fill=tk.X,padx=5)
        for item in self.config["project_files"]:
            label = customtkinter.CTkLabel(list_frame, text=item, anchor="w", width=bar_width)
            label.pack(anchor=tk.W,padx=5)

        # Update scrollregion of the canvas
        def update_scrollregion(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        list_frame.bind("<Configure>", update_scrollregion)

        return frame

    def create_header(self, header_title, header_path=None):
        if header_path!=None:
            customtkinter.CTkLabel(self, text=header_path,
                                   font=('Helvetica', 12),text_color="#3a7ebf").place(x=int(self.width*0.3), y=int(self.height*0.02))
        customtkinter.CTkLabel(self, text=header_title,
                               font=('Helvetica', 32, "bold")).place(x=int(self.width*0.65), y=int(self.height*0.09),anchor=tk.CENTER)
        customtkinter.CTkLabel(self, text="Project "+self.config["project_name"],
                               font=('Helvetica', 20, "bold")).place(x=int(self.width*0.65), y=int(self.height*0.16),anchor=tk.CENTER)

    def window3a_PE_menu(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Analysis Menu")

        # Buttons to analysis windows
        # usage_img = PhotoImage(file='other/usage_icon.png')
        # subgroups_img = PhotoImage(file='other/subgroup_icon.png')
        # embed_img = PhotoImage(file='other/embed_icon.png')
        # classify_img = PhotoImage(file='other/classify_icon.png')
        # remap_img = PhotoImage(file='other/remap_icon.png')
        button_width=int(self.width*0.2)
        button_height=int(self.height*0.15)
        customtkinter.CTkLabel(self, text="Further configure project:",
                               font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width * 0.3), y=int(self.height * 0.2))
        # customtkinter.CTkButton(self, text="Define subgroups \nwithin data",
        #                         command=self.window4a_define_subgroups, image=subgroups_img,
        #                         font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.3), y=int(self.height*0.3))
        # customtkinter.CTkButton(self, text="Compare modules to \nmanual scoring",
        #                         command=self.window4h_pose_vs_BORIS, image=remap_img,
        #                         font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.52), y=int(self.height*0.3))
        # customtkinter.CTkButton(self, text="Manually combine \npose modules",
        #                         command=self.window1_start, image=remap_img,
        #                         font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.74), y=int(self.height*0.3))

        customtkinter.CTkLabel(self, text="Visualize and classify:",
                               font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width * 0.3), y=int(self.height * 0.5))
        # customtkinter.CTkButton(self, text="Analyze pose module \nusage and transitions",
        #                         command=self.window4b_usage_transitions_menu, image=usage_img,
        #                         font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.3), y=int(self.height*0.6))
        # customtkinter.CTkButton(self, text="Embed and measure \ndistance between \ngroups",
        #                         command=self.window4e_embed_menu, image=embed_img,
        #                         font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.52), y=int(self.height*0.6))
        # customtkinter.CTkButton(self, text="Classify conditions",
        #                         command=self.window4g_classify_menu, image=classify_img,
        #                         font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.74), y=int(self.height*0.6))

        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window3b_PS_menu(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Analysis Menu")

        # Buttons to analysis windows
        # usage_img = PhotoImage(file='other/usage_icon.png')
        # subgroups_img = PhotoImage(file='other/subgroup_icon.png')
        # embed_img = PhotoImage(file='other/embed_icon.png')
        # classify_img = PhotoImage(file='other/classify_icon.png')
        # remap_img = PhotoImage(file='other/remap_icon.png')
        menu_item_width=int(self.width*0.15)
        button_width=int(self.width*0.45)
        button_height=int(self.height*0.05)
        customtkinter.CTkLabel(self, text="Further configure project:",
                               font=('Helvetica', 16),width=menu_item_width,height=button_height).place(x=int(self.width * 0.3), y=int(self.height * 0.2))
        customtkinter.CTkButton(self, text="Define subgroups within data",
                                command=self.window4a_define_subgroups,
                                font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.5), y=int(self.height*0.2))
        customtkinter.CTkButton(self, text="Compare modules to manual scoring",
                                command=self.window4b_pose_vs_BORIS,
                                font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.5), y=int(self.height*0.27))
        customtkinter.CTkButton(self, text="Manually combine pose modules",
                                command=self.window1_start,
                                font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.5), y=int(self.height*0.34))

        customtkinter.CTkLabel(self, text="Analyze and visualize:",
                               font=('Helvetica', 16),width=menu_item_width,height=button_height).place(x=int(self.width * 0.3), y=int(self.height * 0.44))
        customtkinter.CTkButton(self, text="Measure pose module usage and transitions",
                                command=self.window5a_usage_transitions_menu,
                                font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.5), y=int(self.height*0.44))
        customtkinter.CTkButton(self, text="Embed and/or measure distance between groups",
                                command=self.window5b_embed_distance_menu,
                                font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.5), y=int(self.height*0.51))
        customtkinter.CTkButton(self, text="Classify and/or regress conditions",
                                command=self.window5c_classify_regress_menu,
                                font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.5), y=int(self.height*0.58))

        customtkinter.CTkLabel(self, text="Model and simulate:",
                               font=('Helvetica', 16),width=menu_item_width,height=button_height).place(x=int(self.width * 0.3), y=int(self.height * 0.68))
        customtkinter.CTkButton(self, text="Fit curve to within-session pose data",
                                command=self.window6a_fit_curve_menu,
                                font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.5), y=int(self.height*0.68))
        customtkinter.CTkButton(self, text="Simulate module labels, usage, or transitions",
                                command=self.window6b_simulate_menu,
                                font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.5), y=int(self.height*0.75))
        customtkinter.CTkButton(self, text="Get cumulative distribution function from real or simulated behavior",
                                command=self.window6c_cdf_menu,
                                font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.5), y=int(self.height*0.82))

        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))

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
        instruction_block = tk.Label(self,text=instruction_text, wraplength=int(self.width * 0.65),
            padx=10,pady=10,bg="darkgray",fg="white",justify=tk.LEFT)

        # Place the label in the window
        instruction_block.place(x=int(self.width * 0.3),y=int(self.height * 0.22),anchor=tk.NW)
        customtkinter.CTkButton(self,
                                text="Edit config.yaml",font=('Helvetica', 16),
                                command=lambda: metadata.edit_config(self.config_path)).place(x=int(self.width*0.6), y=int(self.height*0.75))
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window4b_pose_vs_BORIS(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Compare Pose Data to Manual Scoring",header_path="Configure ▶ Manual scoring comparison")

        customtkinter.CTkButton(self, text="Update config file with manual scoring info from BORIS",
                                command=self.window4b2_boris_config,height=int(self.width*0.1),width=int(self.width*0.6),
                                font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.4),anchor=tk.CENTER)
        customtkinter.CTkButton(self, text="Get and plot pose module to BORIS comparison matrix",
                                command=self.window4b3_pose_vs_BORIS,height=int(self.width*0.1),width=int(self.width*0.6),
                                font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.6),anchor=tk.CENTER)
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window4b2_boris_config(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Compare Pose Data to Manual Scoring",header_path="Configure ▶ Manual scoring comparison ▶ Update config")

        instruction_text = """For this step, you will need to manually edit the config file, which you should be able to access by pressing the 'Edit config.yaml' button below.
        You'll have to edit the boris_directory and boris_to_pose_pairings."""
        instruction_block = tk.Label(self,text=instruction_text, wraplength=int(self.width * 0.65),
            padx=10,pady=10,bg="darkgray",fg="white",justify=tk.LEFT)

        instruction_block.place(x=int(self.width * 0.3),y=int(self.height * 0.3),anchor=tk.NW)
        customtkinter.CTkButton(self,
                                text="Edit config.yaml",
                                font=('Helvetica', 16),
                                command=lambda: metadata.edit_config(self.config["project_directory"] + "/config.yaml")).place(x=int(self.width * 0.65),y=int(self.height * 0.6),anchor=tk.CENTER)
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ update config and go back to BORIS menu", command=self.load_project_BORIS,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.76))
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window4b3_pose_vs_BORIS(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Compare Pose Data to Manual Scoring",header_path="Configure ▶ Manual scoring comparison ▶ Plot comparison matrix")

        customtkinter.CTkLabel(self, text="Get a matrix showing overlap between pose modules and behaviors manually scored in BORIS.",
                               font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.4),anchor=tk.CENTER)

        customtkinter.CTkButton(self, text="Compare!",
                                command=lambda: self.plot_pose_vs_BORIS(),
                                font=('Helvetica', 16)).place(x=int(self.width*0.85), y=int(self.height*0.9))
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window5a_usage_transitions_menu(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Pose Usage and Transition Analysis Menu",header_path="Viz & Analyze ▶ Usage & Transitions")

        customtkinter.CTkLabel(self, text="What kind of plot would you like to do?",
                               font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.3),anchor=tk.CENTER)

        customtkinter.CTkButton(self, text="1. Get pose usage and transitions",
                                command=lambda: self.window5a1_get_usage_transitions(),height=int(self.height*0.2),width=int(self.width*0.6),
                                font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.35),anchor=tk.CENTER)
        customtkinter.CTkButton(self, text="2. Plot pose usage and/or transitions",
                                command=lambda: self.window5a2_plot_usage_transitions(),height=int(self.height*0.2),width=int(self.width*0.6),
                                font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.58),anchor=tk.CENTER)

        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))
    def window5a1_get_usage_transitions(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Measure Pose Usage and Transitions",header_path="Viz & Analyze ▶ Usage & Transitions ▶ Measure")

        customtkinter.CTkLabel(self, text="Enter info about your analysis.",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.25))

        customtkinter.CTkLabel(self, text="Start time (seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.35))
        start = customtkinter.CTkEntry(self)
        start.place(x=int(self.width*0.45), y=int(self.height*0.35))
        customtkinter.CTkLabel(self, text="End time (seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.45))
        end = customtkinter.CTkEntry(self)
        end.place(x=int(self.width*0.45), y=int(self.height*0.45))
        customtkinter.CTkLabel(self, text="Data to include",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.55))
        dropdown = MultiDropDown(self,options=["all combined"]+list(self.config["subgroups"].keys()))
        dropdown.place(x=int(self.width*0.45), y=int(self.height*0.55))
        customtkinter.CTkLabel(self, text="Binsize (seconds; blank for no binning)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.35))
        binsize = customtkinter.CTkEntry(self)
        binsize.place(x=int(self.width*0.65), y=int(self.height*0.45))
        save_option = customtkinter.StringVar(value="scatter")
        save_options = ["Save to pickle (for further using in MARIPoSA)",
                         "Save to csv (for external use)",
                         "Save to pickle AND csv"]
        style_vars = ["pickle", "csv", "both"]
        for s in range(len(style_vars)):
            radio_btn = customtkinter.CTkRadioButton(self, text=save_options[s],
                                                     variable=save_option, value=style_vars[s],
                                                     font=('Helvetica', 16))
            radio_btn.place(x=int(self.width*0.65), y=int(self.height*(0.55+0.08*s)))
        customtkinter.CTkButton(self, text="Analyze & Save",
                                command=lambda: self.save_usage_transitions(int(start.get()),
                                                                            int(end.get()),
                                                                            dropdown.get_selected_values(),
                                                                            binsize.get(),
                                                                            save_to=save_option.get()),
                                font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.8))
        customtkinter.CTkButton(self, text="Proceed to plotting",
                                command=lambda: self.window5a2_plot_usage_transitions(),
                                font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.9))
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window5a2_plot_usage_transitions(self,pickle_path_prefill = "/path/to/usage.pickle",
                                         tx_pickle_path_prefill="/path/to/transitions.pickle",
                                         plot_option=None):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Get & Plot Pose Usage",
                           header_path="Viz & Analyze ▶ Usage & Transitions ▶ Plot")

        customtkinter.CTkLabel(self, text="Path to module usage .pickle:",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.25))
        pickle_path = customtkinter.CTkEntry(self)
        pickle_path.insert(0, pickle_path_prefill)
        pickle_path.place(x=int(self.width*0.6), y=int(self.height*0.25))
        customtkinter.CTkButton(self,
                                text="Browse",
                                command=lambda: self.window_browse(pickle_path, type="file")).place(x=int(self.width*0.8), y=int(self.height*0.25))
        customtkinter.CTkLabel(self, text="Path to module transitions .pickle:",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.32))
        tx_pickle_path = customtkinter.CTkEntry(self)
        tx_pickle_path.insert(0, tx_pickle_path_prefill)
        tx_pickle_path.place(x=int(self.width*0.6), y=int(self.height*0.32))
        customtkinter.CTkButton(self,
                                text="Browse",
                                command=lambda: self.window_browse(tx_pickle_path, type="file")).place(x=int(self.width*0.8), y=int(self.height*0.32))
        if plot_option==None:
            customtkinter.CTkLabel(self, text="What type of plot do you want to make?",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.4))
            customtkinter.CTkButton(self, text="Usage comparison",
                                    command=lambda: self.window5a2_plot_usage_transitions(pickle_path_prefill=pickle_path.get(),
                                                                                          tx_pickle_path_prefill=tx_pickle_path.get(),
                                                                                          plot_option="usage"),
                                    font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.45))
            customtkinter.CTkButton(self, text="Pairwise network\ncomparison",
                                    command=lambda: self.window5a2_plot_usage_transitions(pickle_path_prefill=pickle_path.get(),
                                                                                          tx_pickle_path_prefill=tx_pickle_path.get(),
                                                                                          plot_option="network"),
                                    font=('Helvetica', 16)).place(x=int(self.width*0.55), y=int(self.height*0.45))
            customtkinter.CTkButton(self, text="Within-session\ntime dynamic usage",
                                    command=lambda: self.window5a2_plot_usage_transitions(pickle_path_prefill=pickle_path.get(),
                                                                                          tx_pickle_path_prefill=tx_pickle_path.get(),
                                                                                          plot_option="sandplot"),
                                    font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.45))
        elif plot_option=="usage":
            customtkinter.CTkLabel(self, text="Selected plot type: USAGE",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.4))
            customtkinter.CTkLabel(self, text="Colormap",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.5))
            color = customtkinter.CTkComboBox(self, values=["jet", "cividis", "viridis", "magma"])
            color.place(x=int(self.width*0.75), y=int(self.height*0.5))
            color.set("jet")
            # Choose style for plot
            customtkinter.CTkLabel(self, text="What should the usage plot style be?",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.5))
            style = customtkinter.StringVar(value="scatter")
            style_options = ["Bar with scattered individual points",
                             "Bar with standard error of the mean",
                             "Points with standard error of the mean",
                             "Stacked means"]
            style_vars = ["bar_scatter", "bar_error", "points", "stacked"]
            for s in range(len(style_vars)):
                radio_btn = customtkinter.CTkRadioButton(self, text=style_options[s],
                                                         variable=style, value=style_vars[s],
                                                         font=('Helvetica', 16))
                radio_btn.place(x=int(self.width*0.3), y=int(self.height*(0.5+0.06*s)))
            customtkinter.CTkButton(self, text="Plot it!",
                                    command=lambda: self.plot_usage(pickle_path.get(), style.get(), color.get()),
                                    font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.9))

        elif plot_option == "network":
            customtkinter.CTkLabel(self, text="Selected plot type: NETWORK",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.4))
            customtkinter.CTkLabel(self, text="Colormap",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.5))
            color = customtkinter.CTkComboBox(self, values=["bwr", "seismic", "PiYG", "BrBG", "PRGn"])
            color.place(x=int(self.width*0.75), y=int(self.height*0.5))
            color.set("bwr")
            customtkinter.CTkButton(self, text="Plot it!",
                                    command=lambda: self.plot_network(pickle_path.get(), tx_pickle_path.get(), color.get()),
                                    font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.9))
        elif plot_option=="sandplot":
            customtkinter.CTkLabel(self, text="Selected plot type: SANDPLOT",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.4))
            customtkinter.CTkButton(self, text="Plot it!",
                                    command=lambda: self.plot_sandplot(pickle_path.get()),
                                    font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.9))
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window5b_embed_distance_menu(self,pickle_path_prefill = "/path/to/file.pickle",
                                         emb_dist_option=None):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Embed and Distance Menu",header_path="Viz & Analyze ▶ Embed")

        customtkinter.CTkLabel(self, text="Path to module feature object (transitions or usage) .pickle:",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.25))
        pickle_path = customtkinter.CTkEntry(self)
        pickle_path.insert(0, pickle_path_prefill)
        pickle_path.place(x=int(self.width*0.7), y=int(self.height*0.25))
        customtkinter.CTkButton(self,
                                text="Browse",
                                command=lambda: self.window_browse(pickle_path, type="file")).place(x=int(self.width*0.85), y=int(self.height*0.25))

        if emb_dist_option is None:
            customtkinter.CTkLabel(self, text="Embed or measure distance between groups?",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.4))
            customtkinter.CTkButton(self, text="Embed",
                                    command=lambda: self.window5b_embed_distance_menu(
                                        pickle_path_prefill=pickle_path.get(),
                                        emb_dist_option="embed"),
                                    font=('Helvetica', 16)).place(x=int(self.width * 0.3), y=int(self.height * 0.45))
            customtkinter.CTkButton(self, text="Distance",
                                    command=lambda: self.window5b_embed_distance_menu(
                                        pickle_path_prefill=pickle_path.get(),
                                        emb_dist_option="distance"),
                                    font=('Helvetica', 16)).place(x=int(self.width * 0.55), y=int(self.height * 0.45))
        elif emb_dist_option=="embed":
            customtkinter.CTkLabel(self, text="Embed or measure distance option: EMBED",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.4))
            customtkinter.CTkLabel(self, text="Dimensionality reduction method",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.6))
            embedding_type = customtkinter.CTkComboBox(self, values=["pca", "lda"])
            embedding_type.place(x=int(self.width*0.65), y=int(self.height*0.6))
            embedding_type.set("pca")
            customtkinter.CTkLabel(self, text="Colormap",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.7))
            color = customtkinter.CTkComboBox(self, values=["jet", "cividis", "viridis", "magma"])
            color.place(x=int(self.width*0.65), y=int(self.height*0.7))
            color.set("jet")
            customtkinter.CTkButton(self, text="Plot embeddings",
                                    command=lambda: self.embed_plot(pickle_path.get(), embedding_type.get(), color.get()),
                                    font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.9))
        elif emb_dist_option=="distance":
            customtkinter.CTkLabel(self, text="Embed or measure distance option: DISTANCE",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.4))
            customtkinter.CTkLabel(self, text="Distance metric to use",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.6))
            dist_metric = customtkinter.CTkComboBox(self, values=["euclidean","cityblock","correlation"])
            dist_metric.place(x=int(self.width*0.5), y=int(self.height*0.6))
            dist_metric.set("euclidean")
            customtkinter.CTkLabel(self, text="Pairwise or centroid",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.7))
            pairwise_centroid_opt = customtkinter.CTkComboBox(self, values=["pairwise", "centroid"])
            pairwise_centroid_opt.place(x=int(self.width*0.5), y=int(self.height*0.7))
            pairwise_centroid_opt.set("centroid")
            customtkinter.CTkLabel(self, text="Plot type",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.6))
            plot_type = customtkinter.CTkComboBox(self, values=["boxplot", "heatmap"])
            plot_type.place(x=int(self.width*0.8), y=int(self.height*0.6))
            plot_type.set("heatmap")
            customtkinter.CTkButton(self, text="Plot distance",
                                    command=lambda: self.distance_plot(pickle_path.get(), dist_metric.get(),
                                                                       pairwise_centroid_opt.get(),
                                                                       plot_type.get()),
                                    font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.9))
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))


    def window5c_classify_regress_menu(self,pickle_path_prefill="/path/to/file.pickle",classify_regress_opt=None):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Classify and Embed Menu",header_path="Viz & Analyze ▶ Classify/Regress")

        customtkinter.CTkLabel(self, text="Path to module feature object (transitions or usage) .pickle:",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.25))
        pickle_path = customtkinter.CTkEntry(self)
        pickle_path.insert(0, pickle_path_prefill)
        pickle_path.place(x=int(self.width*0.7), y=int(self.height*0.25))
        customtkinter.CTkButton(self,
                                text="Browse",
                                command=lambda: self.window_browse(pickle_path, type="file")).place(x=int(self.width*0.85), y=int(self.height*0.25))
        if classify_regress_opt is None:
            customtkinter.CTkLabel(self, text="Would you like to predict independent variable associated with subgroups using classification or regression?",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.4))
            customtkinter.CTkButton(self, text="Classify",
                                    command=lambda: self.window5c_classify_regress_menu(
                                        pickle_path_prefill=pickle_path.get(),
                                        classify_regress_opt="classify"),
                                    font=('Helvetica', 16)).place(x=int(self.width * 0.3), y=int(self.height * 0.45))
            customtkinter.CTkButton(self, text="Regress",
                                    command=lambda: self.window5c_classify_regress_menu(
                                        pickle_path_prefill=pickle_path.get(),
                                        classify_regress_opt="regress"),
                                    font=('Helvetica', 16)).place(x=int(self.width * 0.55), y=int(self.height * 0.45))
        elif classify_regress_opt=="classify":
            customtkinter.CTkLabel(self, text="Classification method to use",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.6))
            method = customtkinter.CTkComboBox(self, values=["LogisticRegression","LDA","MLP","NaiveBayes","KNN","RandomForest"])
            method.place(x=int(self.width*0.5), y=int(self.height*0.6))
            method.set("LogisticRegression")
            customtkinter.CTkLabel(self, text="Classify or regress option: CLASSIFY",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.4))
            customtkinter.CTkButton(self, text="Leave-one-out-cross-validation (LOOCV)",
                                    command=lambda: self.classify(pickle_path.get(), method.get(), "loocv"),
                                    font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.8))
            customtkinter.CTkButton(self, text="Fit and save classifier",
                                    command=lambda: self.classify(pickle_path.get(), method.get(), "fullfit"),
                                    font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.9))
        elif classify_regress_opt=="regress":
            customtkinter.CTkLabel(self, text="Classify or regress option: REGRESS",
                                   font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.4))
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window6a_fit_curve_menu(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Fit a Curve",header_path="Model & Simulate ▶ Curve Fitting")

        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window6b_simulate_menu(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Simulate Pose Data",header_path="Model & Simulate ▶ Simulate Data")

        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window6c_cdf_menu(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Fit Cumulative Distribution Function",
                           header_path="Model & Simulate ▶ Cumulative Distribution Function")

        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width * 0.3), y=int(self.height * 0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width * 0.3), y=int(self.height * 0.9))
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


class MultiDropDown(customtkinter.CTkFrame):
    def __init__(self, parent, options):
        super().__init__(parent)

        self.menubutton = customtkinter.CTkButton(self, text="Select option(s) ▼", command=self.toggle_menu,
                                                  fg_color="gray16",width=int(self.winfo_screenwidth()*0.6*0.1))
        self.menubutton.pack(padx=0, pady=0)

        # Create a canvas to host the menu items
        self.canvas = tk.Canvas(self, borderwidth=0, background="gray16",
                                width=int(self.winfo_screenwidth()*factor*0.1),
                                height=int(self.winfo_screenheight()*factor*0.15))
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)

        # Create a frame inside the canvas to contain the checkboxes
        self.menu_frame = customtkinter.CTkFrame(self.canvas, corner_radius=0, bg_color="gray16")
        self.canvas.create_window((0, 0), window=self.menu_frame, anchor="nw")

        # Configure the scrollbar
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack the scrollbar and canvas but keep them hidden initially
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack_forget()
        self.scrollbar.pack_forget()

        # Add checkboxes to the menu_frame
        self.choices = {}
        for choice in options:
            var = tk.IntVar(value=0)
            self.choices[choice] = var
            checkbox = customtkinter.CTkCheckBox(self.menu_frame, text=choice, variable=var, onvalue=1, offvalue=0)
            checkbox.pack(anchor="w", padx=5, pady=5)

        # Update the scroll region to encompass the menu_frame
        self.menu_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def toggle_menu(self):
        if self.canvas.winfo_ismapped():
            self.canvas.pack_forget()
            self.scrollbar.pack_forget()
        else:
            self.canvas.pack(side="left", fill="both", expand=True)
            self.scrollbar.pack(side="right", fill="y")
            # self.canvas.lift()  # Ensure it appears on top
            self.canvas.config(scrollregion=self.canvas.bbox("all"))

    # def update_z_order(self, event=None):
    #     """Update the z-order of the canvas when the widget is configured."""
    #     if self.canvas.winfo_ismapped():
    #         self.canvas.lift()

    def get_selected_values(self):
        """Return a list of selected values."""
        return [name for name, var in self.choices.items() if var.get() == 1]


class PlotWindow(customtkinter.CTk):
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
        save_button = customtkinter.CTkButton(self, text="Save Plot", command=lambda: self.save_plot(self.fig))
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
    cProfile.run('app.mainloop()')
    #app.mainloop()
