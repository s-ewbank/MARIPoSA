import tkinter as tk
from tkinter import filedialog, PhotoImage
import customtkinter
from utils import metadata, analysis, plot
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import io

factor=0.6

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


class Application(customtkinter.CTk):
    """
    Main application
    """
    def __init__(self):
        super().__init__()
        self.title("MARIPoSA")
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")
        # Set window style
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        factor = 0.6
        self.width = int(screen_width * factor)
        self.height = int(screen_height * factor)
        x = (screen_width - self.width) // 2
        y = (screen_height - self.height) // 2
        self.geometry(f'{self.width}x{self.height}+{x}+{y}')
        self.projectstart_choice = customtkinter.StringVar(value="New project")  # Default choice
        self.datatype = customtkinter.StringVar(value="B-SOiD")  # Default choice
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
        print('error')
        self.error_label = customtkinter.CTkLabel(self, text=message, text_color="red", bg_color="#ffe5e3",anchor=tk.CENTER)
        self.error_label.place(x=x,y=y,anchor=tk.CENTER)
        self.after(3000, lambda: self.error_label.configure(text="",bg_color="gray16"))

    @staticmethod
    def window_browse(item_path_entry, type="file"):
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
        self.projectstart_choice = customtkinter.StringVar(value="New project")
        customtkinter.CTkLabel(self, text="MARIPoSA 🦋",
                               font=('Helvetica', 32, "bold")).place(x=int(self.width*0.5), y=int(self.height*0.1),anchor=tk.CENTER)
        customtkinter.CTkLabel(self, text="Would you like to start a new project or load a previous project?",
                               font=('Helvetica', 16)).place(x=int(self.width*0.5), y=int(self.height*0.4),anchor=tk.CENTER)
        # Radio buttons for starting new project or loading old project
        projectstart_options = ["New project","Load previous"]
        grid_last = 0
        for option in projectstart_options:
            radio_btn = customtkinter.CTkRadioButton(self, text=option, variable=self.projectstart_choice, value=option,
                                                     font=('Helvetica', 16))
            radio_btn.place(x=int(self.width*0.2), y=int(self.height*(0.5+0.1*grid_last)))
            grid_last = grid_last + 1

        # File path entry and Browse button
        config_path_entry = customtkinter.CTkEntry(self)
        config_path_entry.insert(0, "/path/to/config.yaml")
        config_path_entry.place(x=int(self.width*0.4), y=int(self.height*0.6))
        browse_button = customtkinter.CTkButton(self, text="Browse",
                                                command=lambda: self.window_browse(config_path_entry, type="file"))
        browse_button.place(x=int(self.width*0.6), y=int(self.height*0.6))

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

        customtkinter.CTkButton(self, text="Pose estimation\n\n(DeepLabCut or SLEAP)",
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
            project_name = customtkinter.CTkEntry(self)
            project_name.place(x=int(self.width*0.5), y=int(self.height*0.2))

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
            datatype_options = ["DeepLabCut", "SLEAP"]
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
                                   font=('Helvetica', 16)).place(x=int(self.width * 0.2), y=int(self.height * 0.3))
            data_path_entry = customtkinter.CTkEntry(self)
            data_path_entry.place(x=int(self.width * 0.5), y=int(self.height * 0.3))
            browse_button = customtkinter.CTkButton(self, text="Browse",
                                                    command=lambda: self.window_browse(data_path_entry,
                                                                                       type="directory"))
            browse_button.place(x=int(self.width * 0.7), y=int(self.height * 0.3))

            # Info about data
            customtkinter.CTkLabel(self, text="Source of data for new project",
                                   font=('Helvetica', 16)).place(x=int(self.width * 0.2), y=int(self.height * 0.4))

            # Radio buttons for starting new project or loading old project
            datatype_options = ["B-SOiD", "VAME", "Keypoint-MoSeq"]
            for r, option in enumerate(datatype_options):
                radio_btn = customtkinter.CTkRadioButton(self, text=option, variable=self.datatype, value=option,
                                                         font=('Helvetica', 16))
                radio_btn.place(x=int(self.width * 0.5), y=int(self.height * (0.4 + r * 0.1)))

            customtkinter.CTkLabel(self, text="Frames per second",
                                   font=('Helvetica', 16)).place(x=int(self.width * 0.2), y=int(self.height * 0.7))
            fps = customtkinter.CTkEntry(self)
            fps.place(x=int(self.width * 0.5), y=int(self.height * 0.7))

            # Enter project directory
            customtkinter.CTkLabel(self, text="Destination path for MARIPoSA output",
                                   font=('Helvetica', 16)).place(x=int(self.width * 0.2), y=int(self.height * 0.8))
            project_path_entry = customtkinter.CTkEntry(self)
            project_path_entry.place(x=int(self.width * 0.5), y=int(self.height * 0.8))
            browse_button = customtkinter.CTkButton(self, text="Browse",
                                                    command=lambda: self.window_browse(project_path_entry,
                                                                                       type="directory"))
            browse_button.place(x=int(self.width * 0.7), y=int(self.height * 0.8))

            # Back to the initial view
            customtkinter.CTkButton(self, text="◀", command=self.window1_start,
                                    font=('Helvetica', 16)).place(x=int(self.width * 0.1), y=int(self.height * 0.9))
            customtkinter.CTkButton(self, text="▶",
                                    command=lambda: self.create_PS_project(
                                        project_name.get(), data_path_entry.get(), self.datatype.get(),
                                        project_path_entry.get(), fps.get()),
                                    font=('Helvetica', 16)).place(x=int(self.width * 0.8), y=int(self.height * 0.9))

    def create_PE_project(self, project_name, data_directory, data_source, output_directory,fps):
        metadata.create_PE_project(project_name, data_directory, data_source, output_directory,fps)
        self.clear_window()
        self.project_name = datetime.now().strftime('%y%m%d_') + project_name
        self.config_path = output_directory + "/" + self.project_name + "/config_PE.yaml"
        config = metadata.load_project(self.config_path)
        self.config = config
        self.window3a_PE_menu()

    def create_PS_project(self, project_name, data_directory, data_source, output_directory,fps):
        metadata.create_PS_project(project_name, data_directory, data_source, output_directory,fps)
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
        self.window3b_PS_menu()

    def load_project_BORIS(self):
        self.clear_window()
        config = metadata.load_project(self.config_path)
        self.config = config
        self.window4h_pose_vs_BORIS()

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
        usage_img = PhotoImage(file='other/usage_icon.png')
        subgroups_img = PhotoImage(file='other/subgroup_icon.png')
        embed_img = PhotoImage(file='other/embed_icon.png')
        classify_img = PhotoImage(file='other/classify_icon.png')
        remap_img = PhotoImage(file='other/remap_icon.png')
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
        usage_img = PhotoImage(file='other/usage_icon.png')
        subgroups_img = PhotoImage(file='other/subgroup_icon.png')
        embed_img = PhotoImage(file='other/embed_icon.png')
        classify_img = PhotoImage(file='other/classify_icon.png')
        remap_img = PhotoImage(file='other/remap_icon.png')
        button_width=int(self.width*0.2)
        button_height=int(self.height*0.15)
        customtkinter.CTkLabel(self, text="Further configure project:",
                               font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width * 0.3), y=int(self.height * 0.2))
        customtkinter.CTkButton(self, text="Define subgroups \nwithin data",
                                command=self.window4a_define_subgroups, image=subgroups_img,
                                font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.3), y=int(self.height*0.3))
        customtkinter.CTkButton(self, text="Compare modules to \nmanual scoring",
                                command=self.window4h_pose_vs_BORIS, image=remap_img,
                                font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.52), y=int(self.height*0.3))
        customtkinter.CTkButton(self, text="Manually combine \npose modules",
                                command=self.window1_start, image=remap_img,
                                font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.74), y=int(self.height*0.3))

        customtkinter.CTkLabel(self, text="Visualize and classify:",
                               font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width * 0.3), y=int(self.height * 0.5))
        customtkinter.CTkButton(self, text="Analyze pose module \nusage and transitions",
                                command=self.window4b_usage_transitions_menu, image=usage_img,
                                font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.3), y=int(self.height*0.6))
        customtkinter.CTkButton(self, text="Embed and measure \ndistance between \ngroups",
                                command=self.window4e_embed_menu, image=embed_img,
                                font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.52), y=int(self.height*0.6))
        customtkinter.CTkButton(self, text="Classify conditions",
                                command=self.window4g_classify_menu, image=classify_img,
                                font=('Helvetica', 16),width=button_width,height=button_height).place(x=int(self.width*0.74), y=int(self.height*0.6))

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
        edit_config_button = customtkinter.CTkButton(self,
                                                     text="Edit config.yaml",font=('Helvetica', 16),
                                                     command=lambda: metadata.edit_config(self.config_path))
        edit_config_button.place(x=int(self.width*0.6), y=int(self.height*0.75))
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window4b_usage_transitions_menu(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Pose Usage and Transition Analysis Menu",header_path="Viz & Analyze ▶ Usage Analysis")

        customtkinter.CTkLabel(self, text="What kind of plot would you like to generate?",
                               font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.3),anchor=tk.CENTER)

        # Next menu options
        # customtkinter.CTkButton(self, text="Plot pose usage for all individuals",
        #                         command=lambda: self.window4c_usage_transitions("no_subgroups"),
        #                         font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.35),anchor=tk.CENTER)

        customtkinter.CTkButton(self, text="Get and plot pose usage",
                                command=lambda: self.window4c_usage_transitions("subgroups"),height=int(self.height*0.15),width=int(self.width*0.6),
                                font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.35),anchor=tk.CENTER)

        # customtkinter.CTkButton(self, text="Network plot of usage and transitions",
        #                         command=lambda: self.window4d_network("single"),
        #                         font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.35),anchor=tk.CENTER)
        customtkinter.CTkButton(self, text="Network plot of usage and transitions for subgroups",
                                command=lambda: self.window4d_network("comparison"),height=int(self.height*0.15),width=int(self.width*0.6),
                                font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.52),anchor=tk.CENTER)
        # Next menu options
        customtkinter.CTkButton(self, text="Plot pose module usage\n in a subgroup over the\n course of a single session",
                                command=lambda: self.window4f_usage_overtime("within_session"),height=int(self.height*0.15),width=int(self.width*0.29),
                                font=('Helvetica', 16)).place(x=int(self.width*0.80), y=int(self.height*0.69),anchor=tk.CENTER)
        customtkinter.CTkButton(self, text="Plot pose module usage\n in a subgroup across sessions",
                                command=lambda: self.window4f_usage_overtime("across_sessions"),height=int(self.height*0.15),width=int(self.width*0.29),
                                font=('Helvetica', 16)).place(x=int(self.width*0.5), y=int(self.height*0.69),anchor=tk.CENTER)

        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))
    def window4c_usage_transitions(self,subgroup_option):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Get & Plot Pose Usage",header_path="Viz & Analyze ▶ Usage Analysis ▶ Get & Plot Usage")

        customtkinter.CTkLabel(self, text="Enter info about your analysis and plotting parameters.",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.25))

        customtkinter.CTkLabel(self, text="Start time (seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.35))
        start = customtkinter.CTkEntry(self)
        start.place(x=int(self.width*0.45), y=int(self.height*0.35))
        customtkinter.CTkLabel(self, text="End time (seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.45))
        end = customtkinter.CTkEntry(self)
        end.place(x=int(self.width*0.45), y=int(self.height*0.45))
        customtkinter.CTkLabel(self, text="Data to plot",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.55))
        dropdown = MultiDropDown(self,options=["all combined"]+list(self.config["subgroups"].keys()))
        dropdown.place(x=int(self.width*0.45), y=int(self.height*0.55))

        customtkinter.CTkLabel(self, text="Colormap",
                               font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.75))
        color = customtkinter.CTkComboBox(self, values=["jet", "cividis", "viridis", "magma"])
        color.place(x=int(self.width*0.75), y=int(self.height*0.75))
        color.set("jet")
        # Choose style for plot
        customtkinter.CTkLabel(self, text="What should the plot style be?",
                               font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.35))
        style = customtkinter.StringVar(value="scatter")
        style_options = ["Bar with scattered individual points",
                         "Bar with standard error of the mean",
                         "Points with standard error of the mean"]
        style_vars = ["bar_scatter", "bar_error", "points"]
        for s in range(len(style_vars)):
            radio_btn = customtkinter.CTkRadioButton(self, text=style_options[s],
                                                     variable=style, value=style_vars[s],
                                                     font=('Helvetica', 16))
            radio_btn.place(x=int(self.width*0.65), y=int(self.height*(0.45+0.08*s)))
        customtkinter.CTkButton(self, text="Plot it!",
                                command=lambda: self.plot_usage(int(start.get()), int(end.get()),
                                                                style.get(), color.get(),
                                                                dropdown.get_selected_values()),
                                font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.9))
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window4d_network(self,subgroup_option):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Network Plot",header_path="Viz & Analyze ▶ Usage Analysis ▶ Network Plot")

        customtkinter.CTkLabel(self, text="Enter info about your plotting parameters.",
                               font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.25),anchor=tk.CENTER)

        # Plot
        # Choose start and end time
        # Set groups to compare
        customtkinter.CTkLabel(self, text="Comparison group 1",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.35))
        group1 = customtkinter.CTkComboBox(self, values=list(self.config["subgroups"].keys()))
        group1.place(x=int(self.width*0.45), y=int(self.height*0.35))
        customtkinter.CTkLabel(self, text="Comparison group 2",
                               font=('Helvetica', 16)).place(x=int(self.width*0.6), y=int(self.height*0.35))
        group2 = customtkinter.CTkComboBox(self, values=list(self.config["subgroups"].keys()))
        group2.place(x=int(self.width*0.75), y=int(self.height*0.35))
        customtkinter.CTkLabel(self, text="Start time\n(seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.45))
        start = customtkinter.CTkEntry(self)
        start.place(x=int(self.width*0.45), y=int(self.height*0.45))
        customtkinter.CTkLabel(self, text="End time\n(seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.55))
        end = customtkinter.CTkEntry(self)
        end.place(x=int(self.width*0.45), y=int(self.height*0.55))
        # Choose color
        customtkinter.CTkLabel(self, text="Colormap for plot",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.65))
        color = customtkinter.CTkComboBox(self, values=["bwr", "seismic", "PRGn", "BrBG", "PuOr", "PiYG"])
        color.place(x=int(self.width*0.45), y=int(self.height*0.65))
        color.set("bwr")
        # Choose style for plot
        customtkinter.CTkLabel(self, text="Label module numbers on plot?",
                               font=('Helvetica', 16)).place(x=int(self.width*0.7), y=int(self.height*0.55))
        style = customtkinter.StringVar(value="scatter")
        style_options = ["Yes, label",
                         "No, don't label"]
        style_vars = [1,0]
        for s in range(len(style_vars)):
            radio_btn = customtkinter.CTkRadioButton(self, text=style_options[s],
                                                     variable=style, value=style_vars[s],
                                                     font=('Helvetica', 16))
            radio_btn.place(x=int(self.width*0.7), y=int(self.height*(0.65+0.1*s)))
        customtkinter.CTkButton(self, text="Plot it!",
                                command=lambda: self.plot_network(group1.get(), group2.get(), int(start.get()),
                                                                  int(end.get()), style.get(), color.get(),
                                                                  subgroup_option),
                                font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.9))
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window4f_usage_overtime(self, session_option):
        self.clear_window()
        self.create_sidebar_widget()
        # Title
        if session_option=="within_session":
            title_string="Time Series of Pose Module Usage Within Session"
        elif session_option=="across_sessions":
            title_string="Pose Module Usage Across Sessions"
        self.create_header(title_string,header_path="Viz & Analyze ▶ Usage Analysis ▶ Within-Session Over Time")

        customtkinter.CTkLabel(self, text="Choose the parameters for your plot.",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.25))


        # Plot
        # Set group
        customtkinter.CTkLabel(self, text="Subgroup",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.35))
        group = customtkinter.CTkComboBox(self, values=list(self.config["subgroups"].keys()))
        group.place(x=int(self.width*0.45), y=int(self.height*0.35))
        # Choose start and end time
        customtkinter.CTkLabel(self, text="Start time\n(seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.35))
        start = customtkinter.CTkEntry(self)
        start.place(x=int(self.width*0.8), y=int(self.height*0.35))
        customtkinter.CTkLabel(self, text="Time per block\n(seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.45))
        time_per_block = customtkinter.CTkEntry(self)
        time_per_block.place(x=int(self.width*0.8), y=int(self.height*0.45))
        customtkinter.CTkLabel(self, text="Number of blocks",
                               font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.55))
        n_blocks = customtkinter.CTkEntry(self)
        n_blocks.place(x=int(self.width*0.8), y=int(self.height*0.55))
        # Choose style for plot
        customtkinter.CTkLabel(self, text="What should the \nplot style be?",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.45))
        style = customtkinter.StringVar(value="scatter")
        style_options = ["Area","Line"]
        style_vars = ["area", "line"]
        for s in range(len(style_vars)):
            radio_btn = customtkinter.CTkRadioButton(self, text=style_options[s],
                                                     variable=style, value=style_vars[s],
                                                     font=('Helvetica', 16))
            radio_btn.place(x=int(self.width*0.3), y=int(self.height*(0.55+0.1*s)))
        customtkinter.CTkButton(self, text="Plot it!",
                                command=lambda: self.plot_usage_overtime(group.get(),
                                                                         session_option,
                                                                         start=int(start.get()),
                                                                         time_per_block=int(time_per_block.get()),
                                                                         n_blocks=int(n_blocks.get())),
                                font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.9))
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window4e_embed_menu(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Embed and Distance Menu",header_path="Viz & Analyze ▶ Embed")

        customtkinter.CTkLabel(self, text="What tool do you want to use to embed and/or measure distance?",
                               font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.24),anchor=tk.CENTER)

        customtkinter.CTkButton(self, text="Sum squared difference from control group",
                                command=self.window4g_ssd_embed,height=int(self.height*0.15),width=int(self.width*0.6),
                                font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.35),anchor=tk.CENTER)
        customtkinter.CTkButton(self, text="Principal components analysis",
                                command=self.window4g_pca_embed,height=int(self.height*0.15),width=int(self.width*0.6),
                                font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.52),anchor=tk.CENTER)
        customtkinter.CTkButton(self, text="Linear discriminant analysis",
                                command=self.window4g_lda_embed_classify,height=int(self.height*0.15),width=int(self.width*0.6),
                                font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.69),anchor=tk.CENTER)
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))


    def window4g_classify_menu(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Classify and Embed Menu",header_path="Viz & Analyze ▶ Classify")

        customtkinter.CTkLabel(self, text="What tool do you want to use to classify and/or embed?",
                               font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.24),anchor=tk.CENTER)

        customtkinter.CTkButton(self, text="Linear discriminant analysis",
                                command=self.window4g_lda_embed_classify,height=int(self.height*0.15),width=int(self.width*0.6),
                                font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.35),anchor=tk.CENTER)
        customtkinter.CTkButton(self, text="Logistic regression",
                                command=self.window4g_lr_classify,height=int(self.height*0.15),width=int(self.width*0.6),
                                font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.52),anchor=tk.CENTER)
        customtkinter.CTkButton(self, text="Natural language processing",
                                command=self.window4g_nlp_classify,height=int(self.height*0.15),width=int(self.width*0.6),
                                font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.69),anchor=tk.CENTER)
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window4g_ssd_embed(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Sum Squared Difference Embedding",header_path="Viz & Analyze ▶ Embed ▶ SSD")

        customtkinter.CTkLabel(self, text="Enter info about your analysis and plotting parameters.",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.25))

        # Plot
        # Choose start and end time
        customtkinter.CTkLabel(self, text="Start time (seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.35))
        start = customtkinter.CTkEntry(self)
        start.place(x=int(self.width*0.45), y=int(self.height*0.35))
        customtkinter.CTkLabel(self, text="End time (seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.45))
        end = customtkinter.CTkEntry(self)
        end.place(x=int(self.width*0.45), y=int(self.height*0.45))
        customtkinter.CTkLabel(self, text="Bin size (seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.55))
        binsize = customtkinter.CTkEntry(self)
        binsize.place(x=int(self.width*0.45), y=int(self.height*0.55))
        # Choose color
        customtkinter.CTkLabel(self, text="Colormap for plot",
                               font=('Helvetica', 16)).place(x=int(self.width*0.6), y=int(self.height*0.35))
        color = customtkinter.CTkComboBox(self, values=["jet", "cividis", "viridis", "magma"])
        color.place(x=int(self.width*0.75), y=int(self.height*0.35))
        color.set("jet")
        customtkinter.CTkButton(self, text="Plot embeddings",
                                command=lambda: self.lda("embed",int(start.get()), int(end.get()), int(binsize.get()), color.get(),),
                                font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.76))
        customtkinter.CTkButton(self, text="Classify and evaluate",
                                command=lambda: self.lda("classify_eval",int(start.get()), int(end.get()), int(binsize.get()), color.get()),
                                font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="Save LDA classifier",
                                command=lambda: self.lda("save",int(start.get()), int(end.get()), int(binsize.get()), color.get()),
                                font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.9))
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))


    def window4g_pca_embed(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Principal Components Analysis",header_path="Viz & Analyze ▶ Embed ▶ PCA")

        customtkinter.CTkLabel(self, text="Enter info about your analysis and plotting parameters.",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.25))

        # Plot
        # Choose start and end time
        customtkinter.CTkLabel(self, text="Start time (seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.35))
        start = customtkinter.CTkEntry(self)
        start.place(x=int(self.width*0.45), y=int(self.height*0.35))
        customtkinter.CTkLabel(self, text="End time (seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.45))
        end = customtkinter.CTkEntry(self)
        end.place(x=int(self.width*0.45), y=int(self.height*0.45))
        customtkinter.CTkLabel(self, text="Bin size (seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.55))
        binsize = customtkinter.CTkEntry(self)
        binsize.place(x=int(self.width*0.45), y=int(self.height*0.55))
        # Choose color
        customtkinter.CTkLabel(self, text="Colormap for plot",
                               font=('Helvetica', 16)).place(x=int(self.width*0.6), y=int(self.height*0.35))
        color = customtkinter.CTkComboBox(self, values=["jet", "cividis", "viridis", "magma"])
        color.place(x=int(self.width*0.75), y=int(self.height*0.35))
        color.set("jet")
        customtkinter.CTkButton(self, text="Plot embeddings",
                                command=lambda: self.lda("embed",int(start.get()), int(end.get()), int(binsize.get()), color.get(),),
                                font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.76))
        customtkinter.CTkButton(self, text="Classify and evaluate",
                                command=lambda: self.lda("classify_eval",int(start.get()), int(end.get()), int(binsize.get()), color.get()),
                                font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="Save LDA classifier",
                                command=lambda: self.lda("save",int(start.get()), int(end.get()), int(binsize.get()), color.get()),
                                font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.9))
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))


    def window4g_lda_embed_classify(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Linear Discriminant Analysis",header_path="Viz & Analyze ▶ Embed ▶ LDA")

        customtkinter.CTkLabel(self, text="Enter info about your analysis and plotting parameters.",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.25))

        # Plot
        # Choose start and end time
        customtkinter.CTkLabel(self, text="Start time (seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.35))
        start = customtkinter.CTkEntry(self)
        start.place(x=int(self.width*0.45), y=int(self.height*0.35))
        customtkinter.CTkLabel(self, text="End time (seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.45))
        end = customtkinter.CTkEntry(self)
        end.place(x=int(self.width*0.45), y=int(self.height*0.45))
        customtkinter.CTkLabel(self, text="Bin size (seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.55))
        binsize = customtkinter.CTkEntry(self)
        binsize.place(x=int(self.width*0.45), y=int(self.height*0.55))
        # Choose color
        customtkinter.CTkLabel(self, text="Colormap for plot",
                               font=('Helvetica', 16)).place(x=int(self.width*0.6), y=int(self.height*0.35))
        color = customtkinter.CTkComboBox(self, values=["jet", "cividis", "viridis", "magma"])
        color.place(x=int(self.width*0.75), y=int(self.height*0.35))
        color.set("jet")
        customtkinter.CTkLabel(self, text="Subgroups",
                               font=('Helvetica', 16)).place(x=int(self.width*0.6), y=int(self.height*0.45))
        dropdown = MultiDropDown(self,options=list(self.config["subgroups"].keys()))
        dropdown.place(x=int(self.width*0.75), y=int(self.height*0.45))
        customtkinter.CTkButton(self, text="Plot embeddings",
                                command=lambda: self.lda("embed",int(start.get()), int(end.get()),
                                                         int(binsize.get()), color.get(),
                                                         dropdown.get_selected_values()),
                                font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.76))
        customtkinter.CTkButton(self, text="Classify and evaluate",
                                command=lambda: self.lda("classify_eval",int(start.get()), int(end.get()),
                                                         int(binsize.get()), color.get(),
                                                         dropdown.get_selected_values()),
                                font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="Save LDA classifier",
                                command=lambda: self.lda("save",int(start.get()), int(end.get()),
                                                         int(binsize.get()), color.get(),
                                                         dropdown.get_selected_values()),
                                font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.9))
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window4g_lr_classify(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Logistic Regression",header_path="Viz & Analyze ▶ Classify ▶ LogReg")

        customtkinter.CTkLabel(self, text="Enter info about your classification parameters.",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.25))

        # Choose start and end time
        customtkinter.CTkLabel(self, text="Start time (seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.35))
        start = customtkinter.CTkEntry(self)
        start.place(x=int(self.width*0.45), y=int(self.height*0.35))
        customtkinter.CTkLabel(self, text="End time (seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.45))
        end = customtkinter.CTkEntry(self)
        end.place(x=int(self.width*0.45), y=int(self.height*0.45))
        # # Choose color
        # customtkinter.CTkLabel(self, text="Colormap for plot",
        #                        font=('Helvetica', 16)).grid(row=4, column=0, pady=10, sticky="E")
        # color = customtkinter.CTkComboBox(self, values=["jet", "cividis", "viridis", "magma"])
        # color.grid(row=4, column=1, pady=10, sticky="W")
        # color.set("jet")
        customtkinter.CTkLabel(self, text="Bin size (seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.55))
        binsize = customtkinter.CTkEntry(self)
        binsize.place(x=int(self.width*0.45), y=int(self.height*0.55))
        customtkinter.CTkButton(self, text="Classify and Evaluate",
                                command=lambda: self.lr("classify_eval",int(start.get()), int(end.get()), int(binsize.get())),
                                font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="Save",
                                command=lambda: self.lr("save",int(start.get()), int(end.get()), int(binsize.get())),
                                font=('Helvetica', 16)).place(x=int(self.width*0.8), y=int(self.height*0.9))
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))
        # TODO:: update LDA classifier window to connect to real functions

    def window4g_nlp_classify(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Natural Language Processing Tools",header_path="Viz & Analyze ▶ Classify ▶ NLP")

        customtkinter.CTkLabel(self, text="Enter info about your classification parameters.",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.25))

        # Plot
        # Choose start and end time
        customtkinter.CTkLabel(self, text="Start time (seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.35))
        start = customtkinter.CTkEntry(self)
        start.place(x=int(self.width*0.45), y=int(self.height*0.35))
        customtkinter.CTkLabel(self, text="End time (seconds)",
                               font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.45))
        end = customtkinter.CTkEntry(self)
        end.place(x=int(self.width*0.45), y=int(self.height*0.45))
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))
        # TODO:: update NLP classifier window to connect to real functions

    def window4h_pose_vs_BORIS(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Compare Pose Data to Manual Scoring",header_path="Configure ▶ Manual scoring comparison")

        customtkinter.CTkButton(self, text="Update config file with manual scoring info from BORIS",
                                command=self.window4h2_boris_config,height=int(self.width*0.1),width=int(self.width*0.6),
                                font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.4),anchor=tk.CENTER)
        customtkinter.CTkButton(self, text="Get and plot pose module to BORIS comparison matrix",
                                command=self.window4h3_pose_vs_BORIS,height=int(self.width*0.1),width=int(self.width*0.6),
                                font=('Helvetica', 16)).place(x=int(self.width*0.65), y=int(self.height*0.6),anchor=tk.CENTER)
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window4h2_boris_config(self):
        self.clear_window()
        self.create_sidebar_widget()
        self.create_header("Compare Pose Data to Manual Scoring",header_path="Configure ▶ Manual scoring comparison ▶ Update config")

        instruction_text = """For this step, you will need to manually edit the config file, which you should be able to access by pressing the 'Edit config.yaml' button below.
        You'll have to edit the boris_directory and boris_to_pose_pairings."""
        instruction_block = tk.Label(self,text=instruction_text, wraplength=int(self.width * 0.65),
            padx=10,pady=10,bg="darkgray",fg="white",justify=tk.LEFT)

        instruction_block.place(x=int(self.width * 0.3),y=int(self.height * 0.3),anchor=tk.NW)
        edit_config_button = customtkinter.CTkButton(self,
                                                     text="Edit config.yaml",font=('Helvetica', 16),
                                                     command=lambda: metadata.edit_config(
                                                         self.config["project_directory"] + "/config.yaml"))
        edit_config_button.place(x=int(self.width * 0.65),y=int(self.height * 0.6),anchor=tk.CENTER)
        # Bottom back buttons
        customtkinter.CTkButton(self, text="◀ update config and go back to BORIS menu", command=self.load_project_BORIS,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.76))
        customtkinter.CTkButton(self, text="◀ back to analysis menu", command=self.window3b_PS_menu,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.83))
        customtkinter.CTkButton(self, text="◀◀ back to start", command=self.window1_start,
                                font=('Helvetica', 16)).place(x=int(self.width*0.3), y=int(self.height*0.9))

    def window4h3_pose_vs_BORIS(self):
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

    def plot_usage(self, start, end, style, color, subgroups):
        if subgroups == ["all combined"]:
            labels_df, n_modules = analysis.label_counter_nosubgroups(self.config, start, end)
            fig = plot.plot_module_usage(self.config, labels_df, start, end, int(self.config["fps"]), style=style, cmap=color)
            self.plots_generated = self.plots_generated + 1
            self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
            self.plot_window.mainloop()
        else:
            selected_subgroups=[i for i in subgroups if (i!="all combined")]
            labels_df, n_modules = analysis.label_counter_subgroups(self.config, start, end, selected_subgroups=selected_subgroups)
            fig = plot.plot_module_usage_subgroups(self.config, labels_df, start, end, int(self.config["fps"]), style=style, cmap=color)
            self.plots_generated = self.plots_generated + 1
            self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
            self.plot_window.mainloop()

    def plot_network(self, group1, group2, start, end, style, cmap, subgroup_option):
        if subgroup_option=="single":
            labels_df, n_modules = analysis.label_counter_nosubgroups(self.config, start, end)
            #plot.plot_module_usage(labels_df, start, end, int(self.config["fps"]), style=style, cmap=color)
        elif subgroup_option=="comparison":
            labels_df, n_modules = analysis.label_counter_subgroups(self.config, start, end)
            fig = plot.network_pairwise_comparison(self.config, labels_df, 0, 1200, [group1, group2],
                                                   cmap=cmap,include_labels=bool(style))
            self.plots_generated = self.plots_generated + 1
            self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
            self.plot_window.mainloop()

    def plot_usage_overtime(self, group, session_option, start=None, time_per_block=None, n_blocks=None):
        if session_option=="within_session":
            print(group)
            labels_df, n_modules = analysis.label_counter_subgroups(self.config, start,
                                                                    int(time_per_block*n_blocks),
                                                                    selected_subgroups=[group])
            fig = plot.SandPlotClusterFrequency_OverTime(self.config, labels_df[group], start,
                                                         time_per_block, n_blocks)
            self.plots_generated = self.plots_generated + 1
            self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
            self.plot_window.mainloop()
        elif session_option=="across_sessions":
            labels_df, n_modules = analysis.label_counter_subgroups(self.config, start, end)
            #plot.plot_module_usage_subgroups(labels_df, start, end, int(self.config["fps"]), style=style, cmap=color)

    def plot_pose_vs_BORIS(self):
        BORIS_to_pose_mat, BORIS_to_pose_mat_normalized, loss = analysis.BORIS_to_pose(self.config) # this line is messing up the loaded config
        fig = plot.BORIS_to_pose_matrix_plot(self.config, BORIS_to_pose_mat_normalized)
        self.plots_generated = self.plots_generated + 1
        self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
        self.plot_window.mainloop()

    def lda(self, do, start, end, binsize, cmap, subgroups):
        labels_df, n_modules = analysis.label_counter_subgroups(self.config, start, end, selected_subgroups=subgroups)
        lda, lda_embeddings, label_counts, group_labels, group_dict, nbins = analysis.lda_labels_timebins(self.config, labels_df, binsize, selected_subgroups=subgroups)
        if do=="embed":
            fig = plot.plot_lda(self.config, lda, lda_embeddings, group_labels, nbins, binsize, cmap=cmap, selected_subgroups=subgroups)
            self.plots_generated = self.plots_generated + 1
            self.plot_window = PlotWindow(fig=fig, plot_number=self.plots_generated, master=self)
            self.plot_window.mainloop()
        elif do=="classify_eval":
            print("Classifying and evaluating")
            confusion, class_num, class_labels, accuracy = analysis.loocv_conf_mat(lda, label_counts, group_labels, group_dict)
            fig = plot.plot_conf_mat(confusion, class_num, class_labels, cmap="Greens")
            self.plots_generated = self.plots_generated + 1
            self.plot_window = PlotWindow(fig=fig, plot_number=self.plots_generated, master=self)
            self.plot_window.mainloop()
        elif do=="save":
            print("Saving LDA classifier")
            fig = plot.plot_lda(self.config, lda, lda_embeddings, group_labels, nbins, binsize, cmap=cmap)
            self.plots_generated = self.plots_generated + 1
            self.plot_window = PlotWindow(fig=fig, plot_number=self.plots_generated, master=self)
            self.plot_window.mainloop()

    def lr(self, do, start, end, binsize):
        labels_df, n_modules = analysis.label_counter_subgroups(self.config, start, end)
        lr, group_labels, label_counts, group_dict, nbins = analysis.lr_labels_timebins(self.config, labels_df, binsize)
        if do=="classify_eval":
            print("Classifying and evaluating")
            confusion, class_num, class_labels, accuracy = analysis.loocv_conf_mat(lr, label_counts, group_labels, group_dict)
            fig = plot.plot_conf_mat(confusion, class_num, class_labels, cmap="Greens")
            self.plots_generated = self.plots_generated + 1
            self.plot_window = PlotWindow(fig=fig, plot_number=self.plots_generated, master=self)
            self.plot_window.mainloop()
        # elif do=="save":
        #     print("Saving LDA plot")
        #     fig = plot.plot_lda(self.config, lda, lda_embeddings, group_labels, nbins, binsize, cmap=cmap)
        #     self.plots_generated = self.plots_generated + 1
        #     self.plot_window = PlotWindow(fig=fig, plot_number=self.plots_generated, master=self)
        #     self.plot_window.mainloop()

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

if __name__ == "__main__":
    app = Application()
    app.mainloop()
