from tkinter import filedialog, PhotoImage
import customtkinter
from utils import metadata, analysis, plot
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import io


class Application(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("MARIPOSA")
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

        # Initialize variable to store the project start choice
        self.projectstart_choice = customtkinter.StringVar(value="New project")  # Default choice
        self.datatype = customtkinter.StringVar(value="B-SOiD")  # Default choice
        self.config = None
        self.config_path = None
        self.project_name = None
        self.plot_window = None
        self.plots_generated = 0

        self.window1_start()

    def clear_window(self):
        for widget in self.winfo_children():
            widget.destroy()

    def configure_window(self, nrows, ncols):
        # Configure the first row and first column to expand
        for row_index in range(nrows):
            self.grid_rowconfigure(row_index, weight=1)
        for column_index in range(ncols):
            self.grid_columnconfigure(column_index, weight=1)

    def display_error(self, message, row, column, columnspan):
        self.error_label = customtkinter.CTkLabel(self, text=message, text_color="red", bg_color="#ffe5e3")
        self.error_label.grid(row=row, column=column, columnspan=columnspan)
        self.after(3000, lambda: self.error_label.configure(text="",bg_color=""))

    @staticmethod
    def window_browse(item_path_entry, type="file"):
        # Function to open the file dialog and update the file_path_entry with the selected file path
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
        customtkinter.CTkLabel(self, text="Welcome to MARIPOSA! 🦋",
                               font=('Helvetica', 32, "bold")).grid(row=0, column=0, columnspan=3, pady=10)
        customtkinter.CTkLabel(self, text="Would you like to start a new project or load a previous project?",
                               font=('Helvetica', 16)).grid(row=1, column=0, columnspan=3, pady=10)
        # Radio buttons for starting new project or loading old project
        projectstart_options = ["New project", "Load previous"]
        grid_last = 2
        for option in projectstart_options:
            radio_btn = customtkinter.CTkRadioButton(self, text=option, variable=self.projectstart_choice, value=option,
                                                     font=('Helvetica', 16))
            radio_btn.grid(row=grid_last, column=0, sticky="E")
            grid_last = grid_last + 1

        # File path entry and Browse button
        config_path_entry = customtkinter.CTkEntry(self)
        config_path_entry.insert(0, "/path/to/config.yaml")
        config_path_entry.grid(row=3, column=1, padx=5, pady=5)
        browse_button = customtkinter.CTkButton(self, text="Browse",
                                                command=lambda: self.window_browse(config_path_entry, type="file"))
        browse_button.grid(row=3, column=2, pady=5, sticky="W")

        customtkinter.CTkButton(self, text=">", command=lambda: self.make_or_load_project(config_path_entry.get()),
                                font=('Helvetica', 16)).grid(row=4, column=2, pady=10)
        self.configure_window(5, 3)

    def make_or_load_project(self, config_path):
        if type(self.projectstart_choice)!=str:
            self.projectstart_choice = self.projectstart_choice.get()
        self.config_path = config_path
        if self.projectstart_choice == "New project":
            self.window2_makeproject()
        elif self.projectstart_choice == "Load previous":
            if os.path.exists(config_path):
                self.load_project()
            else:
                error_message="That path does not exist. please enter an existing path."
                self.display_error(error_message,4,0,columnspan=2)

        else:
            print(self.projectstart_choice)

    def window2_makeproject(self):
        self.clear_window()

        customtkinter.CTkLabel(self, text="Create a new project",
                               font=('Helvetica', 32, "bold")).grid(row=0, column=0, columnspan=3, pady=10)

        customtkinter.CTkLabel(self, text="Project name",
                               font=('Helvetica', 16)).grid(row=1, column=0, pady=10, sticky="E")
        project_name = customtkinter.CTkEntry(self)
        project_name.grid(row=1, column=1, pady=10, sticky="W")

        # Enter data directory
        customtkinter.CTkLabel(self, text="Path to data directory",
                               font=('Helvetica', 16)).grid(row=2, column=0, pady=10, sticky="E")
        data_path_entry = customtkinter.CTkEntry(self)
        data_path_entry.grid(row=2, column=1, padx=5, pady=5, sticky="W")
        browse_button = customtkinter.CTkButton(self, text="Browse",
                                                command=lambda: self.window_browse(data_path_entry, type="directory"))
        browse_button.grid(row=2, column=2, pady=5, sticky="W")

        #Info about data
        customtkinter.CTkLabel(self, text="Data type of new project",
                               font=('Helvetica', 16)).grid(row=3, column=0, pady=10, sticky="E")

        # Radio buttons for starting new project or loading old project
        datatype_options = ["B-SOiD", "VAME", "Keypoint-MoSeq"]
        grid_row = 3
        for option in datatype_options:
            radio_btn = customtkinter.CTkRadioButton(self, text=option, variable=self.datatype, value=option,
                                                     font=('Helvetica', 16))
            radio_btn.grid(row=grid_row, column=1, sticky="W")
            grid_row = grid_row + 1

        customtkinter.CTkLabel(self, text="Frames per second",
                               font=('Helvetica', 16)).grid(row=6, column=0, pady=10, sticky="E")
        fps = customtkinter.CTkEntry(self)
        fps.grid(row=6, column=1, pady=10, sticky="W")

        # Enter project directory
        customtkinter.CTkLabel(self, text="Destination path for MARIPOSA output",
                               font=('Helvetica', 16)).grid(row=7, column=0, pady=10, sticky="E")
        project_path_entry = customtkinter.CTkEntry(self)
        project_path_entry.grid(row=7, column=1, padx=5, pady=5, sticky="W")
        browse_button = customtkinter.CTkButton(self, text="Browse",
                                                command=lambda: self.window_browse(project_path_entry,
                                                                                   type="directory"))
        browse_button.grid(row=7, column=2, pady=5, sticky="W")

        # Back to the initial view
        customtkinter.CTkButton(self, text="<", command=self.window1_start,
                                font=('Helvetica', 16)).grid(row=8, column=0, pady=20, sticky="W")
        customtkinter.CTkButton(self, text=">",
                                command=lambda: self.create_project(
                                    project_name.get(), data_path_entry.get(), self.datatype.get(),
                                    project_path_entry.get(),fps.get()),
                                font=('Helvetica', 16)).grid(row=8, column=2, pady=20, sticky="E")
        self.configure_window(9, 3)

    def create_project(self, project_name, data_directory, datatype, output_directory,fps):
        metadata.create_project(project_name, data_directory, datatype, output_directory,fps)
        self.clear_window()
        self.project_name = datetime.now().strftime('%y%m%d_') + project_name
        self.config_path = output_directory + "/" + self.project_name + "/config.yaml"
        config = metadata.load_project(self.config_path)
        self.config = config
        self.window3_menu()

    def load_project(self):
        self.clear_window()
        config = metadata.load_project(self.config_path)
        self.config = config
        self.window3_menu()

    def window3_menu(self):
        self.clear_window()
        customtkinter.CTkLabel(self, text="Analysis Menu",
                               font=('Helvetica', 32, "bold")).grid(row=0, column=1, columnspan=2, pady=10)
        customtkinter.CTkLabel(self, text="Project " + self.config["project_name"],
                               font=('Helvetica', 20, "bold")).grid(row=1, column=1, columnspan=2, pady=10)
        customtkinter.CTkLabel(self, text="What analysis would you like to do?",
                               font=('Helvetica', 16)).grid(row=2, column=1, columnspan=2, pady=10)

        # Buttons to analysis windows
        usage_img = PhotoImage(file='other/posevis icon 1.png')
        subgroups_img = PhotoImage(file='other/posevis icon 2.png')
        customtkinter.CTkButton(self, text="Define subgroups \nwithin data",
                                command=self.window4a_define_subgroups, image=subgroups_img,
                                font=('Helvetica', 16)).grid(row=3, column=1, pady=20,sticky="NSEW")
        customtkinter.CTkButton(self, text="Compare modules to \nmanual scoring", command=self.window4h_pose_vs_BORIS,
                                font=('Helvetica', 16)).grid(row=3, column=2, pady=20,sticky="NSEW")
        customtkinter.CTkButton(self, text="Manually combine pose modules", command=self.window1_start,
                                font=('Helvetica', 16)).grid(row=4, column=1, pady=20,sticky="NSEW")
        customtkinter.CTkButton(self, text="Analyze pose module \nusage and transitions",
                                command=self.window4b_usage_transitions_menu, image=usage_img,
                                font=('Helvetica', 16)).grid(row=4, column=2, pady=20,sticky="NSEW")
        customtkinter.CTkButton(self, text="Analyze pose module \nusage over time",
                                command=self.window4e_usage_overtime_menu,
                                font=('Helvetica', 16)).grid(row=5, column=1, pady=20,sticky="NSEW")
        customtkinter.CTkButton(self, text="Classify and embed \nconditions",
                                command=self.window4g_classify_and_embed_menu,
                                font=('Helvetica', 16)).grid(row=5, column=2, pady=20,sticky="NSEW")
        customtkinter.CTkButton(self, text="<< back to start", command=self.window1_start,
                                font=('Helvetica', 16)).grid(row=6, column=1, pady=20, padx=20, sticky="W")
        self.configure_window(7, 4)

    def window4a_define_subgroups(self):
        self.clear_window()
        # Title
        customtkinter.CTkLabel(self,text="Define Subgroups",
                               font=('Helvetica', 32, "bold")).grid(row=0, column=0, columnspan=3, pady=10)
        customtkinter.CTkLabel(self, text="Project " + self.config["project_name"],
                               font=('Helvetica', 20, "bold")).grid(row=1, column=0, columnspan=3, pady=10)
        instruction_block = customtkinter.CTkTextbox(self,width=int(self.width*0.8))
        instruction_block.grid(row=2,column=0,columnspan=3,padx=10, pady=10)
        instruction_text="""For this step, you will need to manually edit the config file, which you should be able to access by pressing the 'Edit config.yaml' button below. In the file, there is a section called subgroups with all your files listed as such:
        
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
        instruction_block.insert("1.0", instruction_text)
        edit_config_button = customtkinter.CTkButton(self,
                                                     text="Edit config.yaml",
                                                     command=lambda: metadata.edit_config(self.config["project_directory"]+"/config.yaml"))
        edit_config_button.grid(row=3, column=0, columnspan=3, pady=5)
        # Bottom back buttons
        customtkinter.CTkButton(self, text="< update config and go back to analysis menu", command=self.load_project,
                                font=('Helvetica', 16)).grid(row=4, column=0, columnspan=3, pady=5, padx=20, sticky="W")
        customtkinter.CTkButton(self, text="<< back to start", command=self.window1_start,
                                font=('Helvetica', 16)).grid(row=5, column=0, columnspan=3, pady=5, padx=20, sticky="W")
        self.configure_window(6, 3)

    def window4b_usage_transitions_menu(self):
        self.clear_window()
        self.configure_window(5, 4)
        # Title
        customtkinter.CTkLabel(self, text="Pose Usage and Transition Analysis Menu",
                               font=('Helvetica', 32, "bold")).grid(row=0, column=0, columnspan=4, pady=10)
        customtkinter.CTkLabel(self, text="Project " + self.config["project_name"],
                               font=('Helvetica', 20, "bold")).grid(row=1, column=0, columnspan=4, pady=10)
        customtkinter.CTkLabel(self, text="What kind of plot would you like to generate?",
                               font=('Helvetica', 16)).grid(row=2, column=0, columnspan=4, pady=10)

        # Next menu options
        customtkinter.CTkButton(self, text="Plot pose usage for all individuals",
                                command=lambda: self.window4c_usage_transitions("no_subgroups"),
                                font=('Helvetica', 16)).grid(row=3, column=1, pady=20, sticky="NSEW")
        customtkinter.CTkButton(self, text="Plot pose usage for subgroups",
                                command=lambda: self.window4c_usage_transitions("subgroups"),
                                font=('Helvetica', 16)).grid(row=3, column=2, pady=20, sticky="NSEW")
        customtkinter.CTkButton(self, text="Network plot of usage and transitions",
                                command=lambda: self.window4d_network("single"),
                                font=('Helvetica', 16)).grid(row=4, column=1, pady=20, sticky="NSEW")
        customtkinter.CTkButton(self, text="Network plot of usage and transitions \nfor subgroups",
                                command=lambda: self.window4d_network("comparison"),
                                font=('Helvetica', 16)).grid(row=4, column=2, pady=20, sticky="NSEW")
        # Bottom back buttons
        customtkinter.CTkButton(self, text="< back to analysis menu", command=self.window3_menu,
                                font=('Helvetica', 16)).grid(row=5, column=1, pady=20, padx=20, sticky="W")
        customtkinter.CTkButton(self, text="<< back to start", command=self.window1_start,
                                font=('Helvetica', 16)).grid(row=6, column=1, pady=20, padx=20, sticky="W")
    def window4c_usage_transitions(self,subgroup_option):
        self.clear_window()
        self.configure_window(7, 4)
        # Title
        if subgroup_option=="no_subgroups":
            title_string="Pose Usage and Transition Analysis - All Individuals"
        elif subgroup_option=="subgroups":
            title_string="Pose Usage and Transition Analysis - Subgroups"
        customtkinter.CTkLabel(self, text=title_string,
                               font=('Helvetica', 32, "bold")).grid(row=0, column=0, columnspan=4, pady=10)
        customtkinter.CTkLabel(self, text="Project " + self.config["project_name"],
                               font=('Helvetica', 20, "bold")).grid(row=1, column=1, columnspan=2, pady=10)
        customtkinter.CTkLabel(self, text="Enter info about your analysis and plotting parameters.",
                               font=('Helvetica', 16)).grid(row=2, column=0, columnspan=4, pady=10)

        # Plot
        # Choose start and end time
        customtkinter.CTkLabel(self, text="Start time (seconds)",
                               font=('Helvetica', 16)).grid(row=3, column=0, pady=10, sticky="E")
        start = customtkinter.CTkEntry(self)
        start.grid(row=3, column=1, pady=10, sticky="W")
        customtkinter.CTkLabel(self, text="End time (seconds)",
                               font=('Helvetica', 16)).grid(row=3, column=2, pady=10, sticky="E")
        end = customtkinter.CTkEntry(self)
        end.grid(row=3, column=3, pady=10, sticky="W")
        # Choose color
        customtkinter.CTkLabel(self, text="Colormap for plot",
                               font=('Helvetica', 16)).grid(row=4, column=0, pady=10, sticky="E")
        color = customtkinter.CTkComboBox(self, values=["jet", "cividis", "viridis", "magma"])
        color.grid(row=4, column=1, pady=10, sticky="W")
        color.set("jet")
        # Choose style for plot
        customtkinter.CTkLabel(self, text="What should the \nplot style be?",
                               font=('Helvetica', 16)).grid(row=4, column=2, rowspan=2, pady=10, sticky="E")
        style = customtkinter.StringVar(value="scatter")
        style_options = ["Bar with scattered individual points",
                         "Bar with standard error of the mean",
                         "Points with standard error of the mean"]
        style_vars = ["bar_scatter", "bar_error", "points"]
        for s in range(len(style_vars)):
            radio_btn = customtkinter.CTkRadioButton(self, text=style_options[s],
                                                     variable=style, value=style_vars[s],
                                                     font=('Helvetica', 16))
            radio_btn.grid(row=4 + s, column=3, sticky="W")
        customtkinter.CTkButton(self, text="Plot it!",
                                command=lambda: self.plot_usage(int(start.get()), int(end.get()),
                                                                style.get(), color.get(), subgroup_option),
                                font=('Helvetica', 16)).grid(row=7, column=3, pady=20)
        # Bottom back buttons
        customtkinter.CTkButton(self, text="< back to analysis menu", command=self.window3_menu,
                                font=('Helvetica', 16)).grid(row=6, column=0, pady=20, padx=20, sticky="W")
        customtkinter.CTkButton(self, text="<< back to start", command=self.window1_start,
                                font=('Helvetica', 16)).grid(row=7, column=0, pady=20, padx=20, sticky="W")

    def window4d_network(self,subgroup_option):
        self.clear_window()
        self.configure_window(5, 4)
        # Title
        if subgroup_option=="single":
            title_string="Network Plot - Single Group"
        elif subgroup_option=="comparison":
            title_string="Network Plot - Comparison"
        customtkinter.CTkLabel(self, text=title_string,
                               font=('Helvetica', 32, "bold")).grid(row=0, column=0, columnspan=4, pady=10)
        customtkinter.CTkLabel(self, text="Project " + self.config["project_name"],
                               font=('Helvetica', 20, "bold")).grid(row=1, column=1, columnspan=2, pady=10)
        customtkinter.CTkLabel(self, text="Enter info about your plotting parameters.",
                               font=('Helvetica', 16)).grid(row=2, column=0, columnspan=4, pady=10)

        # Plot
        # Choose start and end time
        customtkinter.CTkLabel(self, text="Start time (seconds)",
                               font=('Helvetica', 16)).grid(row=3, column=0, pady=10, sticky="E")
        start = customtkinter.CTkEntry(self)
        start.grid(row=3, column=1, pady=10, sticky="W")
        customtkinter.CTkLabel(self, text="End time (seconds)",
                               font=('Helvetica', 16)).grid(row=3, column=2, pady=10, sticky="E")
        end = customtkinter.CTkEntry(self)
        end.grid(row=3, column=3, pady=10, sticky="W")
        # Set groups to compare
        customtkinter.CTkLabel(self, text="Comparison group 1",
                               font=('Helvetica', 16)).grid(row=4, column=0, pady=10, sticky="E")
        group1 = customtkinter.CTkComboBox(self, values=list(self.config["subgroups"].keys()))
        group1.grid(row=4, column=1, pady=10, sticky="W")
        customtkinter.CTkLabel(self, text="Comparison group 2",
                               font=('Helvetica', 16)).grid(row=4, column=2, pady=10, sticky="E")
        group2 = customtkinter.CTkComboBox(self, values=list(self.config["subgroups"].keys()))
        group2.grid(row=4, column=3, pady=10, sticky="W")
        # Choose color
        customtkinter.CTkLabel(self, text="Colormap for plot",
                               font=('Helvetica', 16)).grid(row=5, column=0, pady=10, sticky="E")
        color = customtkinter.CTkComboBox(self, values=["bwr", "seismic", "PRGn", "BrBG", "PuOr", "PiYG"])
        color.grid(row=5, column=1, pady=10, sticky="W")
        color.set("bwr")
        # Choose style for plot
        customtkinter.CTkLabel(self, text="Label module numbers on plot?",
                               font=('Helvetica', 16)).grid(row=5, column=2, rowspan=2, pady=10, sticky="E")
        style = customtkinter.StringVar(value="scatter")
        style_options = ["Yes, label",
                         "No, don't label"]
        style_vars = [1,0]
        for s in range(len(style_vars)):
            radio_btn = customtkinter.CTkRadioButton(self, text=style_options[s],
                                                     variable=style, value=style_vars[s],
                                                     font=('Helvetica', 16))
            radio_btn.grid(row=5 + s, column=3, sticky="W")
        customtkinter.CTkButton(self, text="Plot it!",
                                command=lambda: self.plot_network(group1.get(), group2.get(), int(start.get()),
                                                                  int(end.get()), style.get(), color.get(),
                                                                  subgroup_option),
                                font=('Helvetica', 16)).grid(row=7, column=3, pady=20)
        # Bottom back buttons
        customtkinter.CTkButton(self, text="< back to analysis menu", command=self.window3_menu,
                                font=('Helvetica', 16)).grid(row=6, column=0, pady=20, padx=20, sticky="W")
        customtkinter.CTkButton(self, text="<< back to start", command=self.window1_start,
                                font=('Helvetica', 16)).grid(row=7, column=0, pady=20, padx=20, sticky="W")

    def window4e_usage_overtime_menu(self):
        self.clear_window()
        self.configure_window(5, 4)
        # Title
        customtkinter.CTkLabel(self, text="Pose Usage Over Time Analysis Menu",
                               font=('Helvetica', 32, "bold")).grid(row=0, column=0, columnspan=4, pady=10)
        customtkinter.CTkLabel(self, text="Project " + self.config["project_name"],
                               font=('Helvetica', 20, "bold")).grid(row=1, column=0, columnspan=4, pady=10)
        customtkinter.CTkLabel(self, text="What kind of plot would you like to generate?",
                               font=('Helvetica', 16)).grid(row=2, column=0, columnspan=4, pady=10)

        # Next menu options
        customtkinter.CTkButton(self, text="Plot pose module usage in a subgroup over the course of a single session",
                                command=lambda: self.window4f_usage_overtime("within_session"),
                                font=('Helvetica', 16)).grid(row=3, column=1, columnspan=2, pady=20, sticky="NSEW")
        customtkinter.CTkButton(self, text="Plot pose module usage in a subgroup across sessions",
                                command=lambda: self.window4f_usage_overtime("across_sessions"),
                                font=('Helvetica', 16)).grid(row=4, column=1, columnspan=2, pady=20, sticky="NSEW")

        # Bottom back buttons
        customtkinter.CTkButton(self, text="< back to analysis menu", command=self.window3_menu,
                                font=('Helvetica', 16)).grid(row=5, column=1, pady=20, padx=20, sticky="W")
        customtkinter.CTkButton(self, text="<< back to start", command=self.window1_start,
                                font=('Helvetica', 16)).grid(row=6, column=1, pady=20, padx=20, sticky="W")

    def window4f_usage_overtime(self, session_option):
        self.clear_window()
        self.configure_window(5, 4)
        # Title
        if session_option=="within_session":
            title_string="Time Series of Pose Module Usage Within Session"
        elif session_option=="across_sessions":
            title_string="Pose Module Usage Across Sessions"
        customtkinter.CTkLabel(self, text=title_string,
                               font=('Helvetica', 32, "bold")).grid(row=0, column=0, columnspan=4, pady=10)
        customtkinter.CTkLabel(self, text="Project " + self.config["project_name"],
                               font=('Helvetica', 20, "bold")).grid(row=1, column=0, columnspan=4, pady=10)
        customtkinter.CTkLabel(self, text="Choose the parameters for your plot.",
                               font=('Helvetica', 16)).grid(row=2, column=0, columnspan=4, pady=10)


        # Plot
        # Set group
        customtkinter.CTkLabel(self, text="Subgroup",
                               font=('Helvetica', 16)).grid(row=3, column=0, pady=10, sticky="E")
        group = customtkinter.CTkComboBox(self, values=list(self.config["subgroups"].keys()))
        group.grid(row=3, column=1, pady=10, sticky="W")
        # Choose start and end time
        customtkinter.CTkLabel(self, text="Start time (seconds)",
                               font=('Helvetica', 16)).grid(row=4, column=0, pady=10, sticky="E")
        start = customtkinter.CTkEntry(self)
        start.grid(row=4, column=1, pady=10, sticky="W")
        customtkinter.CTkLabel(self, text="Time per block (seconds)",
                               font=('Helvetica', 16)).grid(row=5, column=0, pady=10, sticky="E")
        time_per_block = customtkinter.CTkEntry(self)
        time_per_block.grid(row=5, column=1, pady=10, sticky="W")
        customtkinter.CTkLabel(self, text="Number of blocks",
                               font=('Helvetica', 16)).grid(row=6, column=0, pady=10, sticky="E")
        n_blocks = customtkinter.CTkEntry(self)
        n_blocks.grid(row=6, column=1, pady=10, sticky="W")
        # Choose style for plot
        customtkinter.CTkLabel(self, text="What should the \nplot style be?",
                               font=('Helvetica', 16)).grid(row=5, column=2, rowspan=2, pady=10, sticky="E")
        style = customtkinter.StringVar(value="scatter")
        style_options = ["Area","Line"]
        style_vars = ["area", "line"]
        for s in range(len(style_vars)):
            radio_btn = customtkinter.CTkRadioButton(self, text=style_options[s],
                                                     variable=style, value=style_vars[s],
                                                     font=('Helvetica', 16))
            radio_btn.grid(row=5 + s, column=3, sticky="W")
        customtkinter.CTkButton(self, text="Plot it!",
                                command=lambda: self.plot_usage_overtime(group.get(),
                                                                         session_option,
                                                                         start=int(start.get()),
                                                                         time_per_block=int(time_per_block.get()),
                                                                         n_blocks=int(n_blocks.get())),
                                font=('Helvetica', 16)).grid(row=7, column=3, pady=20)
        # Bottom back buttons
        customtkinter.CTkButton(self, text="< back to analysis menu", command=self.window3_menu,
                                font=('Helvetica', 16)).grid(row=6, column=0, pady=20, padx=20, sticky="W")
        customtkinter.CTkButton(self, text="<< back to start", command=self.window1_start,
                                font=('Helvetica', 16)).grid(row=7, column=0, pady=20, padx=20, sticky="W")

    def window4g_classify_and_embed_menu(self):
        self.clear_window()
        self.configure_window(5, 4)
        # Title
        customtkinter.CTkLabel(self, text="Classify and Embed Menu",
                               font=('Helvetica', 32, "bold")).grid(row=0, column=0, columnspan=4, pady=10)
        customtkinter.CTkLabel(self, text="Project " + self.config["project_name"],
                               font=('Helvetica', 20, "bold")).grid(row=1, column=0, columnspan=4, pady=10)
        customtkinter.CTkLabel(self, text="What analysis would you like to do?",
                               font=('Helvetica', 16)).grid(row=2, column=0, columnspan=4, pady=10)

        # Next menu options
        customtkinter.CTkButton(self, text="Embed with linear discriminant analysis",
                                command=self.window4g_lda_embed,
                                font=('Helvetica', 16)).grid(row=3, column=1, pady=20, sticky="NSEW")
        customtkinter.CTkButton(self, text="Train and evaluate a classifier",
                                command=self.window4g_train_eval_classifier,
                                font=('Helvetica', 16)).grid(row=3, column=2, pady=20, sticky="NSEW")
        customtkinter.CTkButton(self, text="Use a previously created classifier",
                                command=self.window4g_use_prev_classifier,
                                font=('Helvetica', 16)).grid(row=4, column=1, pady=20, sticky="NSEW")
        # Bottom back buttons
        customtkinter.CTkButton(self, text="< back to analysis menu", command=self.window3_menu,
                                font=('Helvetica', 16)).grid(row=5, column=1, pady=20, padx=20, sticky="W")
        customtkinter.CTkButton(self, text="<< back to start", command=self.window1_start,
                                font=('Helvetica', 16)).grid(row=6, column=1, pady=20, padx=20, sticky="W")

    def window4g_lda_embed(self):
        self.clear_window()
        self.configure_window(7, 4)
        # Title
        customtkinter.CTkLabel(self, text="Embed with Linear Discriminant Analysis",
                               font=('Helvetica', 32, "bold")).grid(row=0, column=0, columnspan=4, pady=10)
        customtkinter.CTkLabel(self, text="Project " + self.config["project_name"],
                               font=('Helvetica', 20, "bold")).grid(row=1, column=1, columnspan=2, pady=10)
        customtkinter.CTkLabel(self, text="Enter info about your analysis and plotting parameters.",
                               font=('Helvetica', 16)).grid(row=2, column=0, columnspan=4, pady=10)

        # Plot
        # Choose start and end time
        customtkinter.CTkLabel(self, text="Start time (seconds)",
                               font=('Helvetica', 16)).grid(row=3, column=0, pady=10, sticky="E")
        start = customtkinter.CTkEntry(self)
        start.grid(row=3, column=1, pady=10, sticky="W")
        customtkinter.CTkLabel(self, text="End time (seconds)",
                               font=('Helvetica', 16)).grid(row=3, column=2, pady=10, sticky="E")
        end = customtkinter.CTkEntry(self)
        end.grid(row=3, column=3, pady=10, sticky="W")
        # Choose color
        customtkinter.CTkLabel(self, text="Colormap for plot",
                               font=('Helvetica', 16)).grid(row=4, column=0, pady=10, sticky="E")
        color = customtkinter.CTkComboBox(self, values=["jet", "cividis", "viridis", "magma"])
        color.grid(row=4, column=1, pady=10, sticky="W")
        color.set("jet")
        customtkinter.CTkLabel(self, text="Bin size (seconds)",
                               font=('Helvetica', 16)).grid(row=4, column=2, pady=10, sticky="E")
        binsize = customtkinter.CTkEntry(self)
        binsize.grid(row=4, column=3, pady=10, sticky="W")
        customtkinter.CTkButton(self, text="Plot it!",
                                command=lambda: self.lda(int(start.get()), int(end.get()), int(binsize.get()), color.get()),
                                font=('Helvetica', 16)).grid(row=7, column=3, pady=20)
        # Bottom back buttons
        customtkinter.CTkButton(self, text="< back to analysis menu", command=self.window3_menu,
                                font=('Helvetica', 16)).grid(row=6, column=0, pady=20, padx=20, sticky="W")
        customtkinter.CTkButton(self, text="<< back to start", command=self.window1_start,
                                font=('Helvetica', 16)).grid(row=7, column=0, pady=20, padx=20, sticky="W")

    def window4g_train_eval_classifier(self):
        self.clear_window()
        self.configure_window(7, 4)
        # Title
        customtkinter.CTkLabel(self, text="Train and Evaluate a Classifier",
                               font=('Helvetica', 32, "bold")).grid(row=0, column=0, columnspan=4, pady=10)
        customtkinter.CTkLabel(self, text="Project " + self.config["project_name"],
                               font=('Helvetica', 20, "bold")).grid(row=1, column=1, columnspan=2, pady=10)
        customtkinter.CTkLabel(self, text="Enter info about the classifier you would like to train.",
                               font=('Helvetica', 16)).grid(row=2, column=0, columnspan=4, pady=10)

        # Plot
        # Choose start and end time
        customtkinter.CTkLabel(self, text="Start time (seconds)",
                               font=('Helvetica', 16)).grid(row=3, column=0, pady=10, sticky="E")
        start = customtkinter.CTkEntry(self)
        start.grid(row=3, column=1, pady=10, sticky="W")
        customtkinter.CTkLabel(self, text="End time (seconds)",
                               font=('Helvetica', 16)).grid(row=3, column=2, pady=10, sticky="E")
        end = customtkinter.CTkEntry(self)
        end.grid(row=3, column=3, pady=10, sticky="W")
        # Choose color
        customtkinter.CTkLabel(self, text="Colormap for plot",
                               font=('Helvetica', 16)).grid(row=4, column=0, pady=10, sticky="E")
        color = customtkinter.CTkComboBox(self, values=["jet", "cividis", "viridis", "magma"])
        color.grid(row=4, column=1, pady=10, sticky="W")
        color.set("jet")
        customtkinter.CTkLabel(self, text="Bin size (seconds)",
                               font=('Helvetica', 16)).grid(row=4, column=2, pady=10, sticky="E")
        binsize = customtkinter.CTkEntry(self)
        binsize.grid(row=4, column=3, pady=10, sticky="W")
        customtkinter.CTkButton(self, text="Plot it!",
                                command=lambda: self.lda(int(start.get()), int(end.get()), int(binsize.get()), color.get()),
                                font=('Helvetica', 16)).grid(row=7, column=3, pady=20)
        # Bottom back buttons
        customtkinter.CTkButton(self, text="< back to analysis menu", command=self.window3_menu,
                                font=('Helvetica', 16)).grid(row=6, column=0, pady=20, padx=20, sticky="W")
        customtkinter.CTkButton(self, text="<< back to start", command=self.window1_start,
                                font=('Helvetica', 16)).grid(row=7, column=0, pady=20, padx=20, sticky="W")
        # TODO:: update LDA classifier window to connect to real functions

    def window4g_use_prev_classifier(self):
        self.clear_window()
        self.configure_window(7, 4)
        # Title
        customtkinter.CTkLabel(self, text="Use a Previously Trained Classifier",
                               font=('Helvetica', 32, "bold")).grid(row=0, column=0, columnspan=4, pady=10)
        customtkinter.CTkLabel(self, text="Project " + self.config["project_name"],
                               font=('Helvetica', 20, "bold")).grid(row=1, column=1, columnspan=2, pady=10)
        customtkinter.CTkLabel(self, text="Enter info about your analysis and plotting parameters.",
                               font=('Helvetica', 16)).grid(row=2, column=0, columnspan=4, pady=10)

        # Plot
        # Choose start and end time
        customtkinter.CTkLabel(self, text="Start time (seconds)",
                               font=('Helvetica', 16)).grid(row=3, column=0, pady=10, sticky="E")
        start = customtkinter.CTkEntry(self)
        start.grid(row=3, column=1, pady=10, sticky="W")
        customtkinter.CTkLabel(self, text="End time (seconds)",
                               font=('Helvetica', 16)).grid(row=3, column=2, pady=10, sticky="E")
        end = customtkinter.CTkEntry(self)
        end.grid(row=3, column=3, pady=10, sticky="W")
        # Choose color
        customtkinter.CTkLabel(self, text="Colormap for plot",
                               font=('Helvetica', 16)).grid(row=4, column=0, pady=10, sticky="E")
        color = customtkinter.CTkComboBox(self, values=["jet", "cividis", "viridis", "magma"])
        color.grid(row=4, column=1, pady=10, sticky="W")
        color.set("jet")
        # Choose style for plot
        customtkinter.CTkLabel(self, text="What should the \nplot style be?",
                               font=('Helvetica', 16)).grid(row=4, column=2, rowspan=2, pady=10, sticky="E")
        style = customtkinter.StringVar(value="scatter")
        style_options = ["Bar with scattered individual points",
                         "Bar with standard error of the mean",
                         "Points with standard error of the mean"]
        style_vars = ["bar_scatter", "bar_error", "points"]
        for s in range(len(style_vars)):
            radio_btn = customtkinter.CTkRadioButton(self, text=style_options[s],
                                                     variable=style, value=style_vars[s],
                                                     font=('Helvetica', 16))
            radio_btn.grid(row=4 + s, column=3, sticky="W")
        customtkinter.CTkButton(self, text="Plot it!",
                                command=lambda: self.plot_usage(int(start.get()), int(end.get()),
                                                                style.get(), color.get(), subgroup_option),
                                font=('Helvetica', 16)).grid(row=7, column=3, pady=20)
        # Bottom back buttons
        customtkinter.CTkButton(self, text="< back to analysis menu", command=self.window3_menu,
                                font=('Helvetica', 16)).grid(row=6, column=0, pady=20, padx=20, sticky="W")
        customtkinter.CTkButton(self, text="<< back to start", command=self.window1_start,
                                font=('Helvetica', 16)).grid(row=7, column=0, pady=20, padx=20, sticky="W")
        # TODO:: update NLP classifier window to connect to real functions

    def window4h_pose_vs_BORIS(self):
        self.clear_window()
        self.configure_window(7, 4)
        # Title
        customtkinter.CTkLabel(self, text="Compare Pose Data to Manually Scored Behaviors",
                               font=('Helvetica', 32, "bold")).grid(row=0, column=0, columnspan=4, pady=10)
        customtkinter.CTkLabel(self, text="Project " + self.config["project_name"],
                               font=('Helvetica', 20, "bold")).grid(row=1, column=1, columnspan=2, pady=10)
        customtkinter.CTkLabel(self, text="Get a matrix showing overlap between pose modules and behaviors manually scored in BORIS.",
                               font=('Helvetica', 16)).grid(row=2, column=0, columnspan=4, pady=10)

        # Enter data directory
        customtkinter.CTkLabel(self, text="Path to BORIS data",
                               font=('Helvetica', 16)).grid(row=3, column=0, pady=10, sticky="E")
        data_path_entry = customtkinter.CTkEntry(self)
        data_path_entry.grid(row=3, column=1, padx=5, pady=5, sticky="W")
        browse_button = customtkinter.CTkButton(self, text="Browse",
                                                command=lambda: self.window_browse(data_path_entry, type="directory"))
        browse_button.grid(row=3, column=2, pady=5, sticky="W")

        customtkinter.CTkButton(self, text="Compare!",
                                command=lambda: self.plot_usage(int(start.get()), int(end.get()),
                                                                style.get(), color.get(), subgroup_option),
                                font=('Helvetica', 16)).grid(row=7, column=3, pady=20)
        # Bottom back buttons
        customtkinter.CTkButton(self, text="< back to analysis menu", command=self.window3_menu,
                                font=('Helvetica', 16)).grid(row=6, column=0, pady=20, padx=20, sticky="W")
        customtkinter.CTkButton(self, text="<< back to start", command=self.window1_start,
                                font=('Helvetica', 16)).grid(row=7, column=0, pady=20, padx=20, sticky="W")
        # TODO:: update NLP classifier window to connect to real functions


    def plot_usage(self, start, end, style, color, subgroup_option):
        if subgroup_option=="no_subgroups":
            labels_df, n_modules = analysis.label_counter_nosubgroups(self.config, start, end)
            fig = plot.plot_module_usage(self.config, labels_df, start, end, int(self.config["fps"]), style=style, cmap=color)
            self.plots_generated = self.plots_generated + 1
            self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
            self.plot_window.mainloop()
        elif subgroup_option=="subgroups":
            labels_df, n_modules = analysis.label_counter_subgroups(self.config, start, end)
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
        if subgroup_option=="no_subgroups":
            labels_df, n_modules = analysis.label_counter_nosubgroups(self.config, start, end)
            fig = plot.plot_module_usage(self.config, labels_df, start, end, int(self.config["fps"]), style=style, cmap=color)
            self.plots_generated = self.plots_generated + 1
            self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
            self.plot_window.mainloop()
        elif subgroup_option=="subgroups":
            labels_df, n_modules = analysis.label_counter_subgroups(self.config, start, end)
            fig = plot.plot_module_usage_subgroups(self.config, labels_df, start, end, int(self.config["fps"]), style=style, cmap=color)
            self.plots_generated = self.plots_generated + 1
            self.plot_window = PlotWindow(fig = fig, plot_number = self.plots_generated, master = self)
            self.plot_window.mainloop()

    def lda(self, start, end, binsize, cmap):
        labels_df, n_modules = analysis.label_counter_subgroups(self.config, start, end)
        lda, lda_embeddings, group_labels, nbins = analysis.lda_labels_timebins(self.config, labels_df, binsize)
        fig = plot.plot_lda(self.config, lda, lda_embeddings, group_labels, nbins, binsize, cmap=cmap)
        self.plots_generated = self.plots_generated + 1
        self.plot_window = PlotWindow(fig=fig, plot_number=self.plots_generated, master=self)
        self.plot_window.mainloop()

class PlotWindow(customtkinter.CTk):
    def __init__(self, fig, plot_number, master=None):
        super().__init__()
        self.title('MARIPOSA - Plot ' + str(plot_number))

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
