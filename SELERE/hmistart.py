import tkinter as tk
from classes import Exoskeleton
from kneeMotor.motorCAN import start_can, tkinter_loop, comm_can_transmit_eid, write_log
from kneeMotor.motorControl import current, speed, current_brake
from ankleMotor.motorControl import velocity, torque, setVelocity, setTorque
from PIL import Image, ImageTk

# Initialize main window
root = tk.Tk()
root.title("Touch Screen Interface")
root.geometry("1024x600")
root.configure(bg="lightgray")  
exo = Exoskeleton()

# Variables to track selected mode, joint, tab, and DOC button
selected_mode = tk.StringVar(value=exo.currentMode.name)
selected_joint = tk.StringVar(value=exo.currentJoint.name)
selected_tab = tk.StringVar(value="Edit")
selected_doc_button = tk.StringVar(value="Max Intensity")

# Create image frame and label at global level
image_frame = tk.Frame(root)
image_label = tk.Label(image_frame)
image_label.pack(fill="both", expand=True)
current_photo = None  # This will keep a persistent reference to the image

def display_image(image_path):
    global current_photo
    try:
        img = Image.open(image_path)
        img = img.resize((972, 720), Image.LANCZOS)
        current_photo = ImageTk.PhotoImage(img)
        image_label.config(image=current_photo)
        return current_photo
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Dictionary to store settings for each mode and joint
settings = {
    "Full": {
        "Left Knee": {"max_intensity": exo.leftKnee.maxSpd, "min_intensity": exo.leftKnee.minSpd, "max_height": exo.leftKnee.maxHeight, "min_height": exo.leftKnee.minHeight, "current_intensity": 0, "current_height": 0},
        "Left Ankle": {"max_intensity": exo.leftAnkle.maxSpd, "min_intensity": exo.leftAnkle.minSpd, "max_height": exo.leftAnkle.maxHeight, "min_height": exo.leftAnkle.minHeight, "current_intensity": 0, "current_height": 0},
        "Right Knee": {"max_intensity": exo.rightKnee.maxSpd, "min_intensity": exo.rightKnee.minSpd, "max_height": exo.rightKnee.maxHeight, "min_height": exo.rightKnee.minHeight, "current_intensity": 0, "current_height": 0},
        "Right Ankle": {"max_intensity": exo.rightAnkle.maxSpd, "min_intensity": exo.rightAnkle.minSpd, "max_height": exo.rightAnkle.maxHeight, "min_height": exo.rightAnkle.minHeight, "current_intensity": 0, "current_height": 0},
    },
    "Partial": {
        "Left Knee": {"max_intensity": exo.leftKnee.maxAssist, "min_intensity": exo.leftKnee.minAssist, "max_height": exo.leftKnee.maxHeight, "min_height": exo.leftKnee.minHeight, "current_intensity": 0, "current_height": 0},
        "Left Ankle": {"max_intensity": exo.leftAnkle.maxAssist, "min_intensity": exo.leftAnkle.minAssist, "max_height": exo.leftAnkle.maxHeight, "min_height": exo.leftAnkle.minHeight, "current_intensity": 0, "current_height": 0},
        "Right Knee": {"max_intensity": exo.rightKnee.maxAssist, "min_intensity": exo.rightKnee.minAssist, "max_height": exo.rightKnee.maxHeight, "min_height": exo.rightKnee.minHeight, "current_intensity": 0, "current_height": 0},
        "Right Ankle": {"max_intensity": exo.rightAnkle.maxAssist, "min_intensity": exo.rightAnkle.minAssist, "max_height": exo.rightAnkle.maxHeight, "min_height": exo.rightAnkle.minHeight, "current_intensity": 0, "current_height": 0},
    },
    "Resistance": {
        "Left Knee": {"max_intensity": exo.leftKnee.maxResist, "min_intensity": exo.leftKnee.minResist, "max_height": exo.leftKnee.maxHeight, "min_height": exo.leftKnee.minHeight, "current_intensity": 0, "current_height": 0},
        "Left Ankle": {"max_intensity": exo.leftAnkle.maxResist, "min_intensity": exo.leftAnkle.minResist, "max_height": exo.leftAnkle.maxHeight, "min_height": exo.leftAnkle.minHeight, "current_intensity": 0, "current_height": 0},
        "Right Knee": {"max_intensity": exo.rightKnee.maxResist, "min_intensity": exo.rightKnee.minResist, "max_height": exo.rightKnee.maxHeight, "min_height": exo.rightKnee.minHeight, "current_intensity": 0, "current_height": 0},
        "Right Ankle": {"max_intensity": exo.rightAnkle.maxResist, "min_intensity": exo.rightAnkle.minResist, "max_height": exo.rightAnkle.maxHeight, "min_height": exo.rightAnkle.minHeight, "current_intensity": 0, "current_height": 0},
    },
}

# Function to set the mode
def set_mode(mode):
    if exo.currentMode.number != mode.number:  # Only change if it's 
        selected_mode.set(mode.name)
        update_button_colors()
        if mode in exo.modes:
            print(f"Mode set to: {mode.name}")
            exo.currentMode = exo.modes[mode.number-1]
            if "Ankle" in exo.currentJoint.name:
                if mode.name == "Full":
                    setVelocity(exo.currentJoint.id)
                else:
                    setTorque(exo.currentJoint.id) 
        else:
            print(f"Mode {mode.name} does not exist")
        
        update_sliders()
        update_button_labels()

def switch_tab(tab):
    if selected_tab.get() != tab:
        selected_tab.set(tab)
        update_button_colors()
        update_visibility()
        print(f"Switched to {tab} tab")

def control_joint(joint):
    if selected_joint.get() != joint.name:
        selected_joint.set(joint.name)
        update_button_colors()
        print(f"Controlling {joint.name}")
        if joint in exo.joints:
            exo.currentJoint = joint
        else:
            print(f"Joint {joint.name} does not exist")

        update_sliders()
        update_button_labels()

def start_button_pressed(*args):
    print("Start button clicked")

    exo.currentState = exo.states[1]
    if exo.currentState == "started":
        run()

def run():
    if exo.currentState == "started":
        if exo.currentMode.name == "Full":
            if exo.currentJoint.name == "Right Knee":
                position = -exo.currentJoint.getPosition()
            else:
                position = exo.currentJoint.getPosition()
            
            desSpd = exo.currentJoint.getDesSpeed()
            if position > exo.currentJoint.desHeight:
                exo.currentJoint.currentDirection = -1 * exo.currentJoint.initialDirection
            if position < exo.currentJoint.minHeight:
                exo.currentJoint.currentDirection = exo.currentJoint.initialDirection
            if "Knee" in exo.currentJoint.name:
                velocity(100, 0, 0) #Ankle Stays Stays Stiff During Partial
                comm_can_transmit_eid(*speed(exo.currentJoint.canbus, desSpd*exo.currentJoint.currentDirection, controller_id=exo.currentJoint.id))
            if "Ankle" in exo.currentJoint.name:
                velocity(exo.currentJoint.id, exo.currentJoint.index, desSpd*exo.currentJoint.currentDirection)
                comm_can_transmit_eid(*speed(exo.leftKnee.canbus, 0, controller_id=exo.currentJoint.index)) #Knee will not move while ankle is active

        if exo.currentMode.name == "Partial" or exo.currentMode.name == "Resistance":
            desCurrentMilliamps = exo.currentJoint.getDesCurrent()
            desCurrentAmps = desCurrentMilliamps / 1000
            if exo.currentMode.name == "Partial":
                if "Knee" in exo.currentJoint.name:
                    print("position:" + str(exo.rightKnee.getPosition())+ "secondpos" + str(exo.rightKnee.position)+"  " + str(exo.rightKnee.maxHeight)+"  " + str(exo.currentJoint.initialDirection))
                    comm_can_transmit_eid(*current(exo.currentJoint.canbus, desCurrentAmps*exo.currentJoint.initialDirection, controller_id=exo.currentJoint.id))
                    velocity(100, 0, 0) #Ankle Stays Stays Stiff During Partial
                if "Ankle" in exo.currentJoint.name:
                    print(desCurrentMilliamps*exo.currentJoint.initialDirection)
                    torque(exo.currentJoint.id, exo.currentJoint.index, desCurrentMilliamps*exo.currentJoint.initialDirection)
            if exo.currentMode.name == "Resistance":
                if "Knee" in exo.currentJoint.name:
                    comm_can_transmit_eid(*current(exo.currentJoint.canbus, -desCurrentAmps*exo.currentJoint.initialDirection, controller_id=exo.currentJoint.id))
                if "Ankle" in exo.currentJoint.name:
                    torque(exo.currentJoint.id, exo.currentJoint.index, -desCurrentAmps*exo.currentJoint.initialDirection)
            root.after(100, run) 
            print("position:" + str(exo.rightKnee.getPosition())+ "secondpos" + str(exo.rightKnee.position)+"  " + str(exo.rightKnee.maxHeight)+"  " + str(exo.currentJoint.initialDirection))
               
        
    else:
        if "Knee" in exo.currentJoint.name:
            if exo.currentMode.name == "Partial" or exo.currentMode.name == "Resistance":
                comm_can_transmit_eid(*current_brake(exo.currentJoint.canbus, 6, controller_id=exo.currentJoint.id))
                velocity(100, 0, 0) #Ankle Stays Stiff During Descent
            if exo.currentMode.name == "Full":
                comm_can_transmit_eid(*speed(exo.currentJoint.canbus, 0, controller_id=exo.currentJoint.id))
        if "Ankle" in exo.currentJoint.name:
            if exo.currentMode.name == "Partial" or exo.currentMode.name == "Resistance":
                torque(exo.currentJoint.id, exo.currentJoint.index, 0)
            if exo.currentMode.name == "Full":
                velocity(exo.currentJoint.id, exo.currentJoint.index, 0)
                
            comm_can_transmit_eid(*speed(exo.leftKnee.canbus, 0, controller_id=exo.currentJoint.index)) #Knee will not move while ankle is active
    if exo.currentMode.name == "Full":
        root.after(100, run)

def start_button_released(*args):
    print("Start button released")
    write_log(f"LeftKnee Position:{exo.leftKnee.getPosition()}")
    write_log(f"RightKnee Position:{exo.rightKnee.getPosition()}")
    # print(exo.currentJoint.desHeight,',',exo.currentJoint.currentDirection)

    print("position:" + str(exo.rightKnee.getPosition()))
    write_log(f"LeftKnee Position:{exo.leftKnee.getPosition()}")
    write_log(f"RightKnee Position:{exo.rightKnee.getPosition()}")
    exo.currentState = exo.states[0]

    if exo.currentState == "stopped":
        run()

# Create frames for different sections
slider_frame = tk.Frame(root)
slider_frame.place(relx=0.05, rely=0.3, relwidth=0.25, relheight=0.7)

mode_frame = tk.Frame(root)
mode_frame.place(relx=0.01, rely=0.05, relwidth=0.05, relheight=0.1)


#status_frame = tk.Frame(root)
#status_frame.place(relx=0.05, rely=0.2, relwidth=0.25, relheight=0.08)

tab_frame = tk.Frame(root)
tab_frame.place(relx=0.425, rely=0.05, relwidth=0.5, relheight=0.2)

joint_frame = tk.Frame(root)
joint_frame.place(relx=-.4, rely=0.3, relwidth=0.55, relheight=0.55)

# Tank-style sliders
slider_heights = (0, 650)
slider_widths = (0, 100)

def update_sliders():
    mode = selected_mode.get()
    joint = selected_joint.get()
    intensity_value = settings[mode][joint]["current_intensity"]
    height_value = settings[mode][joint]["current_height"]
    intensity_slider.set(intensity_value)
    height_slider.set(height_value)

def update_intensity(val):
    mode = selected_mode.get()
    joint = selected_joint.get()
    settings[mode][joint]["current_intensity"] = int(float(val))
    intensity_tank.coords(intensity_fill, slider_widths[0], slider_heights[1] - (slider_heights[1] * (float(val) / 100)), slider_widths[1], slider_heights[1])
    if mode == "Full":
        exo.currentJoint.desSpd = (((int(float(val)) / 100) * (settings[mode][joint]["max_intensity"]-settings[mode][joint]["min_intensity"])) + settings[mode][joint]["min_intensity"])
    else:
        exo.currentJoint.desCurrent = (((int(float(val)) / 100) * (settings[mode][joint]["max_intensity"]-settings[mode][joint]["min_intensity"])) + settings[mode][joint]["min_intensity"])


def update_height(val):
    mode = selected_mode.get()
    joint = selected_joint.get()
    settings[mode][joint]["current_height"] = int(float(val))
    height_tank.coords(height_fill, slider_widths[0], slider_heights[1], slider_widths[1], slider_heights[1] - (slider_heights[1] * (float(val) / 100)))
    exo.currentJoint.desHeight = (((int(float(val)) / 100) * (settings[mode][joint]["max_height"]-settings[mode][joint]["min_height"])) + settings[mode][joint]["min_height"])

# Intensity tank
intensity_tank = tk.Canvas(slider_frame, width=slider_widths[1], height=slider_heights[1], bg="lightgray")
intensity_fill = intensity_tank.create_rectangle(slider_widths[0], slider_heights[0], slider_widths[1], slider_heights[0], fill="green")
intensity_tank.grid(row=1, column=0, padx=5, pady=5)

# Height tank
height_tank = tk.Canvas(slider_frame, width=slider_widths[1], height=slider_heights[1], bg="lightgray")
height_fill = height_tank.create_rectangle(slider_widths[0], slider_heights[0], slider_widths[1], slider_heights[0], fill="green")
height_tank.grid(row=1, column=2, padx=5, pady=5)

def create_text_box(parent, text, x, y, width, height, font_size):
    text_box = tk.Entry(
        parent,
        font=("Arial", font_size),
        width=width,
        justify="center",
        bg="lightgray",
        fg="black",
        relief=tk.FLAT,
        bd=2
    )
    text_box.insert(0, text)
    text_box.place(x=x, y=y, width=width * 10, height=height * 20)
    return text_box

intensity_text_box = create_text_box(
    parent=root,
    text="Intensity",
    x=50,
    y=100,
    width=23,
    height=3,
    font_size=35
)

height_text_box = create_text_box(
    parent=root,
    text="Height",
    x=250,
    y=150,
    width=20,
    height=3,
    font_size=35
)

# Intensity slider
intensity_slider = tk.Scale(
    slider_frame, 
    from_=100, 
    to=0, 
    orient="vertical", 
    command=update_intensity, 
    length=400,
    width=80,
    sliderlength=80,
    troughcolor="lightgray",
    bg="lightgray",
    font=("Arial", 28)
)
intensity_slider.set(0)
intensity_slider.grid(row=1, column=1, padx=5, pady=5, sticky="ns")

# Height slider
height_slider = tk.Scale(
    slider_frame, 
    from_=100, 
    to=0, 
    orient="vertical", 
    command=update_height, 
    length=400,
    width=80,
    sliderlength=80,
    troughcolor="lightgray",
    bg="lightgray",
    font=("Arial", 28)
)
height_slider.set(0)
height_slider.grid(row=1, column=3, padx=5, pady=5, sticky="ns")

for i in range(4):
    slider_frame.grid_columnconfigure(i, weight=1)
slider_frame.grid_rowconfigure(1, weight=1)

# Mode buttons
mode_buttons = []
modes = [exo.modeFA, exo.modePA, exo.modePR]
for idx, mode in enumerate(modes):
    mode_button = tk.Button(
        mode_frame, 
        text=mode.name, 
        command=lambda m=mode: set_mode(m), 
        height=10,
        width=1,
        font=("Arial", 36),
        activebackground="green"
    )
    mode_button.pack(side="left", padx=5, pady=5, expand=True, fill="both")
    mode_buttons.append(mode_button)

for i in range(len(modes)):
    mode_frame.grid_columnconfigure(i, weight=1)
mode_frame.grid_rowconfigure(0, weight=1)

# Tab buttons
tab_buttons = []
tabs = ["User", "Edit", "Analytics", "DOC"]
for tab in tabs:
    tab_button = tk.Button(
        tab_frame, 
        text=tab, 
        command=lambda t=tab: switch_tab(t), 
        font=("Arial", 28),
        height=1, 
        width=10, 
        activebackground="green"
    )
    tab_button.pack(side="left", padx=5, pady=5, expand=True, fill="both")
    tab_buttons.append(tab_button)

# Joint control buttons
joint_buttons = []
joints = [exo.leftKnee, exo.rightKnee, exo.leftAnkle, exo.rightAnkle]
row, col = 0, 0
for joint in joints:
    joint_button = tk.Button(joint_frame, text=joint.name, command=lambda j=joint: control_joint(j), height=6, width=20, font=("Arial", 50), activebackground="green")
    joint_button.grid(row=row, column=col, padx=20, pady=20, sticky="nsew")
    joint_buttons.append(joint_button)
    col += 1
    if col > 1:
        col = 0
        row += 1

# DOC tab buttons frame
doc_button_frame = tk.Frame(root)

max_intensity_var = tk.StringVar(value="Max Intensity\n100")
min_intensity_var = tk.StringVar(value="Min Intensity\n0")
max_height_var = tk.StringVar(value="Max Height\n100")
min_height_var = tk.StringVar(value="Min Height\n0")

max_intensity_button = tk.Button(doc_button_frame, textvariable=max_intensity_var, height=4, width=20, font=("Arial", 24), activebackground="green")
min_intensity_button = tk.Button(doc_button_frame, textvariable=min_intensity_var, height=4, width=20, font=("Arial", 24), activebackground="green")
max_height_button = tk.Button(doc_button_frame, textvariable=max_height_var, height=4, width=20, font=("Arial", 24), activebackground="green")
min_height_button = tk.Button(doc_button_frame, textvariable=min_height_var, height=4, width=20, font=("Arial", 24), activebackground="green")

max_intensity_button.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
min_intensity_button.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
max_height_button.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
min_height_button.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

doc_button_frame.grid_rowconfigure(0, weight=1)
doc_button_frame.grid_rowconfigure(1, weight=1)
doc_button_frame.grid_columnconfigure(0, weight=1)
doc_button_frame.grid_columnconfigure(1, weight=1)

def update_button_labels():
    mode = selected_mode.get()
    joint = selected_joint.get()
    max_intensity_var.set(f"Max Intensity\n{settings[mode][joint]['max_intensity']}")
    min_intensity_var.set(f"Min Intensity\n{settings[mode][joint]['min_intensity']}")
    max_height_var.set(f"Max Height\n{settings[mode][joint]['max_height']}")
    min_height_var.set(f"Min Height\n{settings[mode][joint]['min_height']}")

def select_doc_button(label):
    selected_doc_button.set(label)
    update_doc_button_colors()

def on_button_click(label):
    select_doc_button(label)

max_intensity_button.config(command=lambda: on_button_click("Max Intensity"))
min_intensity_button.config(command=lambda: on_button_click("Min Intensity"))
max_height_button.config(command=lambda: on_button_click("Max Height"))
min_height_button.config(command=lambda: on_button_click("Min Height"))

def update_doc_button_colors():
    buttons = [
        (max_intensity_button, "Max Intensity"),
        (min_intensity_button, "Min Intensity"),
        (max_height_button, "Max Height"),
        (min_height_button, "Min Height")
    ]
    
    for button, label in buttons:
        button.config(bg="green" if selected_doc_button.get() == label else root.cget("bg"))

update_button_labels()

for i in range(2):
    joint_frame.grid_rowconfigure(i, weight=1)
for i in range(2):
    joint_frame.grid_columnconfigure(i, weight=1)

def update_button_colors():
    for button, mode in zip(mode_buttons, modes):
        button.config(bg="green" if mode.name == selected_mode.get() else root.cget("bg"))

    for button, joint in zip(joint_buttons, joints):
        button.config(bg="green" if joint.name == selected_joint.get() else root.cget("bg"))

    for button, tab in zip(tab_buttons, tabs):
        button.config(bg="green" if tab == selected_tab.get() else root.cget("bg"))

update_button_colors()

# [Previous imports and code remain the same until the update_visibility function]

def update_visibility():
    global button_tank_frame, start_button, blank_tank, lock_screen_frame
    
    try:
        button_tank_frame.place_forget()
    except NameError:
        pass
    
    mode_frame.place(relx=0.01, rely=0.05, relwidth=0.40, relheight=0.15)
    #status_frame.place(relx=0.05, rely=0.2, relwidth=0.25, relheight=0.08)

    if selected_tab.get() == "Edit":
        slider_frame.place(relx=0.05, rely=0.3, relwidth=0.25, relheight=0.7)
        joint_frame.place(relx=0.4, rely=0.3, relwidth=0.55, relheight=0.55)
        doc_button_frame.place_forget()
        new_button_frame.place_forget()
        intensity_text_box.place(x=100, y=230, width=230, height=60)
        height_text_box.place(x=350, y=230, width=200, height=60)
        image_frame.place_forget()
        # Remove lock screen if it exists
        try:
            lock_screen_frame.place_forget()
        except NameError:
            pass
        root.update_idletasks()
        root.tk.call("raise", intensity_tank._w)
        root.tk.call("raise", height_tank._w)

    elif selected_tab.get() == "DOC":
        # Create lock screen frame if it doesn't exist
        if 'lock_screen_frame' not in globals():
            create_lock_screen()
        lock_screen_frame.place(relx=0.2, rely=0.3, relwidth=0.6, relheight=0.4)
        # Hide the normal DOC content initially
        joint_frame.place_forget()
        slider_frame.place_forget()
        doc_button_frame.place_forget()
        new_button_frame.place_forget()
        intensity_text_box.place_forget()
        height_text_box.place_forget()
        image_frame.place_forget()
        
    elif selected_tab.get() == "Analytics":
        joint_frame.place(relx=0.4, rely=0.3, relwidth=0.55, relheight=0.65)
        slider_frame.place_forget()
        doc_button_frame.place_forget()
        new_button_frame.place_forget()
        intensity_text_box.place_forget()
        height_text_box.place_forget()
        joint_frame.place_forget()
        # Remove lock screen if it exists
        try:
            lock_screen_frame.place_forget()
        except NameError:
            pass
        
        display_image("Assets/3stepCurrent.PNG")
        image_frame.place(relx=0.25, rely=0.26, relwidth=0.5125, relheight=0.725)

    else:  # User tab
        joint_frame.place(relx=0.4, rely=0.3, relwidth=0.55, relheight=0.55)
        slider_frame.place_forget()
        doc_button_frame.place_forget()
        new_button_frame.place_forget()
        intensity_text_box.place_forget()
        height_text_box.place_forget()
        image_frame.place_forget()
        # Remove lock screen if it exists
        try:
            lock_screen_frame.place_forget()
        except NameError:
            pass
        
        button_tank_frame = tk.Frame(root)
        button_tank_frame.place(x=25, y=350, width=700, height=560)
        start_button = tk.Button(button_tank_frame, text="Start", height=6, width=10, font=("Arial", 50))
        start_button.place(x=0, y=0, width=500, height=560)
        start_button.bind("<ButtonPress>", start_button_pressed)
        start_button.bind("<ButtonRelease>", start_button_released)  
        blank_tank = tk.Canvas(button_tank_frame, bg="lightgray")
        blank_tank.place(x=550, y=0, width=100, height=560)

def create_lock_screen():
    global lock_screen_frame, lock_label, back_button, unlock_button
    
    # Create the lock screen frame
    lock_screen_frame = tk.Frame(root, bg="lightgray", bd=5, relief=tk.RAISED)
    
    # Add "Locked" label
    lock_label = tk.Label(
        lock_screen_frame, 
        text="LOCKED", 
        font=("Arial", 48, "bold"), 
        bg="lightgray", 
        fg="red"
    )
    lock_label.pack(pady=20)
    
    # Create button frame
    button_frame = tk.Frame(lock_screen_frame, bg="lightgray")
    button_frame.pack(pady=20)
    
    # Add "Go Back" button
    back_button = tk.Button(
        button_frame, 
        text="Go Back", 
        font=("Arial", 24),
        width=10,
        command=go_back_from_lock,
        bg="lightblue"
    )
    back_button.pack(side=tk.LEFT, padx=20)
    
    # Add "Unlock" button
    unlock_button = tk.Button(
        button_frame, 
        text="Unlock", 
        font=("Arial", 24),
        width=10,
        command=unlock_doc_screen,
        bg="lightgreen"
    )
    unlock_button.pack(side=tk.LEFT, padx=20)

def go_back_from_lock():
    # Hide lock screen and go back to previous tab
    lock_screen_frame.place_forget()
    # Find the current tab index and go to the previous one
    current_tab_index = tabs.index(selected_tab.get())
    previous_tab = tabs[max(0, current_tab_index - 1)]
    switch_tab(previous_tab)

def unlock_doc_screen():
    # Hide the lock screen
    lock_screen_frame.place_forget()
    
    # Show the actual DOC content
    joint_frame.place(relx=0.4, rely=0.3, relwidth=0.55, relheight=0.55)
    doc_button_frame.place(relx=0.025, rely=0.225, relwidth=0.35, relheight=0.35)
    new_button_frame.place(relx=0.025, rely=0.625, relwidth=0.35, relheight=0.35)

# [Rest of the code remains the same]

# Create a frame for the new buttons
new_button_frame = tk.Frame(root)

def update_value(delta):
    mode = selected_mode.get()
    joint = selected_joint.get()
    selected = selected_doc_button.get()
    if selected == "Max Intensity":
        settings[mode][joint]["max_intensity"] += delta
    elif selected == "Min Intensity":
        settings[mode][joint]["min_intensity"] += delta
    elif selected == "Max Height":
        settings[mode][joint]["max_height"] += delta
    elif selected == "Min Height":
        settings[mode][joint]["min_height"] += delta
    update_button_labels()
    update_intensity()
    update_height()

buttons = [
    ("+1", lambda: update_value(1)),
    ("+5", lambda: update_value(5)),
    ("+15", lambda: update_value(15)),
    ("-1", lambda: update_value(-1)),
    ("-5", lambda: update_value(-5)),
    ("-15", lambda: update_value(-15))
]

for idx, (text, command) in enumerate(buttons):
    button = tk.Button(new_button_frame, text=text, command=command, height=4, width=10, font=("Arial", 24), activebackground="green")
    button.grid(row=idx // 3, column=idx % 3, padx=5, pady=5)

update_button_colors()
update_visibility()

# Start the main loop
kneeMotors = [exo.leftKnee, exo.rightKnee]
ankleMotors = [exo.leftAnkle, exo.rightAnkle]
components = [kneeMotors, ankleMotors]
start_can(components, tkinter_loop, root.mainloop)                                                                                                                                                                             