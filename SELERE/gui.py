import tkinter as tk
import math

from classes import Exoskeleton
import can
import struct
import subprocess

from kneeMotor.motorCAN import start_can, tkinter_loop, comm_can_transmit_eid, write_log, can_handler_thread
from kneeMotor.motorControl import current, speed, current_brake, position_speed_acceleration, move_to_desired_angle, set_origin
from ankleMotor.motorControl import velocity, torque, setVelocity, setTorque, setupCandle, position, calibrate
from PIL import Image, ImageTk

import os
from datetime import datetime
import time
import threading
from kneeMotor.motorControl import set_origin
from ankleMotor.motorControl import setupCandle

CHANNEL = 'can0'
BITRATE = 1000000
BUSTYPE = 'socketcan'

# ── Config ────────────────────────────────────────────────────────────────────
BG        = "#1a1a2e"
PANEL_BG  = "#16213e"
ACCENT1   = "#e94560"   # ankle  – red/pink
ACCENT2   = "#0f3460"   # knee   – deep blue
ACCENT_KNEE = "#4cc9f0" # knee   – cyan
TRACK_COL = "#2a2a4a"
TICK_COL  = "#3a3a5a"
TEXT_COL  = "#e0e0f0"
VAL_COL   = "#ffffff"

RADIUS    = 100
THICKNESS = 18
CX = CY   = 130
SIZE      = 260

# ── Braking config ────────────────────────────────────────────────────────────
BRAKE_MAX   = 3         # max braking intensity (adjust to match motor units)
ANGLE_POLL_MS = 200     # how often (ms) to refresh live angle readout

global KNEE_ANGLE
global EXO
global sendNegative

KNEE_ANGLE   = None
EXO          = None
sendNegative = False


# ── CAN setup ─────────────────────────────────────────────────────────────────

def setupCAN(devices):
    write_log(f"Setting bitrate for {CHANNEL}...")
    os.system(f'sudo ip link set {CHANNEL} type can bitrate ' + str(BITRATE))
    write_log(f"Bringing {CHANNEL} interface up...")
    os.system(f'sudo ip link set {CHANNEL} up')
    write_log(f"{CHANNEL} interface is up.")
    write_log(f'checking for errors in can line')
    result = subprocess.run(['ip', '-details', 'link', 'show', CHANNEL],
                            capture_output=True, text=True)
    write_log(f"{CHANNEL} status:\n{result.stdout}")

    print("Initializing CAN bus...")
    can0 = can.interface.Bus(interface='socketcan', channel=CHANNEL)
    print("CAN bus initialized successfully.")

    receiver_thread = threading.Thread(
        target=can_handler_thread, args=(can0, devices), daemon=True)
    receiver_thread.start()

    return can0


# ── Helper ────────────────────────────────────────────────────────────────────

def angle_to_pos(angle_deg, r):
    rad = math.radians(angle_deg - 90)
    return CX + r * math.cos(rad), CY + r * math.sin(rad)


def pos_to_angle(x, y):
    dx, dy = x - CX, y - CY
    deg = math.degrees(math.atan2(dy, dx)) + 90
    return deg % 360


# ── Circular Slider Widget ────────────────────────────────────────────────────

class CircularSlider(tk.Canvas):
    def __init__(self, parent, label, color, initial=45, callback=None, **kw):
        super().__init__(parent, width=SIZE, height=SIZE,
                         bg=PANEL_BG, highlightthickness=0, **kw)
        self.label     = label
        self.color     = color
        self.angle     = initial
        self.callback  = callback
        self._dragging = False

        self._draw()
        self.bind("<ButtonPress-1>",   self._on_press)
        self.bind("<B1-Motion>",       self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _draw(self):
        self.delete("all")
        r   = RADIUS
        pad = CX - r

        for t in range(0, 360, 10):
            inner = r - (10 if t % 30 == 0 else 5)
            x1, y1 = angle_to_pos(t, inner)
            x2, y2 = angle_to_pos(t, r - 2)
            self.create_line(x1, y1, x2, y2,
                             fill=TICK_COL, width=(2 if t % 30 == 0 else 1))

        self.create_arc(pad, pad, CX + r, CY + r,
                        start=0, extent=359.9,
                        style=tk.ARC, outline=TRACK_COL, width=THICKNESS)

        self.create_arc(pad, pad, CX + r, CY + r,
                        start=90, extent=-self.angle,
                        style=tk.ARC, outline=self.color, width=THICKNESS)

        for deg, txt in [(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")]:
            lx, ly = angle_to_pos(deg, r + 20)
            self.create_text(lx, ly, text=txt, fill=TEXT_COL, font=("Helvetica", 8))

        self.create_text(CX, CY - 12, text=self.label,
                         fill=TEXT_COL, font=("Helvetica", 11, "bold"))
        self.create_text(CX, CY + 14, text=f"{self.angle:.1f}°",
                         fill=VAL_COL, font=("Helvetica", 22, "bold"))

        hx, hy = angle_to_pos(self.angle, r)
        kr = 11
        self.create_oval(hx - kr, hy - kr, hx + kr, hy + kr,
                         fill=self.color, outline=VAL_COL, width=2, tags="handle")

    def _on_press(self, e):
        hx, hy = angle_to_pos(self.angle, RADIUS)
        if math.hypot(e.x - hx, e.y - hy) < 22:
            self._dragging = True

    def _on_drag(self, e):
        if self._dragging:
            self.angle = round(pos_to_angle(e.x, e.y), 1)
            self._draw()
            if self.callback:
                self.callback(self.label, self.angle)

    def _on_release(self, _):
        self._dragging = False

    def get_angle(self):
        return self.angle


# ── Resistive Brake Panel ─────────────────────────────────────────────────────

class BrakePanel(tk.Frame):
    """
    Per-joint panel shown in Resistive Mode.
    Contains:
      • a horizontal intensity slider  (0 – BRAKE_MAX)
      • a live angle readout           (polled from motor)
    """

    def __init__(self, parent, label, color, brake_callback, angle_getter, **kw):
        super().__init__(parent, bg=PANEL_BG,
                         highlightbackground="#2a2a4a", highlightthickness=1,
                         **kw)
        self.label          = label
        self.color          = color
        self.brake_callback = brake_callback   # fn(joint_label, intensity_value)
        self.angle_getter   = angle_getter     # fn() → float | None

        self._build()

    def _build(self):
        # ── Joint title ──
        tk.Label(self, text=self.label.upper(), bg=PANEL_BG, fg=self.color,
                 font=("Helvetica", 12, "bold"), pady=6).pack()

        # ── Live angle display ──
        angle_frame = tk.Frame(self, bg=PANEL_BG)
        angle_frame.pack(pady=(0, 6))

        tk.Label(angle_frame, text="Current Angle", bg=PANEL_BG, fg=TEXT_COL,
                 font=("Helvetica", 9)).pack()

        self.angle_var = tk.StringVar(value="– –")
        tk.Label(angle_frame, textvariable=self.angle_var,
                 bg=PANEL_BG, fg=self.color,
                 font=("Helvetica", 26, "bold")).pack()

        tk.Label(angle_frame, text="degrees", bg=PANEL_BG, fg="#7070a0",
                 font=("Helvetica", 8)).pack()

        # ── Separator ──
        tk.Frame(self, bg=TRACK_COL, height=1).pack(fill=tk.X, padx=16, pady=6)

        # ── Intensity slider ──
        tk.Label(self, text="BRAKE INTENSITY", bg=PANEL_BG, fg=TEXT_COL,
                 font=("Helvetica", 9, "bold")).pack()

        self.intensity_var = tk.DoubleVar(value=0.0)

        slider_frame = tk.Frame(self, bg=PANEL_BG)
        slider_frame.pack(padx=16, pady=6, fill=tk.X)

        tk.Label(slider_frame, text="0", bg=PANEL_BG, fg="#7070a0",
                 font=("Helvetica", 8)).pack(side=tk.LEFT)

        self.scale = tk.Scale(
            slider_frame,
            from_=0, to=BRAKE_MAX,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.intensity_var,
            command=self._on_slider,
            bg=PANEL_BG, fg=self.color,
            troughcolor=TRACK_COL,
            activebackground=self.color,
            highlightthickness=0,
            sliderlength=18,
            length=200,
            showvalue=False,
        )
        self.scale.pack(side=tk.LEFT, padx=6)

        tk.Label(slider_frame, text=str(BRAKE_MAX), bg=PANEL_BG, fg="#7070a0",
                 font=("Helvetica", 8)).pack(side=tk.LEFT)

        # ── Numeric readout under slider ──
        self.intensity_lbl_var = tk.StringVar(value="0")
        tk.Label(self, textvariable=self.intensity_lbl_var,
                 bg=PANEL_BG, fg=self.color,
                 font=("Helvetica", 18, "bold")).pack()
        tk.Label(self, text="intensity", bg=PANEL_BG, fg="#7070a0",
                 font=("Helvetica", 8), pady=(0)).pack(pady=(0, 10))

    def _on_slider(self, value):
        val = round(float(value), 1)
        self.intensity_lbl_var.set(f"{val:.2f}")
        if self.brake_callback:
            self.brake_callback(self.label, val)

    def refresh_angle(self):
        """Call periodically to update the live angle display."""
        angle = self.angle_getter()
        if angle is not None:
            self.angle_var.set(f"{angle:.1f}°")
        else:
            self.angle_var.set("– –")

    def reset(self):
        self.intensity_var.set(0)
        self.intensity_lbl_var.set("0")
        self.angle_var.set("– –")


# ── Main Application ──────────────────────────────────────────────────────────

class JointAngleApp(tk.Tk):

    MODE_SERVO     = "servo"
    MODE_RESISTIVE = "resistive"

    def __init__(self):
        super().__init__()
        self.title("Joint Angle Limiter")
        self.configure(bg=BG)
        self.resizable(False, False)

        self._mode = self.MODE_SERVO
        self._build_ui()
        self._print_header()

        # Start live-angle polling (runs regardless of mode; only visible in resistive)
        self._poll_angles()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Title bar ──
        tk.Label(self, text="⚙  Joint Angle Limiter",
                 bg=BG, fg=TEXT_COL,
                 font=("Helvetica", 16, "bold"), pady=14).pack()

        # ── Mode toggle ──
        toggle_frame = tk.Frame(self, bg=BG)
        toggle_frame.pack(pady=(0, 10))

        # Segmented-style toggle: two adjacent buttons share a container
        seg = tk.Frame(toggle_frame, bg="#2a2a4a", bd=0,
                       highlightbackground="#2a2a4a", highlightthickness=1)
        seg.pack()

        self.btn_servo = tk.Button(
            seg, text="⚙  Servo Mode",
            command=lambda: self._set_mode(self.MODE_SERVO),
            bg=ACCENT1, fg="white", relief=tk.FLAT,
            font=("Helvetica", 10, "bold"),
            padx=16, pady=6, cursor="hand2", bd=0)
        self.btn_servo.pack(side=tk.LEFT)

        self.btn_resist = tk.Button(
            seg, text="⟳  Resistive Mode",
            command=lambda: self._set_mode(self.MODE_RESISTIVE),
            bg="#2a2a4a", fg=TEXT_COL, relief=tk.FLAT,
            font=("Helvetica", 10, "bold"),
            padx=16, pady=6, cursor="hand2", bd=0)
        self.btn_resist.pack(side=tk.LEFT)

        # ── Mode indicator label ──
        self.mode_var = tk.StringVar(value="MODE: SERVO")
        tk.Label(self, textvariable=self.mode_var,
                 bg=BG, fg=ACCENT1,
                 font=("Helvetica", 9, "bold"), pady=2).pack()

        # ── Servo panel ──
        self.servo_panel = tk.Frame(self, bg=BG)
        self._build_servo_panel(self.servo_panel)
        self.servo_panel.pack(padx=24, pady=4)

        # ── Resistive panel (hidden initially) ──
        self.resistive_panel = tk.Frame(self, bg=BG)
        self._build_resistive_panel(self.resistive_panel)
        # not packed yet

        # ── Status bar ──
        self.status_var = tk.StringVar(value="Drag a handle to adjust angle.")
        self.status_lbl = tk.Label(self, textvariable=self.status_var,
                                   bg=BG, fg="#7070a0",
                                   font=("Helvetica", 10), pady=10)
        self.status_lbl.pack()

        # ── Action buttons ──
        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(pady=(0, 16))

        tk.Button(btn_row, text="Print Values", command=self._print_values,
                  bg=ACCENT1, fg="white", relief=tk.FLAT,
                  font=("Helvetica", 10, "bold"),
                  padx=14, pady=6, cursor="hand2").pack(side=tk.LEFT, padx=8)

        tk.Button(btn_row, text="Reset", command=self._reset,
                  bg="#2a2a4a", fg=TEXT_COL, relief=tk.FLAT,
                  font=("Helvetica", 10, "bold"),
                  padx=14, pady=6, cursor="hand2").pack(side=tk.LEFT, padx=8)

    def _build_servo_panel(self, parent):
        row = tk.Frame(parent, bg=BG)
        row.pack()

        for label, color, init in [("Ankle", ACCENT1, 0), ("Knee", ACCENT_KNEE, 0)]:
            col = tk.Frame(row, bg=PANEL_BG, bd=0,
                           highlightbackground="#2a2a4a", highlightthickness=1)
            col.pack(side=tk.LEFT, padx=12, pady=4, ipadx=8, ipady=8)

            tk.Label(col, text=label.upper(), bg=PANEL_BG, fg=color,
                     font=("Helvetica", 12, "bold"), pady=4).pack()

            slider = CircularSlider(col, label, color, initial=init,
                                    callback=self._on_servo_change)
            slider.pack()

            var = tk.StringVar(value=f"{init:.1f}°")
            tk.Label(col, textvariable=var, bg=PANEL_BG, fg=color,
                     font=("Helvetica", 13)).pack(pady=(4, 4))

            # ── Angle text entry + submit ──
            entry_frame = tk.Frame(col, bg=PANEL_BG)
            entry_frame.pack(pady=(2, 8))

            entry = tk.Entry(
                entry_frame,
                width=7,
                bg="#2a2a4a", fg=VAL_COL,
                insertbackground=VAL_COL,
                relief=tk.FLAT,
                font=("Helvetica", 11),
                justify="center",
            )
            entry.pack(side=tk.LEFT, padx=(0, 6), ipady=4)

            def make_submit(jlabel, jslider, jentry):
                def _submit(event=None):
                    raw = jentry.get().strip().rstrip("°")
                    try:
                        angle = float(raw) % 360
                    except ValueError:
                        jentry.delete(0, tk.END)
                        return
                    jslider.angle = angle
                    jslider._draw()
                    self._on_servo_change(jlabel, angle)
                    jentry.delete(0, tk.END)
                return _submit

            submit_fn = make_submit(label, slider, entry)
            entry.bind("<Return>", submit_fn)

            tk.Button(
                entry_frame,
                text="Set",
                command=submit_fn,
                bg=color, fg=BG,
                relief=tk.FLAT,
                font=("Helvetica", 9, "bold"),
                padx=8, pady=3,
                cursor="hand2",
            ).pack(side=tk.LEFT)

            if label == "Ankle":
                self.ankle_slider = slider
                self.ankle_var    = var
                self.ankle_entry  = entry
            else:
                self.knee_slider = slider
                self.knee_var    = var
                self.knee_entry  = entry

    def _build_resistive_panel(self, parent):
        row = tk.Frame(parent, bg=BG)
        row.pack()

        self.ankle_brake = BrakePanel(
            row,
            label="Ankle",
            color=ACCENT1,
            brake_callback=self._on_brake_change,
            angle_getter=self._get_ankle_angle,
        )
        self.ankle_brake.pack(side=tk.LEFT, padx=12, pady=4, ipadx=8, ipady=8)

        self.knee_brake = BrakePanel(
            row,
            label="Knee",
            color=ACCENT_KNEE,
            brake_callback=self._on_brake_change,
            angle_getter=self._get_knee_angle,
        )
        self.knee_brake.pack(side=tk.LEFT, padx=12, pady=4, ipadx=8, ipady=8)

    # ── Mode switching ────────────────────────────────────────────────────────

    def _set_mode(self, mode):
        if mode == self._mode:
            return
        self._mode = mode

        if mode == self.MODE_SERVO:
            self.resistive_panel.pack_forget()
            self.servo_panel.pack(padx=24, pady=4)
            self.btn_servo.config(bg=ACCENT1, fg="white")
            self.btn_resist.config(bg="#2a2a4a", fg=TEXT_COL)
            self.mode_var.set("MODE: SERVO")
            self.status_var.set("Servo mode active – drag handles to set angle.")
            print("[MODE] Switched to SERVO mode")
        else:
            self.servo_panel.pack_forget()
            self.resistive_panel.pack(padx=24, pady=4)
            self.btn_resist.config(bg=ACCENT_KNEE, fg=BG)
            self.btn_servo.config(bg="#2a2a4a", fg=TEXT_COL)
            self.mode_var.set("MODE: RESISTIVE")
            self.status_var.set("Resistive mode active – set brake intensity per joint.")
            print("[MODE] Switched to RESISTIVE mode")

    # ── Servo callbacks ───────────────────────────────────────────────────────

    def _on_servo_change(self, joint: str, angle: float):
        global KNEE_ANGLE, sendNegative

        if joint == "Ankle":
            self.ankle_var.set(f"{angle:.1f}°")
        else:
            if KNEE_ANGLE is not None and angle < KNEE_ANGLE:
                sendNegative = True

            KNEE_ANGLE = angle
            self.knee_var.set(f"{angle:.1f}°")
            print(f"[KNEE] target angle = {KNEE_ANGLE}")

            comm_can_transmit_eid(*move_to_desired_angle(
                bus=can0,
                position=KNEE_ANGLE if not sendNegative else KNEE_ANGLE - 360,
                controller_id=EXO.rightKnee.id
            ))
            sendNegative = False

        a = self.ankle_slider.get_angle()
        k = self.knee_slider.get_angle()
        msg = f"Ankle: {a:.1f}°  |  Knee: {k:.1f}°"
        self.status_var.set(msg)
        print(f"[SERVO UPDATE]  {msg}")

    # ── Resistive brake callbacks ─────────────────────────────────────────────

    def _on_brake_change(self, joint: str, intensity: int):
        """
        Called whenever either brake slider changes.

        For the knee  → calls current_brake() from kneeMotor.motorControl.
        For the ankle → TODO: replace the placeholder with the appropriate
                        ankle-side braking function once identified.
        """
        if joint == "Knee":
            print(f"[BRAKE] Knee intensity = {intensity}")
            comm_can_transmit_eid(*current_brake(
                bus=can0,
                current=intensity,          # adjust argument name to match signature
                controller_id=EXO.rightKnee.id
            ))
        elif joint == "Ankle":
            print(f"[BRAKE] Ankle intensity = {intensity}")
            # ── TODO ──────────────────────────────────────────────────────────
            # Replace this placeholder with the correct ankle braking call, e.g.:
            #   setTorque(intensity)
            #   torque(bus=can0, value=intensity, controller_id=EXO.rightAnkle.id)
            # ─────────────────────────────────────────────────────────────────

        self.status_var.set(
            f"Resistive – Ankle: {self.ankle_brake.intensity_var.get()}  "
            f"| Knee: {self.knee_brake.intensity_var.get()}"
        )

    # ── Live angle getters (plug in real sensor reads) ────────────────────────

    def _get_knee_angle(self) -> float | None:
        """
        Return the current knee angle in degrees from the motor feedback.
        Replace the body with your actual CAN feedback read, e.g.:
            return EXO.rightKnee.current_angle
        """
        # ── TODO: return real sensor value ───────────────────────────────────
        return KNEE_ANGLE  # fallback: last commanded angle

    def _get_ankle_angle(self) -> float | None:
        """
        Return the current ankle angle in degrees from the motor feedback.
        Replace with your actual ankle sensor read.
        """
        # ── TODO: return real sensor value ───────────────────────────────────
        return None

    # ── Polling loop ──────────────────────────────────────────────────────────

    def _poll_angles(self):
        """Refresh live angle readouts in the resistive panel."""
        if self._mode == self.MODE_RESISTIVE:
            self.ankle_brake.refresh_angle()
            self.knee_brake.refresh_angle()

            # 2. Continuously send the knee brake packet based on the current slider value
            knee_intensity = self.knee_brake.intensity_var.get()
            comm_can_transmit_eid(*current_brake(
                bus=can0,
                current=knee_intensity,
                controller_id=EXO.rightKnee.id
            ))
            
        # Re-queue the loop
        self.after(ANGLE_POLL_MS, self._poll_angles)

    # ── Shared button callbacks ───────────────────────────────────────────────

    def _print_values(self):
        if self._mode == self.MODE_SERVO:
            a = self.ankle_slider.get_angle()
            k = self.knee_slider.get_angle()
            print(f"\n[SNAPSHOT – SERVO]  Ankle = {a:.1f}°   Knee = {k:.1f}°\n")
        else:
            ab = self.ankle_brake.intensity_var.get()
            kb = self.knee_brake.intensity_var.get()
            print(f"\n[SNAPSHOT – RESISTIVE]  Ankle brake = {ab}   Knee brake = {kb}\n")

    def _reset(self):
        # Servo reset
        self.ankle_slider.angle = 0.0
        self.ankle_slider._draw()
        self.ankle_var.set("0.0°")
        self.ankle_entry.delete(0, tk.END)
        self.knee_slider.angle = 0.0
        self.knee_slider._draw()
        self.knee_var.set("0.0°")
        self.knee_entry.delete(0, tk.END)

        # Resistive reset
        self.ankle_brake.reset()
        self.knee_brake.reset()

        self.status_var.set("Reset to defaults.")
        print("[RESET]  All values cleared to 0.")

    @staticmethod
    def _print_header():
        print("=" * 42)
        print("   Joint Angle Limiter  –  started")
        print("   Modes: Servo | Resistive")
        print("=" * 42)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    currExo = Exoskeleton()
    EXO     = currExo
    kneeMotors  = [currExo.leftKnee, currExo.rightKnee]
    ankleMotors = [currExo.leftAnkle, currExo.rightAnkle]

    can0 = setupCAN(kneeMotors)

    comm_can_transmit_eid(*set_origin(can0,
                                      set_origin_mode=0,
                                      controller_id=EXO.rightKnee.id))

    app = JointAngleApp()
    app.mainloop()