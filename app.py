import cv2
import csv
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Object Detection Tracker")
        self.root.geometry("1200x800")
        
        # Initialize YOLO
        self.model = YOLO('yolov8n.pt')  # using the smallest model for speed
        
        # Initialize detection tracking
        self.active_detections = {}  # {object_id: [class_name, start_time, last_seen, center_point]}
        self.completed_detections = {}  # Store completed detection periods
        self.tracking_threshold = 30  # frames to wait before considering object gone
        self.frame_count = 0
        self.missing_counts = {}  # Track how many frames an object has been missing
        
        # Create GUI elements
        self.setup_gui()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Start video processing
        self.process_video()
    
    def setup_gui(self):
        # Create main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video frame
        self.video_label = ttk.Label(self.main_frame)
        self.video_label.grid(row=0, column=0, padx=5, pady=5)
        
        # Detection list
        self.tree = ttk.Treeview(self.main_frame, columns=("Object", "Start Time", "Last Seen"), 
                                show="headings", height=10)
        self.tree.heading("Object", text="Object")
        self.tree.heading("Start Time", text="Start Time")
        self.tree.heading("Last Seen", text="Last Seen")
        self.tree.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control buttons
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        self.save_button = ttk.Button(self.button_frame, text="Save Results", 
                                    command=self.save_results)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        self.quit_button = ttk.Button(self.button_frame, text="Quit", 
                                    command=self.quit_app)
        self.quit_button.pack(side=tk.LEFT, padx=5)

    def calculate_center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def get_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def process_video(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            current_time = datetime.now()
            
            # Run YOLOv8 detection
            results = self.model(frame, conf=0.5)
            
            # Track detected objects in this frame
            detected_this_frame = set()
            
            for result in results[0]:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    # Get detection info
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = box.conf[0]
                    
                    # Only track specific objects
                    if class_name not in ['cell phone', 'book', 'person']:
                        continue
                        
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Calculate center point
                    center = self.calculate_center((x1, y1, x2, y2))
                    
                    # Try to match with existing detection
                    matched = False
                    for obj_id, (existing_class, start_time, last_seen, existing_center) in self.active_detections.items():
                        if (existing_class == class_name and 
                            self.get_distance(center, existing_center) < 50):  # threshold for considering same object
                            # Update existing detection
                            self.active_detections[obj_id] = [class_name, start_time, current_time, center]
                            detected_this_frame.add(obj_id)
                            matched = True
                            break
                    
                    if not matched:
                        # Create new detection
                        new_id = f"{class_name}_{len(self.active_detections)}"
                        self.active_detections[new_id] = [class_name, current_time, current_time, center]
                        detected_this_frame.add(new_id)
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} ({confidence:.2f})", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Check for objects that weren't detected in this frame
            all_objects = set(self.active_detections.keys())
            missing_objects = all_objects - detected_this_frame
            
            for obj_id in missing_objects:
                if obj_id not in self.missing_counts:
                    self.missing_counts[obj_id] = 1
                else:
                    self.missing_counts[obj_id] += 1
                
                # If object has been missing for too long, move it to completed detections
                if self.missing_counts[obj_id] >= self.tracking_threshold:
                    if obj_id in self.active_detections:
                        class_name, start_time, last_seen, _ = self.active_detections[obj_id]
                        self.completed_detections[obj_id] = [class_name, start_time, last_seen]
                        del self.active_detections[obj_id]
                        del self.missing_counts[obj_id]
            
            # Remove missing counts for detected objects
            for obj_id in detected_this_frame:
                if obj_id in self.missing_counts:
                    del self.missing_counts[obj_id]
            
            # Update GUI
            self.update_detection_list()
            
            # Convert frame to PhotoImage
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        # Schedule next frame processing
        self.root.after(10, self.process_video)
    
    def update_detection_list(self):
        # Clear current items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add active detections
        for obj_id, (class_name, start_time, last_seen, _) in self.active_detections.items():
            self.tree.insert("", "end", values=(
                f"{class_name} (Active)",
                start_time.strftime("%H:%M:%S"),
                last_seen.strftime("%H:%M:%S")
            ))
        
        # Add completed detections
        for obj_id, (class_name, start_time, last_seen) in self.completed_detections.items():
            self.tree.insert("", "end", values=(
                f"{class_name} (Completed)",
                start_time.strftime("%H:%M:%S"),
                last_seen.strftime("%H:%M:%S")
            ))
    
    def save_results(self):
        with open('detection_results.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Object", "Start Time", "Last Seen", "Status"])
            
            # Save active detections
            for obj_id, (class_name, start_time, last_seen, _) in self.active_detections.items():
                writer.writerow([
                    class_name,
                    start_time.strftime("%H:%M:%S"),
                    last_seen.strftime("%H:%M:%S"),
                    "Active"
                ])
            
            # Save completed detections
            for obj_id, (class_name, start_time, last_seen) in self.completed_detections.items():
                writer.writerow([
                    class_name,
                    start_time.strftime("%H:%M:%S"),
                    last_seen.strftime("%H:%M:%S"),
                    "Completed"
                ])
    
    def quit_app(self):
        self.cap.release()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()