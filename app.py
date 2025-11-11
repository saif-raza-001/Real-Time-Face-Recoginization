import streamlit as st
import json
import os
import pandas as pd
from echoeye_core import collect_face_data, recognize_faces, load_roles

ROLES_FILE = "roles.json"
DATASET_PATH = "./face_dataset/"
ATTENDANCE_FILE = "attendance.csv"

st.set_page_config(page_title="EchoEye", layout="centered")

st.title("🔐 EchoEye – Real-time Face Recognition")

menu = st.sidebar.selectbox("Choose an action", [
    "Home",
    "Register Face",
    "Start Recognition",
    "Manage Roles",
    "View Attendance"  # New menu option
])

# --------------
# Home Section
# --------------
if menu == "Home":
    st.markdown("""
        ## Welcome to EchoEye 👁️  
        Real-time face recognition system with role-based classification:
        - 🛡️ Admin  
        - 👤 User  
        - 🚫 Intruder  
    """)

# --------------
# Register Face
# --------------
elif menu == "Register Face":
    name = st.text_input("Enter name for new face:")
    if st.button("Start Face Scan"):
        if name.strip():
            st.success(f"Collecting data for '{name}'... Close the webcam window to stop.")
            collect_face_data(name)
        else:
            st.error("Please enter a valid name.")

# --------------
# Start Recognition
# --------------
elif menu == "Start Recognition":
    st.warning("Press 'q' in the recognition window to quit.")
    recognize_faces()

# --------------
# Manage Roles
# --------------
elif menu == "Manage Roles":
    st.subheader("🔧 Manage User Roles")
    
    roles = load_roles(ROLES_FILE)
    users = sorted([f[:-4] for f in os.listdir(DATASET_PATH) if f.endswith('.npy')])

    if not users:
        st.info("No registered users found.")
    else:
        for user in users:
            current_role = roles.get(user, "Intruder")
            new_role = st.selectbox(f"Role for **{user.capitalize()}**:", ["Admin", "User", "Intruder"], index=["Admin", "User", "Intruder"].index(current_role))
            roles[user] = new_role

        if st.button("💾 Save Roles"):
            with open(ROLES_FILE, "w") as f:
                json.dump(roles, f, indent=4)
            st.success("Roles updated successfully.")

# --------------
# View Attendance
# --------------
elif menu == "View Attendance":
    st.subheader("📋 Attendance Records")

    if os.path.exists(ATTENDANCE_FILE):
        try:
            df = pd.read_csv(ATTENDANCE_FILE)
            if not df.empty:
                st.dataframe(df)
            else:
                st.info("Attendance file is empty.")
        except Exception as e:
            st.error(f"Failed to read attendance file: {e}")
    else:
        st.warning("No attendance file found.")
