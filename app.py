import streamlit as st
import subprocess
import os
from PIL import Image

st.title("AI Fairness Audit Dashboard")

st.write("""
This dashboard evaluates fairness in a loan approval model
using AI Fairness 360 and applies bias mitigation.
""")

if st.button("Run Fairness Mitigation"):
    st.write("Running mitigation process...")
    subprocess.run(["python", "src/mitigation.py"])
    st.success("Mitigation Complete!")

if os.path.exists("results/fairness_comparison.png"):
    st.subheader("Fairness Comparison")
    image = Image.open("results/fairness_comparison.png")
    st.image(image)
