import streamlit as st
import sys
import os

def main():
    print("Python version:", sys.version)
    print("Current directory:", os.getcwd())
    print("Streamlit version:", st.__version__)
    
    st.write("# Debug Test")
    st.write("If you can see this, Streamlit is working!")

if __name__ == "__main__":
    main()
