"""
AI-Powered Waste Segregation - Setup Script
===========================================
Run this script to set up the project and download dependencies.
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")

def install_requirements():
    """Install Python requirements."""
    
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Error installing dependencies")
        sys.exit(1)

def create_directories():
    """Create necessary directories."""
    
    dirs = ['models', 'data', 'samples']
    
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✅ Created directory: {dir_name}/")
        else:
            print(f"📁 Directory exists: {dir_name}/")

def main():
    """Main setup function."""
    
    print("=" * 60)
    print("AI WASTE SEGREGATION - PROJECT SETUP")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    print("\n📁 Creating project directories...")
    create_directories()
    
    # Install requirements
    print("\n📦 Installing dependencies...")
    install_requirements()
    
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    
    print("\n📋 Next steps:")
    print("1. Download the dataset:")
    print("   python download_dataset.py")
    print("\n2. Train the model:")
    print("   python train_model.py")
    print("\n3. Run the Streamlit app:")
    print("   streamlit run app.py")
    
    print("\n📖 See README.md for detailed instructions.")

if __name__ == "__main__":
    main()

