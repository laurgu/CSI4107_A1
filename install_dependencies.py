import subprocess

def install_dependencies():
    try:
        subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])
        print("All dependencies have been successfully installed.")
    except subprocess.CalledProcessError as e:
        print("An error occurred while installing dependencies:", e)

if __name__ == "__main__":
    install_dependencies()