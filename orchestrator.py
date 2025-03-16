import subprocess

def run_script(script_name):
    print(f"Running {script_name}...")
    result = subprocess.run(["python", f"scripts/{script_name}"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Shit don't work here {script_name}: {result.stderr}, figure that shit out now")

if __name__ == "__main__":
    run_script("download_csv.py")
    run_script("download_images.py")
    run_script("compare_cards.py")