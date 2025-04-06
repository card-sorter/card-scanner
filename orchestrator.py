import subprocess
import sys
import os
from datetime import datetime, timedelta

DATA_FRESHNESS_DAYS = 7 
SCRIPTS_DIR = "scripts"
CSV_DIR = "sets"
IMAGE_DIR = "images"

def needs_refresh(filepath):
    """Check if file needs refreshing based on age"""
    if not os.path.exists(filepath):
        return True
    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(filepath))
    return file_age > timedelta(days=DATA_FRESHNESS_DAYS)

def run_script(script_path, force_run=False):
    """Run a script with smart skipping"""
    script_name = os.path.basename(script_path)
    
    if "download_csv" in script_name and not needs_refresh(CSV_DIR):
        print(f"\n CSV files are fresh, skipping {script_name}")
        return True
    if "download_images" in script_name and not needs_refresh(IMAGE_DIR):
        print(f"\n Images are fresh, skipping {script_name}")
        return True
        
    print(f"\n=== RUNNING {script_name.upper()} ===")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"\n✓ {script_name} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n🔥 SHIT DON'T WORK HERE {script_name}:")
        print(f"ERROR CODE: {e.returncode}")
        print(f"STDERR:\n{e.stderr}")
        return False

def main():
    print("""
    ████████╗ ██████╗ ██████╗     ██████╗ █████╗ ██████╗ ██████╗ 
    ╚══██╔══╝██╔════╝██╔════╝    ██╔════╝██╔══██╗██╔══██╗██╔══██╗
       ██║   ██║     ██║         ██║     ███████║██████╔╝██║  ██║
       ██║   ██║     ██║         ██║     ██╔══██║██╔══██╗██║  ██║
       ██║   ╚██████╗╚██████╗    ╚██████╗██║  ██║██║  ██║██████╔╝
       ╚═╝    ╚═════╝ ╚═════╝     ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ 
    """)
    
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    other_scripts = sorted(
        os.path.join(SCRIPTS_DIR, f) 
        for f in os.listdir(SCRIPTS_DIR) 
        if f.endswith('.py') and f != 'matcher.py'
    )
    
    for script in other_scripts:
        success = run_script(script)
        if not success:
            print("\n❌ Pipeline failed - fix errors and try again")
            sys.exit(1)
    
    matcher_path = os.path.join(SCRIPTS_DIR, "matcher.py")
    if os.path.exists(matcher_path):
        print("\n=== RUNNING MATCHER ===")
        run_script(matcher_path, force_run=True)
    else:
        print("\n matcher.py not found")
    
    print("\n Orchestration complete")

if __name__ == "__main__":
    main()
