import subprocess
import sys
import os

def run_module(module_name):
    print(f"Running {module_name} as a module...")
    result = subprocess.run([sys.executable, "-m", module_name], check=True)
    print(f"{module_name} finished with exit code {result.returncode}")

if __name__ == "__main__":
    # Change working directory to project root (one level up from this file)
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Run the modules in order
    run_module("app.stockScraper")
    run_module("app.stockAnalyzer")
    run_module("app.stockSimulator")
    run_module("app.webui")