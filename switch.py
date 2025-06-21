import os
import argparse
import subprocess
import zipfile
from pathlib import Path
from datetime import datetime

class CodebaseSwitcher:
    def __init__(self):
        self.states = ['preedit', 'beetle', 'sonnet']
        self.current_dir = Path.cwd()
        self.git_dir = self.current_dir / '.git'
        self.script_files = ['switch.py']
        
    def run_command(self, command, capture_output=False):
        """Run shell command"""
        try:
            if capture_output:
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                return result.stdout.strip() if result.returncode == 0 else None
            else:
                return subprocess.run(command, shell=True, check=True).returncode == 0
        except subprocess.CalledProcessError:
            return False if not capture_output else None
    
    def is_git_repo(self):
        return self.git_dir.exists()
    
    def get_current_branch(self):
        return self.run_command("git branch --show-current", capture_output=True) or "main"
    
    def branch_exists(self, branch):
        result = self.run_command(f"git branch --list {branch}", capture_output=True)
        return branch in result if result else False
    
    def create_clineignore(self):
        """Create .clineignore to exclude script files from Cline context"""
        clineignore_path = self.current_dir / '.clineignore'
        ignore_content = "# Codebase State Switch Scripts - exclude from Cline context\n"
        for script_file in self.script_files:
            ignore_content += f"{script_file}\n"
        ignore_content += "*.zip\n.git/\n"
        
        with open(clineignore_path, 'w') as f:
            f.write(ignore_content)
    
    def initialize(self):
        """Initialize git repo with states"""
        if not self.is_git_repo():
            print("🔧 Initializing git repository...")
            self.run_command("git init")
            self.run_command('git config user.name "Codebase Switcher"')
            self.run_command('git config user.email "switcher@codebase.local"')
        
        self.create_clineignore()
        
        # Create initial commit if needed
        result = self.run_command("git log --oneline -1", capture_output=True)
        if not result:
            readme_content = f"# Codebase States\n\nProject Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            with open(self.current_dir / 'README.md', 'w') as f:
                f.write(readme_content)
            
            self.run_command("git add .")
            self.run_command('git commit -m "Initial commit"')
        
        # Create preedit branch
        if not self.branch_exists('preedit'):
            print("📝 Creating preedit branch")
            self.run_command("git checkout -b preedit")
            self.run_command("git add .")
            if self.run_command("git status --porcelain", capture_output=True):
                self.run_command('git commit -m "Initialize preedit state"')
        else:
            self.run_command("git checkout preedit")
        
        # Create beetle and sonnet branches FROM preedit so they have the same content
        for state in ['beetle', 'sonnet']:
            if not self.branch_exists(state):
                print(f"📝 Creating {state} branch from preedit")
                self.run_command(f"git checkout -b {state}")
                self.run_command("git checkout preedit")
        
        # Switch back to preedit state
        self.run_command("git checkout preedit")
        print(f"✅ Initialized with states: {', '.join(self.states)}")
        print("📍 Currently on: preedit")
    
    def switch_state(self, state):
        """Switch to specified state"""
        if state not in ['beetle', 'sonnet']:
            print(f"❌ Invalid state. Available: beetle, sonnet")
            return False
        
        if not self.is_git_repo():
            print("❌ Not a git repo. Run --init first.")
            return False
        
        if not self.branch_exists(state):
            print(f"❌ State '{state}' doesn't exist. Run --init first.")
            return False
        
        current = self.get_current_branch()
        if current == state:
            print(f"✅ Already on {state}")
            return True
        
        # Auto-commit any uncommitted changes
        status = self.run_command("git status --porcelain", capture_output=True)
        if status:
            msg = f"Auto-commit from {current} before switching to {state}"
            print(f"💾 Auto-committing changes: {msg}")
            self.run_command(f"git add . && git commit -m '{msg}'")
        
        print(f"🔄 Switching to {state}...")
        if self.run_command(f"git checkout {state}"):
            print(f"✅ Now on {state}")
            return True
        else:
            print(f"❌ Failed to switch to {state}")
            return False
    
    def show_status(self):
        """Show current status"""
        if not self.is_git_repo():
            print("❌ Not a git repo. Run --init first.")
            return
        
        current = self.get_current_branch()
        print(f"📍 Current state: {current}")
        print(f"📋 Available: beetle, sonnet")
        
        print(f"\n🔄 Git status:")
        self.run_command("git status --short")
        
        print(f"\n📚 Recent commits:")
        self.run_command("git log --oneline -3")
    
    def cleanup_switcher_branches(self):
        """Remove only switcher branches, keep files"""
        print("🧹 Cleaning up switcher branches...")
        
        current = self.get_current_branch()
        
        # Find a safe branch to switch to
        safe_branch = None
        for branch in ['main', 'master']:
            if self.branch_exists(branch):
                safe_branch = branch
                break
        
        # If no main/master, create main from current state
        if not safe_branch:
            print("🔄 Creating main branch...")
            self.run_command("git checkout -b main")
            safe_branch = 'main'
        elif current in self.states:
            print(f"🔄 Switching to {safe_branch}...")
            self.run_command(f"git checkout {safe_branch}")
        
        # Delete only switcher branches
        for state in self.states:
            if self.branch_exists(state):
                print(f"🗑️  Deleting {state} branch...")
                self.run_command(f"git branch -D {state}")
        
        print("✅ Cleanup complete - switcher branches removed, files kept")
    
    def create_zip(self):
        """Create project zip with git structure"""
        zip_name = f"codebase_{self.current_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = self.current_dir / zip_name
        
        print(f"📦 Creating {zip_name} with git structure...")
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(self.current_dir):
                    for file in files:
                        file_path = Path(root) / file
                        relative_path = file_path.relative_to(self.current_dir)
                        
                        # Skip ONLY script files and the zip itself
                        if file in self.script_files or file == zip_name or file == '.clineignore':
                            continue
                        
                        zipf.write(file_path, relative_path)
            
            print(f"✅ Created {zip_name} ({zip_path.stat().st_size / 1024:.1f} KB)")
            print("📋 Zip contains: codebase + .git structure (recipient can switch states)")
            
            # Clean up only branches from local repo
            self.cleanup_switcher_branches()
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Codebase State Switcher')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--init', action='store_true', help='Initialize states')
    group.add_argument('-2', '--beetle', action='store_true', help='Switch to beetle') 
    group.add_argument('-3', '--sonnet', action='store_true', help='Switch to sonnet')
    group.add_argument('-s', '--status', action='store_true', help='Show status')
    group.add_argument('-z', '--zip', action='store_true', help='Create zip')
    
    args = parser.parse_args()
    switcher = CodebaseSwitcher()
    
    if args.init:
        switcher.initialize()
    elif args.beetle:
        switcher.switch_state('beetle')
    elif args.sonnet:
        switcher.switch_state('sonnet')
    elif args.status:
        switcher.show_status()
    elif args.zip:
        switcher.create_zip()

if __name__ == '__main__':
    main() 