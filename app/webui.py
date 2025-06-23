from flask import Flask, send_from_directory, render_template_string, request
import os

app = Flask(__name__)
PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))

def get_directory_contents(path):
    """Get files and directories in the given path."""
    full_path = os.path.join(PLOTS_DIR, path) if path else PLOTS_DIR
    if not os.path.exists(full_path):
        return [], []
    
    items = os.listdir(full_path)
    directories = [item for item in items if os.path.isdir(os.path.join(full_path, item))]
    files = [item for item in items if item.endswith('.html')]
    
    return sorted(directories), sorted(files)

@app.route('/')
@app.route('/browse/')
@app.route('/browse/<path:subpath>')
def list_plots(subpath=''):
    directories, files = get_directory_contents(subpath)
    
    # Build breadcrumb navigation
    breadcrumbs = []
    if subpath:
        parts = subpath.split('/')
        current_path = ''
        for part in parts:
            current_path = os.path.join(current_path, part).replace('\\', '/')
            breadcrumbs.append(f'<a href="/browse/{current_path}">{part}</a>')
    
    breadcrumb_html = ' / '.join(['<a href="/browse/">Home</a>'] + breadcrumbs)
    
    # Build directory links
    dir_links = []
    for dirname in directories:
        dir_path = os.path.join(subpath, dirname).replace('\\', '/') if subpath else dirname
        dir_links.append(f'<li>📁 <a href="/browse/{dir_path}"><strong>{dirname}/</strong></a></li>')
    
    # Build file links
    file_links = []
    for filename in files:
        file_path = os.path.join(subpath, filename).replace('\\', '/') if subpath else filename
        file_links.append(f'<li>📊 <a href="/plots/{file_path}" target="_blank">{filename}</a></li>')
    
    # Combine all links
    all_links = dir_links + file_links
    
    # Add back navigation if in subdirectory
    back_link = ''
    if subpath:
        parent_path = '/'.join(subpath.split('/')[:-1]) if '/' in subpath else ''
        back_link = f'<p><a href="/browse/{parent_path}">⬅️ Back to parent directory</a></p>'
    
    current_dir = f" - {subpath}" if subpath else ""
    
    html_template = '''
    <h1>Stock Analysis Plots{{current_dir}}</h1>
    <p><strong>Navigation:</strong> {{breadcrumbs|safe}}</p>
    {{back_link|safe}}
    <ul style="list-style-type: none; padding-left: 0;">
        {{links|safe}}
    </ul>
    <hr>
    <p><em>📁 = Folder (click to browse) | 📊 = Plot file (click to view)</em></p>
    '''
    
    return render_template_string(
        html_template,
        current_dir=current_dir,
        breadcrumbs=breadcrumb_html,
        back_link=back_link,
        links=''.join(all_links) if all_links else '<li><em>No plots or folders found</em></li>'
    )

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
