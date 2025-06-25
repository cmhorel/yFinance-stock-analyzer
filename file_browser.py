"""
Simple File Browser for Stock Analyzer

A standalone Flask app to browse all generated files and plots.
"""

import os
from flask import Flask, render_template_string, send_from_directory

app = Flask(__name__)

@app.route('/')
@app.route('/browse/')
@app.route('/browse/<path:subpath>')
def browse_files(subpath=''):
    """Browse files and directories."""
    base_dir = os.path.abspath('.')
    full_path = os.path.join(base_dir, subpath) if subpath else base_dir
    
    if not os.path.exists(full_path):
        return f"Directory not found: {subpath}", 404
    
    if os.path.isfile(full_path):
        # If it's a file, serve it
        directory = os.path.dirname(full_path)
        filename = os.path.basename(full_path)
        return send_from_directory(directory, filename)
    
    # List directory contents
    try:
        items = os.listdir(full_path)
    except PermissionError:
        return f"Permission denied: {subpath}", 403
    
    # Separate directories and files
    directories = []
    files = []
    
    for item in items:
        item_path = os.path.join(full_path, item)
        if os.path.isdir(item_path):
            directories.append(item)
        else:
            files.append(item)
    
    # Sort them
    directories.sort()
    files.sort()
    
    # Build navigation breadcrumbs
    breadcrumbs = []
    if subpath:
        parts = subpath.split('/')
        current_path = ''
        for part in parts:
            current_path = os.path.join(current_path, part).replace('\\', '/')
            breadcrumbs.append(f'<a href="/browse/{current_path}">{part}</a>')
    
    breadcrumb_html = ' / '.join(['<a href="/browse/">📁 Root</a>'] + breadcrumbs)
    
    # Build directory links
    dir_links = []
    for dirname in directories:
        dir_path = os.path.join(subpath, dirname).replace('\\', '/') if subpath else dirname
        dir_links.append(f'<li class="dir-item">📁 <a href="/browse/{dir_path}"><strong>{dirname}/</strong></a></li>')
    
    # Build file links
    file_links = []
    for filename in files:
        file_path = os.path.join(subpath, filename).replace('\\', '/') if subpath else filename
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Choose appropriate icon
        if file_ext in ['.html']:
            icon = '📊'
        elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
            icon = '🖼️'
        elif file_ext in ['.db', '.sqlite']:
            icon = '🗄️'
        elif file_ext in ['.py']:
            icon = '🐍'
        elif file_ext in ['.md']:
            icon = '📝'
        elif file_ext in ['.json']:
            icon = '📋'
        else:
            icon = '📄'
        
        file_links.append(f'<li class="file-item">{icon} <a href="/browse/{file_path}" target="_blank">{filename}</a></li>')
    
    # Combine all links
    all_links = dir_links + file_links
    
    # Add back navigation if in subdirectory
    back_link = ''
    if subpath:
        parent_parts = subpath.split('/')[:-1]
        parent_path = '/'.join(parent_parts) if parent_parts else ''
        back_link = f'<p><a href="/browse/{parent_path}" class="back-link">⬅️ Back to parent directory</a></p>'
    
    current_dir = f" - {subpath}" if subpath else ""
    
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Analyzer File Browser{current_dir}</title>
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background-color: #f5f5f5; 
            }}
            .container {{ 
                max-width: 1000px; 
                margin: 0 auto; 
                background: white; 
                padding: 30px; 
                border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            }}
            .header {{
                text-align: center;
                color: #333;
                border-bottom: 2px solid #007bff;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .breadcrumbs {{
                background: #e9ecef;
                padding: 10px 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                font-size: 14px;
            }}
            .back-link {{
                display: inline-block;
                padding: 8px 15px;
                background: #6c757d;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .back-link:hover {{ background: #5a6268; }}
            ul {{ 
                list-style-type: none; 
                padding-left: 0; 
                margin: 0;
            }}
            li {{ 
                margin: 8px 0; 
                padding: 12px 15px; 
                border-radius: 5px; 
                transition: background-color 0.2s;
            }}
            .dir-item {{ 
                background: #e3f2fd; 
                border-left: 4px solid #2196f3;
            }}
            .file-item {{ 
                background: #f8f9fa; 
                border-left: 4px solid #28a745;
            }}
            li:hover {{ 
                background: #dee2e6; 
            }}
            a {{ 
                text-decoration: none; 
                color: #007bff; 
                font-weight: 500;
            }}
            a:hover {{ 
                text-decoration: underline; 
            }}
            .stats {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
                text-align: center;
                color: #6c757d;
            }}
            .empty {{
                text-align: center;
                color: #6c757d;
                font-style: italic;
                padding: 40px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📁 Stock Analyzer File Browser</h1>
                <p>Browse all generated files, plots, databases, and architecture</p>
            </div>
            
            <div class="breadcrumbs">
                <strong>📍 Location:</strong> {breadcrumbs}
            </div>
            
            {back_link}
            
            {content}
            
            <div class="stats">
                📊 {dir_count} directories • 📄 {file_count} files
            </div>
        </div>
    </body>
    </html>
    '''
    
    if all_links:
        content = f'<ul>{"".join(all_links)}</ul>'
    else:
        content = '<div class="empty">📭 This directory is empty</div>'
    
    return html_template.format(
        current_dir=current_dir,
        breadcrumbs=breadcrumb_html,
        back_link=back_link,
        content=content,
        dir_count=len(directories),
        file_count=len(files)
    )

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    """Serve plot files."""
    plots_dir = os.path.abspath('plots')
    return send_from_directory(plots_dir, filename)

@app.route('/data/<path:filename>')
def serve_data(filename):
    """Serve data files."""
    data_dir = os.path.abspath('data')
    return send_from_directory(data_dir, filename)

if __name__ == '__main__':
    print("🗂️  Starting File Browser")
    print("=" * 40)
    print("📁 Browse files: http://localhost:5001")
    print("📊 Direct plots: http://localhost:5001/plots/")
    print("🗄️ Direct data: http://localhost:5001/data/")
    print("=" * 40)
    
    app.run(host='0.0.0.0', port=5001, debug=False)
