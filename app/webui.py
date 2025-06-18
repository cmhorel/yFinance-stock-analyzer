from flask import Flask, send_from_directory, render_template_string
import os

app = Flask(__name__)
PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))

@app.route('/')
def list_plots():
    files = [f for f in os.listdir(PLOTS_DIR) if f.endswith('.html')]
    links = [f'<li><a href="/plots/{fname}">{fname}</a></li>' for fname in files]
    return render_template_string('<h1>Available Plots</h1><ul>{{ links|safe }}</ul>', links=''.join(links))

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)