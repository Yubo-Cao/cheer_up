from flask import Flask

app = Flask(__name__)


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/static/<path:path>')
def send_static(path):
    return app.send_static_file(path)

@app.route('/css/<path:path>')
def send_css(path):
    return app.send_static_file('css/' + path)

@app.route('/js/<path:path>')
def send_js(path):
    return app.send_static_file('js/' + path)

@app.route('/img/<path:path>')
def send_img(path):
    return app.send_static_file('img/' + path)


if __name__ == '__main__':
    app.run()
