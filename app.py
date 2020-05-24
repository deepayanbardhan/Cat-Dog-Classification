from flask import Flask, render_template, request

app=Flask(__name__)

@app.route('/')

def main():
    return render_template('app.html')

@app.route('/send', methods=['POST'])
def send():
    if request.method == 'POST':
        print("insed post")
        img = request.form['avatar']
        operation = request.form['operation']
        if operation=="classify":
            print(str(len(img))+" inside calssift")
            return render_template('app.html')
        else:
            return render_template('apps.html')