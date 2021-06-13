import flask
import scan_service
from flask import request

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
    path_img_sheet = request.args.get('path')

    if path_img_sheet is None:
        return {'student_code': '', 'permutation_exam_code': '', 'answers': []}

    return scan_service.read_sheet_info_from_image(path_img_sheet);

app.run()
