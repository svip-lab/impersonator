from utils.util import load_pickle_file, write_pickle_file


neural_smpl_path = 'pretrains/smpl_model.pkl'
body_regressor_path = '/public/liuwen/octopus/assets/J_regressor.pkl'
face_regressor_path = '/public/liuwen/octopus/assets/face_regressor.pkl'


neural_smpl = load_pickle_file(neural_smpl_path)
body_regressor = load_pickle_file(body_regressor_path)
face_regressor = load_pickle_file(face_regressor_path)


neural_smpl['body25_regressor'] = body_regressor
neural_smpl['face70_regressor'] = face_regressor

write_pickle_file(neural_smpl_path, neural_smpl)
print(neural_smpl_path)