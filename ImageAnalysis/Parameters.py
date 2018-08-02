import errno
import os
import re
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import ast
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras.models import model_from_json
import keras.losses
import Parameter_Models
from PIL import Image
import json
from tensorflow.python.client import device_lib
import tensorflow as tf


def main():
	"""
	Contains the command line interface for the ParameterModel class
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('operation', action='store', choices=['train', 'test', 'predict', 'analyze','info'], help='Which operation to do')
	parser.add_argument('directories', action='store', help='The directories to load images from')
	parser.add_argument('-l', '--model_to_load', action='store', dest='model_to_load', help='Previous model to load and use for training or predictions', default=None)
	parser.add_argument('-v', '--model_to_create', action='store', dest='model_to_create',type=int, help='Model to create', default='1')
	parser.add_argument('-t', '--image_types', action='store', dest='image_types', help='Types (first 3 letters of file names) of images to train on. String of characters with no spaces', default=None)
	parser.add_argument('-n', '--run_id', action='store', dest='run_id', help='Name for operation. Supplying a name will create a new results folder with that name', default=None)
	parser.add_argument('-e', '--epochs', action='store', dest='n_epochs', type=int, help='Number of epochs to use for training and traintest operations', default=1)
	parser.add_argument('-m', '--parameters', action='store', dest='parameter', help='Map one parameter to another with a dictionary', default=None)
	parser.add_argument('-p', '--number_parameter', action='store', dest='n_param', type=int, help='Number of parameters to train', default='1')
	parser.add_argument('-d', '--number_division', action='store', dest='n_division', type=int, help='Number of division to analyze', default='5')
	parser.add_argument('-g', '--gpu', action='store', dest='gpu_to_use', type=int, help='GPU to use', default='1')

	args = parser.parse_args()

	if args.operation == 'train':
		if args.gpu_to_use == 0:
			os.environ["CUDA_VISIBLE_DEVICES"] = "0"
			print('Using GPU:0');
			print('Device used:', device_lib.list_local_devices())
			if args.run_id is None and args.model_to_load is None:  # The run will have a new folder created for it and it needs a new name
				args.run_id = args.operation + ' ' + str(args.n_epochs) + ' epochs ' + 'from ' + os.path.basename(os.path.normpath(args.directories[0]))

			model = ParameterModel(args.model_to_load, args.run_id)

			if hasattr(args, 'image_types'):
				model.set_image_types(args.image_types)
			print('Directories to load from:', ast.literal_eval(args.directories))


			model.train_operation(ast.literal_eval(args.directories), args.n_epochs, args.image_types, ast.literal_eval(args.parameter), args.n_param, args.model_to_create)
		elif args.gpu_to_use == 1:
			os.environ["CUDA_VISIBLE_DEVICES"] = "1"
			print('Using GPU:1');
			print('Device used:', device_lib.list_local_devices())
			if args.run_id is None and args.model_to_load is None:  # The run will have a new folder created for it and it needs a new name
				args.run_id = args.operation + ' ' + str(args.n_epochs) + ' epochs ' + 'from ' + os.path.basename(os.path.normpath(args.directories[0]))

			model = ParameterModel(args.model_to_load, args.run_id)

			if hasattr(args, 'image_types'):
				model.set_image_types(args.image_types)
			print('Directories to load from:', ast.literal_eval(args.directories))
			model.train_operation(ast.literal_eval(args.directories), args.n_epochs, args.image_types, ast.literal_eval(args.parameter), args.n_param, args.model_to_create)

	elif args.operation == 'analyze':
		print('Analyze Operation')
		if args.model_to_load is None:  # This is a completely new model and a new training run
			sys.exit('A test operation must load a model')

		print('MODEL TO LOAD', args.model_to_load)
		model = ParameterModel(args.model_to_load, args.run_id)

		model.analyze_operation(ast.literal_eval(args.directories), args.image_types, ast.literal_eval(args.parameter), args.n_division, args.model_to_load, args.model_to_create)


	elif args.operation == 'test':
		print('Test Operation')
		if args.model_to_load is None:  # This is a completely new model and a new training run
			sys.exit('A test operation must load a model')

		print('MODEL TO LOAD', args.model_to_load)
		model = ParameterModel(args.model_to_load, args.run_id)

		model.test_operation(ast.literal_eval(args.directories), args.image_types, ast.literal_eval(args.parameter), args.model_to_load, args.n_param)

	elif args.operation == 'predict':
		print('Predict Operation')
		if args.model_to_load is None:  # This is a completely new model and a new training run
			sys.exit('A predict operation must load a model')
		print('MODEL TO LOAD', args.model_to_load)
		print('Device used:', device_lib.list_local_devices())
		with tf.device('/cpu:0'):
			model = ParameterModel(args.model_to_load, args.run_id)

		print('Image Directories to load from:', ast.literal_eval(args.directories))
		model.predict_operation(ast.literal_eval(args.directories), ast.literal_eval(args.parameter), args.model_to_load, args.n_param, args.model_to_create)

	elif args.operation == 'info':
		pass



class ParameterModel:
	"""
	Creating a new model and training on it
		model = ParameterModel(run_id, None)
		model.train_operation(directories, epochs, image_types=image_types, parameter=parameter)

	Training an existing model without duplicating/creating a new folder
		model = ParameterModel(None, model_to_load)
		model.train_operation(directories, epochs, image_types=image_types, parameter=parameter)

	Training an existing model and creating a new folder for the changed model
		model = ParameterModel(run_id, model_to_load)
		model.train_operation(directories, epochs, image_types=image_types, parameter=parameter)

	Testing an existing model on a data set
		model = ParameterModel(run_id (optional), model_to_load)
		model.test_op(directories, image_types=image_types, parameter=parameter)

	Predicting values for a set of unlabeled images
		model = ParameterModel(run_id (optional), model_to_load)
		model.predict_op(directories)

	Getting info on an existing ParameterModel Object that has been saved to a directories
		model = ParameterModel(None, model_to_load)
		model.get_info()
	"""

	model = None
	results_dir = ''

	#trained_parameter_map = {}  # Dictionary mapping of equivalent parameters. {'f':'r'} means 'f' should be equivalent to 'r' for training
	trained_image_types = []  # A list of the first three identifier letters at the beginning of the image files to only train on those types
	image_names = []
	test_names = []
	analyze_index = []
	train_names = []
	trained_parameters = None
	n_trained_parameters = 1
	indicator = 0
	loaded_model = None
	trained_epochs = 0

	test_margin = 10

	batch_size = 100
	image_dim = 0  # TODO: Add support for rectangular images

	x_train = []
	y_train = []
	x_test = []
	y_test = []

	train_predictions = []
	test_predictions = []

	def __init__(self, model_to_load, run_id):
		# Loads data and creates model by either loading an existing model or creating a new one

		if model_to_load:
			self.loaded_model = os.path.basename(os.path.normpath(model_to_load))
			self.retrieve_model(model_to_load)

		if run_id:
			self.make_results_directory(run_id)  # Overrides the self.results_dir value from retrieve model
			print('Created results directory:', self.results_dir)
		else:
			# Existing results directory should be reused because no run_id has been provided
			# run_id must be provided if model_to_load is not provided, because a new model is created TODO: Check this
			print('Results in directory:', self.results_dir)


	def train_operation(self, image_dirs, epochs, image_types, parameter, n_param, model_to_create):
		# Loads and trains on data and saves/shows result data and plots

		print('Loading Data')
		self.load_train_and_test_data(image_dirs, image_types, parameter, n_param, model_to_create)
		print(model_to_create)

		if self.model is None:
			if model_to_create==1 :
				print('Creating VGG19 Model')
				self.create_vggmodel(self.n_trained_parameters)  # n_trained_parameters will be filled because data loading happens just before
			elif model_to_create==2 :
				print('Creating Customized Model')
				self.create_model(self.n_trained_parameters)
			elif model_to_create==3 :
				print('Creating ResNet50 Model')
				self.create_resmodel(self.n_trained_parameters)


		history = self.train(epochs)  # Fills self.train_predictions
		self.plot_loss_history(history)
		train_scores = margin_metric(self.test_margin, self.train_predictions, self.y_train)
		test_scores = self.test()
		print('Test Score: ', test_scores)
		print('Plotting training results.')
		self.plot_against_y(self.train_predictions, self.y_train, 'Train Predictions vs Actual Values', train_scores)
		print('Plotting test results.')
		self.plot_against_y(self.test_predictions, self.y_test, 'Test Predictions vs Actual Values', test_scores)
		print('Saving model files.')
		self.save_model_and_params()
		self.save_training_description(image_dirs)

	def test_operation(self, image_dirs, image_types, parameter, loaded_model, n_param, model_to_create):
		# Assumes model has already been loaded when the ParameterObject object was created

		print('Loading Data')
		self.load_test_data(image_dirs, image_types, parameter,n_param,model_to_create)

		test_scores = self.test()
		print('Test Score: ', test_scores)
		print('Plotting the results:')
		self.plot_against_y(self.test_predictions, self.y_test, 'Test Predictions vs Values', test_scores)


	def analyze_operation(self, image_dirs, image_types, parameter, n_div, loaded_model, model_to_create):
		# Assumes model has already been loaded when the ParameterObject object was created
		image_path = os.path.normpath(image_dirs[0])
		print('Loading Data')
		self.load_analyze_data(image_path, image_types, parameter, n_div, model_to_create)
		print('Analyzing:')
		self.analyze() #generates 'predict results' file

		# self.plot_analysis(self.test_predictions)


	def predict_operation(self, image_dirs, parameter, model_to_load, n_param, model_to_create):
		# For making a set of predictions from unlabeled data
		print('Loading Data')
		self.trained_parameters = parameter;
		self.load_only_x(image_dirs,n_param,model_to_create) #fills x_test
		print('Predicting:')
		self.predict() #generates 'predict results' file



	def retrieve_model(self, model_dir):
		try:
			print('Loading model from:', os.path.abspath(os.path.join(model_dir, 'Model.h5')))
			#self.model = load_model(os.path.join(model_dir, 'Model.h5'),custom_objects={'get_customLoss': Parameter_Models.get_customLoss})
			with open(os.path.join(model_dir, 'model_architecture.json'), 'r') as f:
				self.model = model_from_json(f.read())

			self.model.load_weights(os.path.join(model_dir, 'Model_Weight.h5'))
			self.model.compile(loss=Parameter_Models.get_customLoss(),optimizer='adam')
		except (ImportError, ValueError):
			sys.exit('Error importing model.h5 file.' + os.path.join(model_dir, 'Model.h5') + ' No such file, or incompatible')

		with open(os.path.join(model_dir, 'Parameters.pickle'), 'rb') as parameters_file:
			parameters = pickle.load(parameters_file)
			self.results_dir                = parameters['results_dir']
			#self.trained_parameter_map      = parameters['parameter_map']
			self.trained_image_types        = parameters['image_types']
			self.trained_parameters         = parameters['trained_parameters']
			self.n_trained_parameters       = len(self.trained_parameters)
			self.batch_size                 = parameters['batch_size']


	def make_results_directory(self, run_id):
		results_dir = str(run_id)
		try:
			os.makedirs(results_dir)
		except OSError as exception:
			if exception.errno != errno.EEXIST:
				raise

		self.results_dir = results_dir


	def load_train_and_test_data(self, image_dirs, image_types, parameter,n_param, model_to_create):
		x, y = self.load_data(image_dirs, image_types, parameter, True, n_param, model_to_create)

		print("X shape:",x.shape)
		# print(y[0:30, :])


		test_split = int(x.shape[0] * 0.8)
		names = np.asarray(self.image_names)

		self.train_names, self.test_names = np.array_split(names, [test_split])
		self.x_train, self.x_test = np.array_split(x, [test_split])
		self.y_train, self.y_test = np.array_split(y, [test_split])
		print(self.x_train.shape)

	def load_test_data(self, image_dirs, image_types, parameter, n_param, model_to_create):
		x, y = self.load_data(image_dirs, image_types, parameter, True, n_param, model_to_create)

		names = np.asarray(self.image_names)
		self.test_names = names
		self.x_test = x
		self.y_test = y


	def load_only_x(self, image_dirs, n_param, model_to_create):
		# For loading unlabelled data.
		x = self.load_data(image_dirs, None, None, False, n_param, model_to_create)
		names = np.asarray(self.image_names)
		self.test_names = names
		self.x_test = x

	def get_analyze_image(self, image_dirs,gap,x_coord, y_coord):
		with Image.open(image_dirs) as image:
			crop_rectangle = (x_coord, y_coord, x_coord+self.image_dim, y_coord+self.image_dim)
			cropped_im = image.crop(crop_rectangle)
			ix = int(x_coord/gap)
			iy = int(y_coord/gap)
			name = str(ix)+str(iy)
			file_path = os.path.join(self.results_dir,name)
			cropped_im.save(file_path+".jpeg","JPEG")
			image = np.array(cropped_im) / 255
			image = image[:, :].reshape((self.image_dim, self.image_dim, 1)).astype(np.float32)
			return image

	def load_analyze_data(self, image_dirs, image_types, parameter, n_div, model_to_create):
		# For loading unlabelled data.

		self.trained_parameters = parameter
		if(model_to_create==1 or model_to_create == 3):
			self.image_dim = 224
		elif(model_to_create==2):
			self.image_dim = 200

		with Image.open(image_dirs) as image:
			width, height = image.size
			original_dim = width

		print('Loading x:')
		gap = (original_dim-self.image_dim)/(n_div-1)
		print("gap:",gap)
		for ix in range(n_div):
			for iy in range(n_div):
				self.x_test.append(self.get_analyze_image(image_dirs,gap,ix*gap,iy*gap))
				self.analyze_index.append(str(ix)+str(iy))

		self.x_test = np.asarray(self.x_test)
		self.analyze_index = np.asarray(self.analyze_index).reshape(n_div*n_div,1)
		print(self.x_test.shape)
		print(self.analyze_index.shape)


	def load_data(self, image_dirs, image_types, parameter, load_y, n_param, model_to_create):  # TODO: Add support for loading images from multiple directories
		"""
		Gets x and y values for images with types matching values in image_types in the specified directories.
		Maps parameters

		Fills the following instance variables:
			trained_parameters
			n_trained_parameters
			x_train, x_test, y_train, y_test
			image_dim
		"""
		# Each image name should be the combination of its immediate folder and the file name of the image
		#image_names = []
		self.trained_parameters = parameter

		for image_dir in image_dirs:  # Get image names from all provided directories
			self.image_names += [os.path.join(os.path.basename(os.path.dirname(image_dir)), name) for name in os.listdir(image_dir)]  # Add the name of the folder the image is from to every image name
		np.random.shuffle(self.image_names)

		image_dir_dir = os.path.dirname(os.path.dirname(image_dirs[0]))  # The directory that holds all of the image directories. This assumes all of the directories live in a common folder.
		#self.get_image_dim(os.path.join(image_dir_dir, self.image_names[1]))

		if(model_to_create==1 or model_to_create == 3):
			self.image_dim = 224
		elif(model_to_create==2):
			self.image_dim = 200

		# Remove the images types that aren't wanted for training if some are specified with a list
		# if image_types is not None:
		# 	self.image_names = [name for name in self.image_names if any(kind in name for kind in image_types)]

		n_names = len(self.image_names)
		print('Number of images:',n_names)
		print('Loading x:')
		x = np.array([self.get_image(os.path.join(image_dir_dir, name),model_to_create) for name in self.image_names])

		if( model_to_create==1 or model_to_create==3):
			x = x.reshape((n_names, self.image_dim, self.image_dim, 3)).astype(np.float32)
		elif(model_to_create==2):
			x = x.reshape((n_names, self.image_dim, self.image_dim, 1)).astype(np.float32)


		if not load_y:  # For unlabeled data, the function should return before it tries to gather labels
			return x

		self.n_trained_parameters = n_param
		y = np.full((n_names, self.n_trained_parameters),np.nan) # Initialize the array to nan to mask the missing data

		temp_image_names = list(self.image_names)  # Duplicate list so the original names aren't changed in case they need to be used

		# Map the parameters in the parameter map dictionary
		#for i in range(len(temp_image_names)):
		#	for letter in parameter_map:
		#		if '-' + letter in temp_image_names[i]:
		#			temp_image_names[i] = temp_image_names[i].replace('-' + letter, '-' + parameter_map[letter])
		#self.trained_parameter_map = parameter_map


		#parameter_indexes = [m.start() + 1 for m in re.finditer('-', temp_image_names[0])][0:-1]  # Don't use the last '-' in the name. It's before the last number, not a parameter
		#self.trained_parameters = [temp_image_names[0][i] for i in parameter_indexes]  # Get the parameters from the first image. Assumes all images have consistent parameters
		#self.n_trained_parameters = len(self.trained_parameters)

		# Build y values
		print('Loading y:')
		for i_name in range(len(temp_image_names)):
			parameter_indexes = [m.start() + 1 for m in re.finditer('-', temp_image_names[i_name])]
			temp_trained_parameters = [temp_image_names[i_name][i] for i in parameter_indexes[0:-1]]
			# print('The parameters found in this image are:', temp_trained_parameters)
			temp_n_parameters = len(temp_trained_parameters)

			for i_letter in range(self.n_trained_parameters):
				if temp_n_parameters == self.n_trained_parameters:
					letter = temp_trained_parameters[i_letter]
					index_in_name = temp_image_names[i_name].find('-' + letter)
					index_end_name = parameter_indexes[i_letter+1]-1
					if index_in_name == -1:
						#print(image_names[i_name], 'is missing the', letter, 'parameter')
						continue
					value = temp_image_names[i_name][index_in_name + 2:index_end_name]
					y[i_name, i_letter] = value
				elif temp_n_parameters == 1:
					cur_letter = self.trained_parameters[i_letter]
					letter = temp_trained_parameters[0]
					if letter == cur_letter:
						index_in_name = temp_image_names[i_name].find('-' + letter)
						index_end_name = parameter_indexes[1]-1
						value = temp_image_names[i_name][index_in_name + 2:index_end_name]
						y[i_name, i_letter] = value



		# # FIND IMAGE NAMES THAT AREN'T GIVING VALUES
		# for i in range(len(temp_image_names)):
		# 	if y[i, 0] == -100:
		#		print('Value not assigned, name:', temp_image_names[i])
		return x, y


	# def get_image_dim(self, image_path):
	# 	with Image.open(image_path) as image:
	# 		width, height = image.size
	# 	assert (width == height), 'Images should be square'
	# 	self.image_dim = width


	def get_image(self, path,model_to_create):
		with Image.open(path) as image:
			if(model_to_create == 1 or model_to_create==3):
				temp_image = image.resize((224,224))
				image = np.array(temp_image) / 255
				print(image.shape)
				if(image.shape == (224,224,4)):
					return image[:, :, 0:3].reshape((self.image_dim, self.image_dim, 3)).astype(np.float32)
				else:
					return image[:, :, :].reshape((self.image_dim, self.image_dim, 3)).astype(np.float32)
			elif(model_to_create == 2):
				temp_image = image.resize((200,200))
				image = np.array(temp_image) / 255
				return image[:, :, 0].reshape((self.image_dim, self.image_dim, 1)).astype(np.float32)



	def create_vggmodel(self, n_parameters):
		self.model = Parameter_Models.vgg19_custom(n_parameters)

	def create_resmodel(self, n_parameters):
		self.model = Parameter_Models.ResNet_custom(n_parameters)

	def create_model(self, n_parameters):
		self.model = Parameter_Models.more_conv_multiple(self.image_dim, n_parameters)

	def save_training_description(self, image_dirs):
		with open(os.path.join(self.results_dir, 'Model Summary.txt'), 'w+') as summary_file:
			summary_file.write('Parameters: ')
			for letter in self.trained_parameters:
				summary_file.write(letter + ', ')
			summary_file.write('\n' + 'Trained from ')
			for image_dir in image_dirs:
				summary_file.write(image_dir + ', ')
			summary_file.write('\n')
			if self.loaded_model:
				summary_file.write('Loaded from previously trained model in ' + self.loaded_model + '\n')
			if self.trained_image_types is None:
				summary_file.write('Trained on all image types')
			else:
				summary_file.write('Valid image types: ')
				for kind in self.trained_image_types:
					summary_file.write(kind + ', ')
			summary_file.write('\n' + 'Trained for ' + str(self.trained_epochs) + ' epochs.' + '\n')
			summary_file.write('Tested with a margin of ' + str(self.test_margin) + ' points.' + '\n')
			summary_file.write('Images had dimension ' + str(self.image_dim) + ' pixels, square.' + '\n')

			summary_file.write('\n')

			self.model.summary(print_fn=lambda x: summary_file.write(x + '\n'))

			# plot_model(self.model, to_file=os.path.join(self.results_dir, 'Model Plot.png'), show_shapes=True, show_layer_names=True)

	def save_model_and_params(self):
		self.model.save(os.path.join(self.results_dir, 'Model.h5'))
		self.model.save_weights(os.path.join(self.results_dir, 'Model_Weight.h5'))

		with open(os.path.join(self.results_dir, 'model_architecture.json'), 'w') as f:
			f.write(self.model.to_json())

		to_save = {'results_dir': self.results_dir, 'image_types': self.trained_image_types, 'trained_parameters': self.trained_parameters, 'batch_size': self.batch_size}
		with open(os.path.join(self.results_dir, 'Parameters.pickle'), 'wb') as parameter_file:
			pickle.dump(to_save, parameter_file)

	def test(self):

		print('Compiling Model:')
		self.model.compile(loss=Parameter_Models.get_customLoss(),optimizer='adam')
		print('Predicting Test Data:')
		self.test_predictions = self.model.predict(self.x_test, batch_size=self.batch_size)
		np.clip(self.test_predictions, 0, 100, out=self.test_predictions)


		print('Test_names shape:',self.test_names.shape)
		print('Predictions shape:',self.test_predictions.shape)

		print('Saving Test Results.csv:')
		results = np.column_stack([self.test_names,self.y_test,self.test_predictions])
		np.savetxt(os.path.join(self.results_dir, 'Test Results.csv'),results, delimiter=',', header="name,r_label,m_label,r_pred,m_pred", fmt='%s')

		# output = np.zeros(names.size, dtype=[('name', 'U32'), ('r_label', float),('m_label',float),('r_pred', float),('m_pred',float)])
		# output['name'] = names
		# output['r_label'] = self.y_test[:,0:1].ravel()
		# output['m_label'] = self.y_test[:,1:2].ravel()
		# output['r_pred'] = self.test_predictions[:,0:1].ravel()
		# output['m_pred'] = self.test_predictions[:,1:2].ravel()
		# np.savetxt(os.path.join(self.results_dir, 'Test Results.csv'), output, delimiter=",", header="name,r_label,m_label,r_pred,m_pred", fmt="%10s %10.3f %10.3f %10.3f %10.3f", comments='')

		#np.savetxt(os.path.join(self.results_dir, 'Test Results.csv'), np.hstack((self.test_predictions, self.y_test)), delimiter=',', header=(','.join(self.trained_parameters) + ',') * 2)

		print('Calculating the Test Scores:')
		test_scores = margin_metric(self.test_margin, self.test_predictions, self.y_test)
		#print('Loss is:', self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size))

		return test_scores


	def train(self, n_epochs):
		# Trains on images that are already loaded into the object's instance variables
		# If n_epochs is less than 1, the model will stop training when the model has not improved in the absolute value of that number os epochs

		if n_epochs < 0:
			converge_monitor = EarlyStopping(patience=abs(n_epochs))
			history = self.model.fit(self.x_train, self.y_train, epochs=n_epochs, batch_size=self.batch_size, callbacks=[converge_monitor])

		else:
			model_checkpoint = ModelCheckpoint(os.path.join(self.results_dir, 'model_weights.{epoch:02d}-{val_loss:.2f}.hdf5)'),monitor='val_loss',save_best_only=True,save_weights_only=True,mode='auto',period=1)
			history = self.model.fit(self.x_train, self.y_train, epochs=n_epochs, batch_size=self.batch_size, callbacks=[model_checkpoint],validation_data=(self.x_test,self.y_test))

		print('Predicting training data:')
		self.train_predictions = self.model.predict(self.x_train, batch_size=self.batch_size)
		np.clip(self.train_predictions, 0, 100, out=self.train_predictions)

		print('Train_names shape:',self.train_names.shape)
		print('Predictions shape:',self.train_predictions.shape)
		print('Saving Train Results.csv:')
		results = np.column_stack([self.train_names,self.y_train,self.train_predictions])
		np.savetxt(os.path.join(self.results_dir, 'Train Results.csv'),results, delimiter=',', header="name,r_label,m_label,r_pred,m_pred", fmt='%s')

		self.trained_epochs = n_epochs
		return history

	def predict(self):

		self.model.compile(loss=Parameter_Models.get_customLoss(),optimizer='adam')
		self.test_predictions = self.model.predict(self.x_test, batch_size=self.batch_size)
		np.clip(self.test_predictions, 0, 100, out=self.test_predictions)


		print('Predict_names shape:',self.test_names.shape)
		print('Predictions shape:',self.test_predictions.shape)

		output = np.column_stack([self.test_names,self.test_predictions])

		np.savetxt(os.path.join(self.results_dir, 'PredictResults.csv'),output, delimiter=',', header="name,r_pred,m_pred", fmt='%s')

	def analyze(self):

		self.model.compile(loss='mse',optimizer='adam')
		self.test_predictions = self.model.predict(self.x_test, batch_size=self.batch_size)
		np.clip(self.test_predictions, 0, 100, out=self.test_predictions)

		print('Predict_names shape:',self.analyze_index.shape)
		print('Predictions shape:',self.test_predictions.shape)

		output = np.column_stack((self.analyze_index,self.test_predictions))
		np.savetxt(os.path.join(self.results_dir, 'AnalyzeResults.csv'),output, delimiter=',', header="region_index,r_pred", fmt='%s')

	def plot_loss_history(self, history):
		train_loss_history = history.history['loss']
		train_loss_history = np.array(train_loss_history).reshape((len(train_loss_history), 1))

		figure, plot = plt.subplots(1, 1, figsize=(8, 6))
		plot.set_title('Model Loss')
		plot.set_xlabel('Epoch')
		plot.set_ylabel('Loss')

		plot.plot(train_loss_history)

		if 'val_loss' in history.history:
			validation_loss_history = history.history['val_loss']
			validation_loss_history = np.array(validation_loss_history).reshape((len(validation_loss_history), 1))
			loss_history = np.hstack((train_loss_history, validation_loss_history))

			plot.plot(validation_loss_history)
			plot.legend(['Train', 'Validation'])
		else:
			loss_history = train_loss_history

		np.savetxt(os.path.join(self.results_dir, 'Loss History.csv'), loss_history, delimiter=',', header=(','.join(self.trained_parameters) + ',') * 2)

		figure.canvas.set_window_title('Loss History')
		figure.savefig(os.path.join(self.results_dir, 'Loss History.png'), dpi=100)


	def plot_predictions(self, predictions, title):
		train_figure, subplots = plt.subplots(1, self.n_trained_parameters, figsize=(6 * self.n_trained_parameters, 6))  # Create a subplot for each parameter

		for i in range(self.n_trained_parameters):
			n_train = predictions.shape[0]

			predictions = predictions.reshape(n_train, 2)
			plot_results(subplots[i], predictions, self.trained_parameters[i], title)  # TODO Fix this, a new function probably needed


	def plot_against_y(self, predictions, y, title, score):
		train_figure, subplots = plt.subplots(1, self.n_trained_parameters, figsize=(6 * self.n_trained_parameters, 6))  # Create a subplot for each parameter

		predictions_and_y = np.hstack((predictions, y))
		if self.n_trained_parameters == 1:
			plot_results(subplots, predictions_and_y, score, self.trained_parameters)
		else:
			for i in range(self.n_trained_parameters):
				n_train = predictions_and_y.shape[0]
				parameter_predictions_and_y = np.hstack((predictions_and_y[:, i].reshape(n_train, 1), predictions_and_y[:, self.n_trained_parameters + i].reshape(n_train, 1)))
				parameter_predictions_and_y = parameter_predictions_and_y.reshape(n_train, 2)
				plot_results(subplots[i], parameter_predictions_and_y, score[i], self.trained_parameters[i])

		train_figure.canvas.set_window_title(title)
		train_figure.legend()

		train_figure.savefig(os.path.join(self.results_dir, title + '.png'), dpi=100)


	def set_parameter(self, new_parameter):
		# The parameter map is used equivocate one image parameter to another for training.
		# For example, to test how similar noise and roundness are, you could map f (noise) to r (roundness) with the map {'f':'r'}
		self.trained_parameter = new_parameter

	def set_image_types(self, image_types):
		self.trained_image_types = image_types

	def set_margin(self, margin):
		self.test_margin = margin


def margin_metric(margin, x, y):
	# Returns ratio of x that is within the provided margin of y
	# x and y are numpy arrays. They can have multiple columns, one for each parameter
	return ((x < y + margin) & (x > y - margin)).sum(axis=0)/x.shape[0]


def plot_results(plot, predictions_and_y, score, parameter, image_names=None):
	x = np.array(range(1, predictions_and_y.shape[0] + 1)).reshape((predictions_and_y.shape[0], 1))

	# Sort the predictions and y values by the y values
	indexes_sorted_by_y = predictions_and_y[:, 1].argsort()
	predictions_and_y = predictions_and_y[indexes_sorted_by_y]

	plot.scatter(x, predictions_and_y[:, 0], label='Predicted', color='firebrick', s=10)
	plot.scatter(x, predictions_and_y[:, 1], label='Actual', color='steelblue', s=10)
	plot.set_title('Score: ' + str(score))
	plot.set_xlabel('Index')
	plot.set_ylabel(parameter)
	plot.grid()


if __name__ == '__main__':
	main()
