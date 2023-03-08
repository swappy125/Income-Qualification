import pickle
import os


class File_Operation:
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.model_directory = 'D:\Python Training\ML Project\Income_Qualification\Assignment\Model\\'

    def save_model(self, model, filename):
        """
        Saving Best model with highest accuracy and minimum features
        """
        self.logger_object.log(
            self.file_object, "Entered the save_model of File_Operation class.")
        try:
            path = os.path.join(self.model_directory)

            with open(path + '\\' + filename + '.sav',
                      'wb') as f:
                # save the model to file
                pickle.dump(model, f)

            self.logger_object.log(self.file_object, 'Model File ' + filename + ' saved. Exited the save_model method '
                                                                                'of the File_Operation class')

            return 'success'

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in save_model method of the File_Operation class. Exception '
                                   'message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model File ' + filename + ' could not be saved. Exited the save_model method of '
                                                              'the File_Operation class')
            raise Exception()
