from ..constants import *

class Config:
    def __init__(self, version, submitters, skip_output_file = False):
        self.version = version
        self.submitters = submitters
        self.skip_output_file = skip_output_file
        
    def check_submitter(self, submitter):
        if self.submitters is None:
            return True
        return submitter in self.submitters
    
    def get_datagen_required_files(self):
        return DATAGEN_REQUIRED_FILES[self.version]
    
    def get_run_required_files(self):
        return RUN_REQUIRED_FILES[self.version]
    
    def get_checkpoint_required_files(self):
        return CHECKPOINT_REQUIRED_FILES[self.version]
    
    def get_datagen_required_folders(self):
        return DATAGEN_REQUIRED_FOLDERS[self.version]
    
    def get_run_required_folders(self):
        return RUN_REQUIRED_FOLDERS[self.version]
    
    def get_checkpoint_required_folders(self):
        return CHECKPOINT_REQUIRED_FOLDERS[self.version]
    
    def get_num_train_files(self, model):
        return NUM_DATASET_TRAIN_FILES[model]
    
    def get_num_eval_files(self, model):
        return NUM_DATASET_EVAL_FILES[model]
    
    def get_checkpoint_file(self, model):
        return CHECKPOINT_FILE_MAP[model]