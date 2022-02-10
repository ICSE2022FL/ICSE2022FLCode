class DataLoader:
    def __init__(self, base_dir, program, bug_id):
        self.base_dir = base_dir
        self.program = program
        self.bug_id = bug_id
        self.file_dir = None
        self.feature_df = None
        self.label_df = None
        self.data_df = None
        self.fault_line = None
        self.rest_columns = []

    def load(self):
        pass

    def _load_features(self):
        pass

    def _load_labels(self):
        pass

    def _load_data(self):
        pass
