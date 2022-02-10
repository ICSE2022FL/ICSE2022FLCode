class ProcessedData:

    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.data_df = self.raw_data.data_df
        self.feature_df = self.raw_data.feature_df
        self.label_df = self.raw_data.label_df
        self.file_dir = self.raw_data.file_dir
        self.fault_line = self.raw_data.fault_line
        self.program = self.raw_data.program
        self.bug_id = self.raw_data.bug_id
        self.rest_columns = []

    def feature_selection(self):
        raise Exception("Not implemented.")

    def data_synthesis(self):
        raise Exception("Not implemented.")
