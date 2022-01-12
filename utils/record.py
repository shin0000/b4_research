import mlflow
from mlflow.tracking import MlflowClient

class MLFlowRecord():
    def __init__(self):
        self.client = MlflowClient()
        self.experiment_id = "0"
        self.run = None
        self.run_id = None
    def set_experiment(self, name):
        experiment = self.client.get_experiment_by_name(name)
        if experiment is not None:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = self.client.create_experiment(name)
        self.experiment_id = experiment_id
    def start_run(self):
        self.run = self.client.create_run(self.experiment_id)
        self.run_id = self.run.info.run_id
    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)
    def log_metric(self, key, value, step=0):
        self.client.log_metric(self.run_id, key, value, step=step)
    def set_tag(self, key, value):
        self.client.set_tag(self.run_id, key, value)
    def log_artifact(self, path):
        self.client.log_artifact(self.run_id, path)
    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)
    def end_run(self):
        self.client.set_terminated(self.run_id)

class MLFlowRecordDummy():
    def __init__(self):
        pass
    def set_experiment(self, name):
        pass
    def start_run(self):
        pass
    def log_param(self, key, value):
        pass
    def log_metric(self, key, value, step=0):
        pass
    def set_tag(self, key, value):
        pass
    def log_artifact(self, path):
        pass
    def log_param(self, key, value):
        pass
    def end_run(self):
        pass