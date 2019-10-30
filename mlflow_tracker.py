
from mlflow.tracking import MlflowClient
from mlflow.entities.run import Run
import mlflow
from fastai import *
from fastai.vision import *

# v 1.0.54
class MLFlowTracker(LearnerCallback):
    "A `TrackerCallback` that tracks the loss and metrics into MLFlow"
    def __init__(self, learn:Learner, exp_name: str, params: dict, nb_path: str=' ', uri: str = "http://localhost:5000"):
        super().__init__(learn)
        self.learn = learn
        self.exp_name = exp_name
        self.params = params
        self.nb_path = nb_path
        self.uri = uri
        self.metrics_names = ['train_loss', 'valid_loss'] + [o.__name__ for o in learn.metrics[:1]]

    def on_train_begin(self, **kwargs: Any) -> None:
        "Prepare MLflow experiment and log params"
        self.client = mlflow.tracking.MlflowClient(self.uri)
        exp = self.client.get_experiment_by_name(self.exp_name)
        if exp is None:
            self.exp_id = self.client.create_experiment(self.exp_name)
        else:
            self.exp_id = exp.experiment_id
        run = self.client.create_run(experiment_id=self.exp_id)
        self.run = run.info.run_uuid
        for k,v in self.params.items():
            self.client.log_param(run_id=self.run, key=k, value=v)

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        "Send loss and metrics values to MLFlow after each epoch"
        if kwargs['smooth_loss'] is None or kwargs["last_metrics"] is None:
            return
        metrics = [kwargs['smooth_loss']] + kwargs["last_metrics"]
        for name, val in zip(self.metrics_names, metrics):
            self.client.log_metric(self.run, name, np.float(val))
        #from IPython.core.debugger import Tracer; Tracer()()
        self.client.log_artifact(run_id=self.run, local_path=self.nb_path) # log noteboook locall

    def on_train_end(self, **kwargs: Any) -> None:  
        "Store the notebook and stop run"
        print(self.nb_path)
        self.client.log_artifact(run_id=self.run, local_path=self.nb_path)
        self.client.set_terminated(run_id=self.run)
        
## Todo:

# manually add logging of version control
# perhaps models as well