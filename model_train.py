from anomalib.data import Folder    
from anomalib.models import Padim
from anomalib.engine import Engine
from anomalib.deploy import ExportType, OpenVINOInferencer
import torch

datamodule = Folder(
    name="bottle_cap_dataset",
    root="dataset",
    normal_dir="augmented_good",
    abnormal_dir="augmented_bad",
    test_split_ratio=0.3,
    seed=42,
)

class myPadim(Padim):
    def __init__(
        self,
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
        pre_trained=True,
        n_features=None,
        pre_processor=True,
        post_processor=True,
        evaluator=True,
        visualizer=True,
    ):
        super().__init__(
            backbone=backbone,
            layers=layers,
            pre_trained=pre_trained,
            n_features=n_features,
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

    @property
    def trainer_arguments(self):
        return {"max_epochs": 3, "val_check_interval": 1.0, "num_sanity_val_steps": 0, "devices": 1}


model = myPadim()
engine = Engine()
torch.cuda.empty_cache()
engine.train(model=model, datamodule=datamodule)
engine.test(model=model, dataloaders=datamodule)

