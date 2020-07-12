import os

from opennmt.config import load_model

from dukang.models import DistillModel

class BaseDistill(DistillModel):
  def __init__(self, model_config=None):
    if not model_config:
      raise ValueError("distill model must include model_config.")
    os.makedirs(model_config["teacher"]["model_dir"], exist_ok=True)
    os.makedirs(model_config["student"]["model_dir"], exist_ok=True)        
    teacher_model = load_model(model_config["teacher"]["model_dir"],
                               model_file=model_config["teacher"].get("model", None),
                               model_name=model_config["teacher"].get("model_type", None))
    student_model = load_model(model_config["student"]["model_dir"],
                               model_file=model_config["student"].get("model", None),
                               model_name=model_config["student"].get("model_type", None))
    super(BaseDistill, self).__init__(
        teacher_model=teacher_model,
        student_model=student_model,
        distill_loss_rate=model_config.get("distill_loss_rate", 0.75),
        student_loss_rate=model_config.get("student_loss_rate", 0.25),
        distill_temperature=model_config.get("distill_temperature", 2.0))
