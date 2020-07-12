"""Main library entrypoint."""

import tensorflow as tf

from opennmt.runner import Runner, _count_batch_accum
from opennmt import evaluation
from opennmt import training as training_util
from opennmt.utils import checkpoint as checkpoint_util
from opennmt.utils import misc

class DistillCheckpoint(checkpoint_util.Checkpoint):
  """Wrapper around onmt-Checkpoint for model distilling."""

  def __init__(self, model, optimizer=None, model_dir=None, keep_checkpoint_max=8, training=False):
    super(DistillCheckpoint, self).__init__(model, optimizer,
                                            model_dir, keep_checkpoint_max)
    self._training = training

  @property
  def model(self):
    if self._training:
      return self._model
    else:
      return self._model.student_model


class DistillRunner(Runner):
  """Class for knowledge distilling, running and exporting models."""

  # pylint: disable=arguments-differ
  def _init_model(self, config, training=False):
    model = misc.clone_layer(self._model)
    model.initialize(config["data"], params=config["params"])
    if "optimizer" in config["params"]:
      optimizer = model.get_optimizer()
    else:
      optimizer = None
    checkpoint = DistillCheckpoint(
        model,
        optimizer=optimizer,
        model_dir=config.get("model_dir"),
        keep_checkpoint_max=config["train"].get("keep_checkpoint_max", 8), training=training)
    return checkpoint

  def _init_run(self, training=False, num_devices=1):
    config = self._finalize_config(training=training, num_devices=num_devices)
    return self._init_model(config, training), config


  def train(self, num_devices=1, with_eval=False, checkpoint_path=None):
    """Runs the training loop.

    Args:
      num_devices: Number of devices to use for training.
      with_eval: Enable evaluation during training.
      checkpoint_path: The checkpoint path to load the model weights from it.

    Returns:
      The path to the final model directory.
    """
    checkpoint, config = self._init_run(num_devices=num_devices, training=True)
    # load teacher model.
    teacher_checkpoint_path = config["model"]["teacher"].get("teacher_checkpoint_path", None)
    if not teacher_checkpoint_path:
      raise ValueError("teacher_checkpoint_path is None.")
    teacher_checkpoint = tf.train.Checkpoint(model=checkpoint.model.teacher_model)
    status = teacher_checkpoint.restore(tf.train.latest_checkpoint(teacher_checkpoint_path))
    # load student model.
    student_checkpoint_path = config["model"]["student"].get("student_checkpoint_path", None)
    if student_checkpoint_path:
      student_checkpoint = tf.train.Checkpoint(model=checkpoint.model.student_model)
      student_checkpoint.restore(tf.train.latest_checkpoint(student_checkpoint_path))
    status = checkpoint.restore(
        checkpoint_path=checkpoint_path, weights_only=checkpoint_path is not None)

    model = checkpoint.model
    data_config = config["data"]
    train_config = config["train"]
    eval_config = config["eval"]

    batch_type = train_config["batch_type"]
    if batch_type == "tokens" and self._mixed_precision:
      batch_size_multiple = 8
    else:
      batch_size_multiple = 1

    dataset = model.student_model.examples_inputter.make_training_dataset(
        data_config["train_features_file"],
        data_config.get("train_labels_file"),
        train_config["batch_size"],
        batch_type=batch_type,
        batch_size_multiple=batch_size_multiple,
        shuffle_buffer_size=train_config["sample_buffer_size"],
        length_bucket_width=train_config["length_bucket_width"],
        maximum_features_length=train_config.get("maximum_features_length"),
        maximum_labels_length=train_config.get("maximum_labels_length"),
        single_pass=train_config.get("single_pass", False),
        prefetch_buffer_size=train_config.get("prefetch_buffer_size"))
    # todo
    if with_eval:
      evaluator = evaluation.Evaluator.from_config(model, config)
    else:
      evaluator = None

    # Set gradients accumulation based on the requested effective batch size.
    if train_config.get("effective_batch_size") is not None:
      accum_steps = _count_batch_accum(
          train_config["batch_size"],
          train_config["effective_batch_size"],
          num_replicas=num_devices)
      tf.get_logger().info(
          "Accumulate gradients of %d iterations to reach effective batch size of %d",
          accum_steps,
          train_config["effective_batch_size"])
    else:
      accum_steps = 1

    trainer = training_util.DistributionStrategyTrainer(
        checkpoint,
        devices=misc.get_devices(count=num_devices))
    trainer(
        dataset,
        max_step=train_config.get("max_step"),
        accum_steps=accum_steps,
        report_steps=train_config.get("save_summary_steps", 100),
        save_steps=train_config.get("save_checkpoints_steps", 5000),
        evaluator=evaluator,
        eval_steps=eval_config.get("steps", 5000),
        export_on_best=eval_config.get("export_on_best"))
    average_last_checkpoints = train_config.get("average_last_checkpoints", 0)
    if average_last_checkpoints > 0:
      return self.average_checkpoints(
          os.path.join(checkpoint.model_dir, "avg"),
          max_count=average_last_checkpoints)
    return checkpoint.model_dir
