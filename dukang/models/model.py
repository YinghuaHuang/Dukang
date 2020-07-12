"""Base class for distilling models."""
import tensorflow as tf 

from opennmt.models import Model

class DistillModel(Model):
  """Distillation from a teacher to student network.
  described in https://arxiv.org/abs/1503.02531.
  """
  def __init__(self, teacher_model,
               student_model,
               distill_loss_rate=1.0,
               student_loss_rate=1.0,
               distill_temperature=2.0):
    super(Model, self).__init__()
    self.params = {}
    self.initialized = False
    self._frozen_layers = False
    self._teacher_model = teacher_model
    self._teacher_model.trainable = False
    self._student_model = student_model
    self.distill_loss_rate = distill_loss_rate
    self.student_loss_rate = student_loss_rate
    self.distill_temperature = distill_temperature

  @property
  def teacher_model(self):
    return self._teacher_model
  
  @property
  def examples_inputter(self):
    return self._student_model.examples_inputter

  @property
  def student_model(self):
    return self._student_model

  def initialize(self, data_config, params=None):
    self._teacher_model.initialize(data_config, params=params)
    self._student_model.initialize(data_config, params=params)
    self.params = params
    dropout = self.params.get("dropout")
    if dropout is not None:
      misc.set_dropout(self, dropout)
    self.initialized = True

  def call(self, features, labels=None, training=None, step=None):
    student_outputs, student_predictions = \
      self._student_model.call(features, labels, training, step)
    if training and labels:
      teacher_outputs, teacher_predictions = \
          self._teacher_model.call(features, labels, training, step)
      return (student_outputs, teacher_outputs), (student_predictions, teacher_predictions)
    else:
      return student_outputs, student_predictions

  def compute_loss(self, outputs, labels, training=True):
    """Computes the loss.
      loss = distill_loss_rate * distill_loss + student_model_loss
    """
    if training:
      student_model_loss = self._student_model.compute_loss(outputs[0], labels)
      if not isinstance(outputs[0], dict):
        logits = outputs[0] / self.distill_temperature
        teacher_labels = tf.nn.softmax(outputs[1] / distill_temperature)
      else:
        logits = outputs[0]['logits'] / self.distill_temperature
        teacher_labels = tf.nn.softmax(outputs[1]['logits'] / self.distill_temperature)
      distill_loss = tf.nn.softmax_cross_entropy_with_logits(teacher_labels, logits)
      distill_loss = tf.reduce_sum(distill_loss)
      if isinstance(student_model_loss, tuple):
        actual_student_loss = student_model_loss[0]
      else:
        actual_student_loss = student_model_loss
      total_loss = self.distill_temperature ** 2 * distill_loss * self.distill_loss_rate + actual_student_loss * self.student_loss_rate
      tf.summary.scalar("distill_loss", distill_loss)
      tf.summary.scalar("student_model_loss", actual_student_loss)
      if isinstance(student_model_loss, tuple):
        return tuple([total_loss] + list(student_model_loss[1:]))
      else:
        return total_loss
    else:
      return self._student_model.compute_loss(outputs, labels)

  def get_metrics(self):
    return self._student_model.get_metrics()

  def update_metrics(self, metrics, predictions, labels):
    return self._student_model.update_metrics(metrics, predictions, labels)

  def print_prediction(self, prediction, params=None, stream=None):
    return self._student_model.print_prediction(prediction, params=params, stream=stream)
  
  def auto_config(self, num_replicas=1):
    return self._student_model.auto_config(num_replicas)
