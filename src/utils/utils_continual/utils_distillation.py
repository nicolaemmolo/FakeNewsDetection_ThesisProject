# -----------------------------------------------
# ========= TensorFlow Implementation ===========
# -----------------------------------------------

import tensorflow as tf

class Distiller(tf.keras.Model):
    def __init__(self, student, teacher=None, alpha=0.5, temperature=3.0):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.alpha = alpha
        self.temperature = temperature
        self.binary_ce = tf.keras.losses.BinaryCrossentropy(from_logits=True) # from_logits=True because Dense layer without activation
        self.kl_div = tf.keras.losses.KLDivergence()

    def compile(self, optimizer, metrics):
        super(Distiller, self).compile(
            optimizer=optimizer,
            metrics=metrics,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
        )

    def train_step(self, data):
        # Unpack dei dati
        x, y = data

        with tf.GradientTape() as tape:
            # Predizioni dello Student
            student_logits = self.student(x, training=True)
            
            # Loss standard (Cross Entropy)
            student_loss = self.binary_ce(y, student_logits)

            if self.teacher is not None:
                # Predizioni del Teacher (frozen)
                teacher_logits = self.teacher(x, training=False)
                
                # Applichiamo la temperatura per "ammorbidire" le probabilità
                # Trasformiamo i logits in probabilità con sigmoid
                soft_student = tf.nn.sigmoid(student_logits / self.temperature)
                soft_teacher = tf.nn.sigmoid(teacher_logits / self.temperature)
                
                distillation_loss = self.kl_div(soft_teacher, soft_student)
                
                # Loss combinata
                loss = (1 - self.alpha) * student_loss + (self.alpha * (self.temperature**2)) * distillation_loss
            else:
                loss = student_loss

        # Calcolo gradienti e ottimizzazione
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Aggiorna le metriche
        self.compiled_metrics.update_state(y, student_logits)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = loss
        return results

    def call(self, x):
        return self.student(x)



# -----------------------------------------------
# ========== PyTorch Implementation =============
# -----------------------------------------------

import torch
import torch.nn.functional as F
from transformers import Trainer

class KDTrainer(Trainer):
    def __init__(self, teacher_model=None, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha

        if self.teacher is not None:
            self.teacher.eval()
            # Freeze teacher parameters
            for param in self.teacher.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs_student = model(**inputs)
        logits_student = outputs_student.logits

        # Standard cross-entropy loss (Current Task)
        loss_ce = outputs_student.loss if labels is not None else 0.0

        # Knowledge Distillation loss (Previous Tasks)
        loss_kd = 0.0
        if self.teacher is not None:
            self.teacher.to(model.device) # ensure teacher is on the same device
            
            with torch.no_grad():
                output_teacher = self.teacher(**inputs)
                logits_teacher = output_teacher.logits

            T = self.temperature
            loss_kd = F.kl_div(
                F.log_softmax(logits_student / T, dim=1),
                F.softmax(logits_teacher / T, dim=1),
                reduction="batchmean"
            ) * (T ** 2)
        
        if self.teacher is None:
            loss = loss_ce
        else:
            # Combine losses
            loss = (1 - self.alpha) * loss_ce + self.alpha * loss_kd

        return (loss, outputs_student) if return_outputs else loss
