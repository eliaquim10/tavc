from .utils import *
from .libs import *

class Trainer:
    def __init__(self,
                 model,
                 loss,
                 learning_rate,
                 checkpoint_dir='./ckpt/eddenoise'):

        self.now = None
        self.loss = loss
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)

        self.restore()

    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, steps, evaluate_every=1000, save_best_only=False):
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()
        with open("loss_denoisecnn.csv", "a") as file:
            file.write(f'geral\n')
            with tf.device("/GPU:0"):
                for noise, real in train_dataset.take(steps - ckpt.step.numpy()):
                    
                    ckpt.step.assign_add(1)
                    step = ckpt.step.numpy()

                    loss = self.train_step(noise, real)
                    loss_mean(loss)
                    file.write(f'{loss_mean.result():.4f}\n')

                    if step % evaluate_every == 0:
                        loss_value = loss_mean.result()
                        loss_mean.reset_states()

                        # Compute PSNR on validation dataset
                        psnr_value = self.evaluate(valid_dataset)
                        mae_value = self.evaluate_mae(valid_dataset)
                        with train_summary_writer.as_default():
                            tf.summary.scalar('LOSS', loss, step=step)
                            tf.summary.scalar('MSE', loss_value, step=step)
                            tf.summary.scalar('MAE', mae_value, step=step)
                            tf.summary.scalar('PSNR', psnr_value, step=step)

                        duration = time.perf_counter() - self.now
                        print(f'{step}/{steps}: LOSS = {loss.numpy():.3f}, MSE = {loss_value.numpy():.3f}, PSNR = {psnr_value.numpy():3f}, MAE = {mae_value.numpy():3f} ({duration:.2f}s)')

                        if save_best_only and mae_value >= ckpt.mae:
                        # if save_best_only and psnr_value <= ckpt.psnr:
                            self.now = time.perf_counter()
                            # skip saving checkpoint, no PSNR improvement
                            continue

                        ckpt.psnr = psnr_value
                        ckpt.mae = mae_value
                        ckpt_mgr.save()

                        self.now = time.perf_counter()

    @tf.function
    def train_step(self, noise, real):
        with tf.GradientTape() as tape:
            noise = tf.cast(noise, tf.float32)
            real = tf.cast(real, tf.float32)

            denoise = self.checkpoint.model(noise, training=True)
            loss_value = self.loss(real, denoise)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value

    def evaluate(self, dataset):
        return evaluate(self.checkpoint.model, dataset)

    def evaluate_mae(self, dataset):
        return evaluate_mae(self.checkpoint.model, dataset)

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')
            
class DenoiseganTrainer:
    #
    # TODO: model and optimizer checkpoints
    #
    def __init__(self,
                 generator,
                 discriminator,
                 d_loss, g_loss,
                 g_learning_rate=PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5]),
                 d_learning_rate=PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5]),
                 Lambda = 0.001, 
                 ):
                 
        self._lambda = Lambda
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = Adam(learning_rate=g_learning_rate)
        self.discriminator_optimizer = Adam(learning_rate=d_learning_rate)

        self.d_loss = d_loss
        self.g_loss = g_loss

        self.binary_cross_entropy = BinaryCrossentropy(from_logits=True)
        self.mean_squared_error = MeanSquaredError()

    def train(self, train_dataset, steps=200000):
        pls_metric = Mean()
        gls_metric = Mean()
        dls_metric = Mean()
        step = 0
        
        with open("loss.csv", "a") as file:
            file.write(f'geral; gerador; descriminador \n')
            # with tf.device("/GPU:0"):

            with tf.device("/GPU:0"):
                for noise, real in train_dataset.take(steps):
                    step += 1
                    # print(tf.shape(noise), tf.shape(real))
                    gl, pl, dl = self.train_step(noise, real)

                    pls_metric(pl)
                    gls_metric(gl)
                    dls_metric(dl)

                    file.write(f'{pls_metric.result():.4f}; {gls_metric.result():.4f}; {dls_metric.result():.4f} \n')

                    if step % 50 == 0:
                        print(f'{step}/{steps}, perceptual loss = {pls_metric.result():.4f}, gan loss = {gls_metric.result():.4f}, discriminator loss = {dls_metric.result():.4f}')
                        pls_metric.reset_states()
                        dls_metric.reset_states()

    # @tf.function
    def train_step(self, noise, real):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            noise = tf.cast(noise, tf.float32)
            real = tf.cast(real, tf.float32)
            
            denoise = self.generator(noise, training=True)
            # print(tf.shape(noise), tf.shape(real), tf.shape(denoise),)

            real_output = self.discriminator([noise, real], training=True)
            denoise_output = self.discriminator([noise, denoise], training=True)
            # print(tf.shape(real_output), tf.shape(denoise_output))

            
            # con_loss = self._content_loss(real, denoise)
            # gen_loss = self._generator_loss(denoise_output)
            # perc_loss = con_loss + self._lambda * gen_loss
            # disc_loss = self._discriminator_loss(real_output, denoise_output)
            disc_loss = self.d_loss(real_output, denoise_output)
            
            # print(tf.shape(denoise_output), tf.shape(denoise), tf.shape(real))
            perc_loss, gen_loss, _ = self.g_loss(denoise_output, denoise, real)
            # print(tf.shape(perc_loss), tf.shape(gen_loss) , tf.shape(disc_loss))

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        # gradients_of_generator = gen_tape.gradient(perc_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return gen_loss, perc_loss, disc_loss

    @tf.function
    def _content_loss(self, real, denoise):
        denoise = preprocess_input(denoise)
        real = preprocess_input(real)
        denoise_features = self.vgg(denoise) / 12.75
        real_features = self.vgg(real) / 12.75
        return self.mean_squared_error(real_features, denoise_features)
        

    def _generator_loss(self, denoise_out):
        return self.binary_cross_entropy(tf.ones_like(denoise_out), denoise_out)

    def _discriminator_loss(self, real_out, denoise_out):
        real_loss = self.binary_cross_entropy(tf.ones_like(real_out), real_out)
        denoise_loss = self.binary_cross_entropy(tf.zeros_like(denoise_out), denoise_out)
        return real_loss + denoise_loss

class DenoiseGeneratorTrainer(Trainer):
    def __init__(self,
                 model,
                 checkpoint_dir,
                 learning_rate=1e-4):
        super().__init__(model, loss=MeanSquaredError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir)

    def train(self, train_dataset, valid_dataset, steps=1000000, evaluate_every=1000, save_best_only=True):
        super().train(train_dataset, valid_dataset, steps, evaluate_every, save_best_only)

def __main__():
    pass
if __name__ == "__main__":
    pass