import tensorflow as tf
import numpy as np

np.random.seed(42)
tf.random.set_seed(42)

# Onde EM 1D
num_samples = 1000
x_train = np.random.uniform(low=0, high=10, size=(num_samples, 1))
t_train = np.random.uniform(low=0, high=10, size=(num_samples, 1))

# Onde progressive E et B en phase
E_exact = np.sin(x_train - t_train)
B_exact = np.sin(x_train - t_train)

u_exact = np.concatenate([E_exact, B_exact], axis=1) 

x_train_tf = tf.convert_to_tensor(x_train, dtype=tf.float32)
t_train_tf = tf.convert_to_tensor(t_train, dtype=tf.float32)
u_exact_tf = tf.convert_to_tensor(u_exact, dtype=tf.float32)
input_train = tf.concat([x_train_tf, t_train_tf], axis=1)

# PINN
class MaxwellPINN(tf.keras.Model):
    def __init__(self):
        super(MaxwellPINN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(50, activation='tanh', input_dim=2)
        self.dense2 = tf.keras.layers.Dense(50, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(50, activation='tanh')
        # 2 sorties à E et B
        self.output_layer = tf.keras.layers.Dense(2, activation=None)

    def call(self, inputs):
        hidden1 = self.dense1(inputs)
        hidden2 = self.dense2(hidden1)
        hidden3 = self.dense3(hidden2)
        output = self.output_layer(hidden3)
        return output


# La loss function de Maxwell-Lorentz
def physics_loss(model, x, t):
    # calcul séparé des gradients de E et B
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(t)
        # prédiction
        u_pred = model(tf.concat([x, t], axis=1))
        # Champ électrique
        E = u_pred[:, 0:1]
        # Champ magnétique
        B = u_pred[:, 1:2]

    # gradients
    E_x = tape.gradient(E, x)
    E_t = tape.gradient(E, t)
    B_x = tape.gradient(B, x)
    B_t = tape.gradient(B, t)
    
    # On épargne de la RAM, même si on est sur GPU T4
    del tape 
    # vitesse de la lumière pour la normalisation
    c = 1.0 
    # loi de Faraday
    f_faraday = E_x + B_t
    # loi d'Ampère-Maxwell
    f_ampere = B_x + (1.0 / (c**2)) * E_t
    # la loss physique est la moyenne des carrés des résidus des deux équations
    loss_faraday = tf.reduce_mean(tf.square(f_faraday))
    loss_ampere = tf.reduce_mean(tf.square(f_ampere))

    return loss_faraday + loss_ampere

# training
model = MaxwellPINN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

num_epochs = 1000

for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        # loss physique 
        loss_physique = physics_loss(model, x_train_tf, t_train_tf)
        # loss sur les datas
        predictions = model(input_train)
        loss_donnees = tf.reduce_mean(tf.square(predictions - u_exact_tf))
        # loss totale
        total_loss = loss_physique + loss_donnees

    # calcul des grad
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{num_epochs} | Total: {total_loss:.5f} | Physique: {loss_physique:.5f} | Données: {loss_donnees:.5f}")

x_test = np.linspace(0, 10, 100).reshape(-1, 1)
t_test = np.linspace(0, 10, 100).reshape(-1, 1)

x_test_tf = tf.convert_to_tensor(x_test, dtype=tf.float32)
t_test_tf = tf.convert_to_tensor(t_test, dtype=tf.float32)
input_test = tf.concat([x_test_tf, t_test_tf], axis=1)

# prédiction
pred_test = model(input_test)
E_pred = pred_test[:, 0]
B_pred = pred_test[:, 1]

# test réel du y_test
E_exact_test = np.sin(x_test - t_test).flatten()

# MSE sur le champ E
mse_E = tf.reduce_mean(tf.square(E_exact_test - E_pred))
print(f"\nTest MSE (Champ E): {mse_E.numpy():.6f}")

