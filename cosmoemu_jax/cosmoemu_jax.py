## Authtor: Pierre Burger
## Date: 2025-05-27
## Description:
# This module provides a JAX-based emulator for predicting arbirtray features from input parameters using a neural network.
# It is an adapted version of CosmoPower Jax emulator (https://github.com/dpiras/cosmopower-jax), which is a JAX-based emulator for predicting power spectra from input parameters using a neural network.

import pickle
import numpy as onp
import jax.numpy as jnp
from jax import random
from jax.nn import sigmoid
from jax import jacfwd, jacrev, jit
from functools import partial
import jax
import optax
from tqdm import trange

class EmulatorJAX:
    """General-purpose JAX-based emulator for predicting features from input parameters using a neural network.

    Supports loading pretrained networks stored in a specific npz format. Designed for rapid evaluation
    and differentiation of outputs with respect to inputs.

    Parameters
    ----------
    filepath : string
        Full path to the .pkl file containing the pretrained model.
    verbose : bool, default=True
        Whether to print information during initialization.
    """

    def __init__(self, filepath=None,
                 verbose=True):
        """
        JAX-based emulator for predicting features from input parameters.

        If `filepath` is given, loads pretrained model. Otherwise, initializes with provided components.
        """

        if filepath is not None:
            if verbose:
                print(f"Loading model from {filepath}")
            loaded_variable_dict = onp.load(filepath, allow_pickle=True)

            if verbose:
                print(f"Loaded keys: {loaded_variable_dict.keys()}")


            self.latent_params = self._convert_weights_to_jax(
                    loaded_variable_dict['weights'], 
                    loaded_variable_dict['hyper_params']
            )

            self.parameters = loaded_variable_dict['parameters']  
            self.n_parameters = loaded_variable_dict['n_parameters']
            
            self.features_subtraction = loaded_variable_dict['features_subtraction']
            self.features_scaling = loaded_variable_dict['features_scaling']
            
            self.parameters_subtraction = loaded_variable_dict['parameters_subtraction']
            self.parameters_scaling = loaded_variable_dict['parameters_scaling']
          
    
    def _convert_weights_to_jax(self, weights, hyper_params):
                weights_jax = [(jnp.array(w), jnp.array(b)) for w, b in weights]
                hyper_params_jax = [(jnp.array(a), jnp.array(bh)) for a, bh in hyper_params]
                return (weights_jax, hyper_params_jax)  
        
    def _dict_to_ordered_arr_jax(self, input_dict):
        """Convert dictionary of input parameters to ordered array based on trained model."""
        if self.parameters is not None:
            return jnp.stack([input_dict[k] for k in self.parameters], axis=1)
        else:
            return jnp.stack([input_dict[k] for k in input_dict], axis=1)

    @partial(jit, static_argnums=0)
    def _activation(self, x, a, b):
        """Custom activation function as used in training.
        Parameters `a` and `b` control the nonlinearity.
        """
        return jnp.multiply(jnp.add(b, jnp.multiply(sigmoid(jnp.multiply(a, x)), jnp.subtract(1., b))), x)

    @partial(jit, static_argnums=0)
    def apply_dropout_fn(self,args):
        act, key, rate = args
        key, subkey = jax.random.split(key)
        keep_prob = 1.0 - rate
        mask = jax.random.bernoulli(subkey, p=keep_prob, shape=act.shape)
        return mask * act / keep_prob

    @partial(jit, static_argnums=0)
    def no_dropout_fn(sel,args):
        act, key, rate = args
        return act

    @partial(jit, static_argnums=0)
    def _predict(self, latent_params, input_vec, key = jax.random.PRNGKey(0), dropout_rate = 0.0):
        weights, hyper_params = latent_params
        layer_out = [input_vec]

        for i in range(len(weights) - 1):
            w, b = weights[i]
            alpha, beta = hyper_params[i]
            act = jnp.dot(layer_out[-1], w.T) + b
            
            activated = self._activation(act, alpha, beta)
            
            activated = jax.lax.cond(
                dropout_rate > 0.0,
                self.apply_dropout_fn,
                self.no_dropout_fn,
                operand=(activated, key, dropout_rate)
            )
            
            layer_out.append(activated)
                
        w, b = weights[-1]
        preds = jnp.dot(layer_out[-1], w.T) + b
        return preds.squeeze()
    

    @partial(jit, static_argnums=0)
    def predict(self, latent_params, input_vec, key = jax.random.PRNGKey(0), dropout_rate = 0.0):
        
        if len(input_vec.shape) == 1:
            input_vec = input_vec.reshape(-1, self.n_parameters)
        assert len(input_vec.shape) == 2

        return self._predict(latent_params, input_vec, key, dropout_rate)

    
    @partial(jit, static_argnums=0)
    def rescaled_predict(self, input_vec, key = jax.random.PRNGKey(0), dropout_rate = 0.0):
        """Return emulator prediction scaled to match physical values."""
        if isinstance(input_vec, dict):
            input_vec = self._dict_to_ordered_arr_jax(input_vec)
            
        input_vec = (input_vec - self.parameters_subtraction) / self.parameters_scaling
        return self.predict(self.latent_params, input_vec, key, dropout_rate) * self.features_scaling + self.features_subtraction

    @partial(jit, static_argnums=0)
    def ten_to_rescaled_predict(self, input_vec, key = jax.random.PRNGKey(0), dropout_rate = 0.0):
        """Return 10^rescaled prediction, useful for log-scaled outputs."""
        return 10 ** self.rescaled_predict(input_vec, key, dropout_rate)

    @partial(jit, static_argnums=0)
    def compute_loss(self, latent_params, training_parameters, training_features, key = jax.random.PRNGKey(0), dropout_rate = 0.0):
        predictions = self.predict(latent_params, training_parameters, key, dropout_rate)
        return jnp.sqrt(jnp.mean((predictions - training_features) ** 2))
    
    @partial(jit, static_argnums=0)
    def compute_gradients(self, latent_params, training_parameters, training_features, key = jax.random.PRNGKey(0), dropout_rate = 0.0):
        loss_fn = lambda p: self.compute_loss(p, training_parameters, training_features, key, dropout_rate)
        _, grads = jax.value_and_grad(loss_fn)(latent_params)
        return grads

    def train(self, 
            parameters,
            raw_training_parameters,
            raw_training_features,
            normalise_mode = 'mean_sigma',
            learning_rate=1e-3,
            epochs=1000,
            n_hidden=[32]*4,
            save_fn=None,
            verbose=True,
            batch_size=64,
            patience=1000,
            validation_split=0.1,
            update_lr=False,
            lr_decay=0.5,
            min_lr=1e-6,
            dropout_rate=0.0,
            random_seed=0):
        """
        Trains the emulator using JAX and optax with internal validation split,
        mini-batching, and adaptive learning rate scheduling.

        Parameters
        ----------
        parameters : list of str
            Names of the input parameters (used for ordering and tracking).
        raw_training_parameters : dict or jnp.ndarray
            Full dataset of input parameters.
        raw_training_features : jnp.ndarray
            Full dataset of output features.
        learning_rate : float
            Initial learning rate for the optimizer.
        epochs : int
            Number of training epochs.
        n_hidden : list of int
            Hidden layer sizes.
        save_fn : str
            Path to save the trained model parameters.
        verbose : bool
            If True, print training progress.
        batch_size : int
            Size of each mini-batch during training.
        patience : int
            Number of epochs to wait for improvement before early stopping.
        validation_split : float
            Fraction of data used for validation.
        update_lr : bool
            Whether to reduce learning rate on plateau.
        lr_decay : float
            Factor by which to reduce the learning rate.
        min_lr : float
            Minimum learning rate threshold.
        random_seed : int
            Random seed for reproducibility.
        """

        self.parameters = parameters
        self.n_parameters = len(parameters)

        # Convert dict input to ordered array if needed
        if isinstance(raw_training_parameters, dict):
            raw_training_parameters = self._dict_to_ordered_arr_jax(raw_training_parameters)

        # === Shuffle and split into training/validation sets ===
        n_total = raw_training_parameters.shape[0]
        n_val = int(n_total * validation_split)
        rng = onp.random.default_rng(seed=random_seed)
        indices = rng.permutation(n_total)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        train_params_raw = raw_training_parameters[train_idx]
        val_params_raw = raw_training_parameters[val_idx]
        train_features_raw = raw_training_features[train_idx]
        val_features_raw = raw_training_features[val_idx]

        # Save the min/max range for each parameter (optional use later)
        training_range = {
            param: (jnp.min(raw_training_parameters[:, i]), jnp.max(raw_training_parameters[:, i]))
            for i, param in enumerate(self.parameters)
        }

        # === Normalize parameters and features (standardization) ===
        if(normalise_mode == 'mean_sigma'):
            parameters_subtraction = jnp.mean(train_params_raw, axis=0)
            parameters_scaling = jnp.std(train_params_raw, axis=0)
            features_subtraction = jnp.mean(train_features_raw, axis=0)
            features_scaling = jnp.std(train_features_raw, axis=0)

            training_parameters = (train_params_raw - parameters_subtraction) / parameters_scaling
            training_features = (train_features_raw - features_subtraction) / features_scaling
            validation_parameters = (val_params_raw - parameters_subtraction) / parameters_scaling
            validation_features = (val_features_raw - features_subtraction) / features_scaling
            
        elif(normalise_mode == 'min_max'):
            
            parameters_subtraction = jnp.min(train_params_raw, axis=0)
            parameters_scaling = jnp.max(train_params_raw, axis=0) - parameters_subtraction
            
            features_subtraction = jnp.min(train_features_raw, axis=0)
            features_scaling = jnp.max(train_features_raw, axis=0) - features_subtraction

            training_parameters = (train_params_raw - parameters_subtraction) / parameters_scaling
            training_features = (train_features_raw - features_subtraction) / features_scaling
            validation_parameters = (val_params_raw - parameters_subtraction) / parameters_scaling
            validation_features = (val_features_raw - features_subtraction) / features_scaling
        
        else:
            
            if verbose:
                print(f"Not appying any normalisation to output and input features. We recommed to either apply min_max or mean_sigma normalisation.")
                
            parameters_subtraction = jnp.zeros(train_params_raw.shape[1])
            parameters_scaling = jnp.ones(train_params_raw.shape[1])
            features_subtraction = jnp.zeros(train_features_raw.shape[1])
            features_scaling = jnp.ones(train_features_raw.shape[1])

            training_parameters = train_params_raw 
            training_features = train_features_raw 
            validation_parameters = val_params_raw
            validation_features = val_features_raw
        
            
        # Store normalization values for inference
        self.parameters_subtraction = parameters_subtraction
        self.parameters_scaling = parameters_scaling
        self.features_subtraction = features_subtraction
        self.features_scaling = features_scaling
        
        self.n_hidden = n_hidden

        # === Initialize neural network weights and biases ===
        rng = random.PRNGKey(random_seed)
        n_inputs = training_parameters.shape[1]
        n_outputs = training_features.shape[1]
        layer_sizes = [n_inputs] + self.n_hidden + [n_outputs]

        weights, hyper_params = [], []
        for i in range(len(layer_sizes) - 1):
            rng, key_w, key_b = random.split(rng, 3)
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            limit = jnp.sqrt(6 / (fan_in + fan_out))  # Glorot uniform init
            W = random.uniform(key_w, shape=(fan_out, fan_in), minval=-limit, maxval=limit)
            b = jnp.zeros(fan_out)
            weights.append((W, b))
            
            # If not the final layer, add hyperparameters for custom activation
            if i < len(layer_sizes) - 2:
                a = jnp.ones(fan_out)
                b_h = jnp.ones(fan_out) * 0.1
                hyper_params.append((a, b_h))

        latent_params = (weights, hyper_params)

        # === Set up optimizer ===
        def create_optimizer(lr): return optax.adam(lr)
        optimizer = create_optimizer(learning_rate)
        opt_state = optimizer.init(latent_params)

        n_samples = training_parameters.shape[0]
        num_batches = int(jnp.ceil(n_samples / batch_size))
        losses, val_losses = [], []

        best_val_loss = float('inf')
        best_params = None
        wait = 0  # Counter for early stopping / LR scheduling

        if verbose:
            print(f"Training with {n_samples} samples and {n_val} validation samples.")

        onp.random.seed(random_seed)
        # === Training loop ===
        with trange(epochs) as t:
            for epoch in t:
                # Shuffle training data each epoch
                perm = jnp.array(onp.random.permutation(n_samples))
                params_shuffled = training_parameters[perm]
                features_shuffled = training_features[perm]

                # Mini-batch loop
                for batch_idx in range(num_batches):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, n_samples)
                    x_batch = params_shuffled[start:end]
                    y_batch = features_shuffled[start:end]

                    key = jax.random.PRNGKey(batch_idx + epoch * num_batches)
                    grads = self.compute_gradients(latent_params, x_batch, y_batch, key=key, dropout_rate=dropout_rate)
                    updates, opt_state = optimizer.update(grads, opt_state)
                    latent_params = optax.apply_updates(latent_params, updates)

                # Evaluate full training and validation loss
                full_train_loss = self.compute_loss(latent_params, training_parameters, training_features, dropout_rate=0.0)
                val_loss = self.compute_loss(latent_params, validation_parameters, validation_features, dropout_rate=0.0)
                losses.append(full_train_loss)
                val_losses.append(val_loss)
                
                # update the progressbar
                t.set_postfix(train_loss=full_train_loss,validation_loss=val_loss,learning_rate=learning_rate)

                # === Learning rate scheduling OR early stopping ===
                if update_lr:
                    if val_loss < best_val_loss - 1e-6:
                        best_val_loss = val_loss
                        best_params = latent_params
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            new_lr = max(learning_rate * lr_decay, min_lr)
                            if new_lr < learning_rate:
                                learning_rate = new_lr
                                optimizer = create_optimizer(learning_rate)
                                latent_params = best_params
                                opt_state = optimizer.init(latent_params)
                                wait = 0
                                if verbose:
                                    print(f"Epoch {epoch}: Reducing learning rate to {learning_rate:.2e}")
                            else:
                                if verbose:
                                    print(f"Epoch {epoch}: Learning rate already at minimum {min_lr}")
                                break  # Early stopping
                else:
                    # Only early stopping
                    if val_loss < best_val_loss - 1e-6:
                        best_val_loss = val_loss
                        best_params = latent_params
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            break  # Early stopping

                # # Optional logging
                # if verbose and epoch % 50 == 0:
                #     print(f"Epoch {epoch:4d}, Train Loss: {full_train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # === Save best weights and stats ===
        self.latent_params = best_params
        if verbose:
            print(f"Training complete. Best val loss: {best_val_loss:.6f}. Model saved to {save_fn}")
        
        # Save to disk as a .npz file (weights as object arrays)
        onp.savez(save_fn, 
                weights=onp.array(self.latent_params[0], dtype=object),
                hyper_params=self.latent_params[1],
                parameters=self.parameters,
                n_parameters=self.n_parameters,
                parameters_subtraction=self.parameters_subtraction,
                parameters_scaling=self.parameters_scaling,
                features_subtraction=self.features_subtraction,
                features_scaling=self.features_scaling,
                training_range=training_range,
                train_loss=losses,
                val_losses=val_losses
                )