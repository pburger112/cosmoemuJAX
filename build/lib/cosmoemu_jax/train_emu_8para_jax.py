from __future__ import print_function
import numpy as np
import jax.numpy as jnp
from astropy.table import Table
from cosmoemu_jax import EmulatorJAX

# Set device
device = 'cpu'
print(device)

# Configuration
bin = 1
run = 1
print(run)

# Load data
features = np.log10(np.load(f'outputs_emulation/dsigma_bin{bin}_8Param_mocks_cleaned_AllVersions.npy'))
HOD_params = Table.read(f'outputs_emulation/HOD_param_table_8Params_Bin{bin}_AllVersions.fits')
HOD_params['index'] = np.arange(len(HOD_params))
HOD_params.sort('chi2_joint')

# Train/test split
np.random.seed(0)
all_indices = np.arange(0, len(HOD_params))
test_sample = np.random.choice(range(500), 500, replace=False)[:250]
train_sample = np.setdiff1d(all_indices, test_sample)

print(f"train_sample: {len(train_sample)}")
print(f"test_sample: {len(test_sample)}")


# Convert to 2D JAX arrays
para_names = HOD_params.colnames[:-5]
print(para_names)
train_params = {}
test_params = {}
for i in range(len(para_names)):
    train_params[para_names[i]]=jnp.array(HOD_params[para_names[i]][train_sample])
    test_params[para_names[i]]=jnp.array(HOD_params[para_names[i]][test_sample])

# train_params = jnp.stack([jnp.array(HOD_params[p][train_sample]) for p in para_names], axis=-1)
# test_params = jnp.stack([jnp.array(HOD_params[p][test_sample]) for p in para_names], axis=-1)

# Convert features to JAX arrays
train_features = jnp.array(features[HOD_params['index'][train_sample]])
test_features = jnp.array(10 ** features[HOD_params['index'][test_sample]])

print(train_params)
print(train_features)

# Train emulator
jax_nn = EmulatorJAX(verbose=True)
jax_nn.train(
    parameters=para_names,
    raw_training_parameters=train_params,
    raw_training_features=train_features,
    
    learning_rate = 1e-3,
    epochs = 10000,
    n_hidden = [32] * 4,
    batch_size = 100,
    
    validation_split = 0.05,
    patience = 200,

    update_lr = False,
    lr_decay=0.1,
    min_lr=1e-6,
    
    random_seed=0,
    save_fn=f'outputs_emulation/dsigma_emulator_jax_8Param_bin{bin}_run{run}',
)

# Load trained emulator and predict
jax_nn = EmulatorJAX(filepath=f'outputs_emulation/dsigma_emulator_jax_8Param_bin{bin}_run{run}.npz')
emu_features = jax_nn.ten_to_rescaled_predict(test_params)

# Save outputs (as NumPy arrays for compatibility)
np.savez(
    f'outputs_emulation/dsigma_jax_8Param_bin{bin}_run{run}_test_features',
    emu_features=np.array(emu_features),
    test_features=np.array(test_features),
    test_params=np.array(test_params),
    test_sample=test_sample
)