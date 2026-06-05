import h5py

with h5py.File('llava_activations.h5', 'r') as f:
    data = f['activations'][:] # Load into memory as numpy array
    # or
    first_image = f['activations'][0] # Load only one
    print(data)