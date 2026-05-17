import re

with open('eeg_simulation.py', 'r') as f:
    content = f.read()

# 1. Revert monkey-patch
old_fit_block = """    all_val_loss = []

    # Inject a LambdaCallback to store metrics since net.fit() doesn't expose the Keras history natively
    history_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: [
            all_loss.append(logs.get('loss')) if 'loss' in logs else None,
            all_val_loss.append(logs.get('val_loss')) if 'val_loss' in logs else None
        ]
    )
    original_fit = net.model.fit
    def custom_fit(*args, **kwargs):
        callbacks = kwargs.get('callbacks', [])
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        callbacks.append(history_callback)
        kwargs['callbacks'] = callbacks
        return original_fit(*args, **kwargs)
    net.model.fit = custom_fit

    for chunk_idx in range(num_chunks):
        logging.info(f"\\n--- Processing Chunk {chunk_idx + 1}/{num_chunks} ---")

        # 1. Simulate a batch
        sim_train = Simulation(fwd, info, settings=sim_settings)
        sim_train.simulate(n_samples=chunk_size)

        # 2. Fit the model (weights are preserved across calls)
        net.fit(sim_train, epochs=10, validation_split=0.1)"""

new_fit_block = """    all_val_loss = []

    for chunk_idx in range(num_chunks):
        logging.info(f"\\n--- Processing Chunk {chunk_idx + 1}/{num_chunks} ---")

        # 1. Simulate a batch
        sim_train = Simulation(fwd, info, settings=sim_settings)
        sim_train.simulate(n_samples=chunk_size)

        # 2. Fit the model (weights are preserved across calls)
        history = net.fit(sim_train, epochs=10, validation_split=0.1)

        if hasattr(history, 'history'):
            all_loss.extend(history.history.get('loss', []))
            all_val_loss.extend(history.history.get('val_loss', []))"""

content = content.replace(old_fit_block, new_fit_block)

# 2. Revert save approach
old_save_block = """        # 3. Save progress locally to Drive safely
        chunk_dir = os.path.join(checkpoints_dir, f'convdip_checkpoint_chunk_{chunk_idx+1}')
        if os.path.exists(chunk_dir):
            shutil.rmtree(chunk_dir)
        os.makedirs(chunk_dir, exist_ok=True)

        # Safely detach and save model due to esinet/keras serialization bugs
        keras_model = net.model
        net.model = None
        joblib.dump(net, os.path.join(chunk_dir, 'net_wrapper.pkl'))
        keras_model.save(os.path.join(chunk_dir, 'model.keras'))
        net.model = keras_model"""

new_save_block = """        # 3. Save progress locally to Drive
        chunk_dir = os.path.join(checkpoints_dir, f'convdip_checkpoint_chunk_{chunk_idx+1}')
        if os.path.exists(chunk_dir):
            shutil.rmtree(chunk_dir)
        os.makedirs(chunk_dir, exist_ok=True)
        net.save(chunk_dir)"""

content = content.replace(old_save_block, new_save_block)

with open('eeg_simulation.py', 'w') as f:
    f.write(content)
