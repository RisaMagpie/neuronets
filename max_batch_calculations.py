from batchflow import Pipeline, D, B, V, C, R, P
from batchflow.opensets import Imagenette160
from batchflow.models.torch import UNet
from batchflow import GPUMemoryMonitor
from batchflow.models.torch import EncoderDecoder
import torch
import numpy as np
import nvidia_smi
import time

def get_mem_info(device_id):
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_id)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    free_memory_in_bytes = info.free
    used_memory_in_bytes = info.used
    nvidia_smi.nvmlShutdown()
    return free_memory_in_bytes, used_memory_in_bytes

def get_run_mem(dataset, device_id, model_config, train_pipeline, batch_size=16, n_iters=50):
    with GPUMemoryMonitor(gpu_list=[device_id]) as monitor:
        torch.cuda.empty_cache()
        #time.sleep(60)
        train_pipeline.run(batch_size, n_iters=n_iters, bar=True)
    return np.max(monitor.data)

def get_max_batch_size(dataset, device_id, model_config, train_pipeline, init_batch_size, n_iters):
    first_run_memory = get_run_mem(dataset, device_id, model_config, train_pipeline, batch_size=init_batch_size, n_iters=n_iters)
    second_run_memory = get_run_mem(dataset, device_id, model_config, train_pipeline, batch_size=2*init_batch_size, n_iters=n_iters)
    max_batch_size = init_batch_size * (100 - 2 * first_run_memory + second_run_memory)/(second_run_memory - first_run_memory)
    return max_batch_size

def main():
    dataset = Imagenette160(bar=True)
    device_id = 3

    model_config = dict(model = UNet)
    model_config['device'] = f'cuda:{device_id}'
    model_config['loss'] = 'mse'

    train_pipeline = (dataset.train.p
                    .crop(shape=(160, 160), origin='center')
                    .init_variable('loss_history', [])
                    .to_array(channels='first', dtype=np.float32)
                    .multiply(1./255)
                    .init_model('dynamic', UNet, 'unet',
                                config=model_config)
                    .train_model('unet', B.images, B.images, 
                                 fetches='loss', save_to=V('loss_history', mode='a'), use_lock=True)
    )
    
    #init_batch_size = 16
    n_iters = 50
    #print("Max batch size:", get_max_batch_size(dataset, device_id, model_config, train_pipeline, init_batch_size, n_iters))
    
    init_batch_size = 16
    for i in range(5):
        print("Max batch size:", get_max_batch_size(dataset, device_id, model_config, train_pipeline, init_batch_size, n_iters))
    
if __name__ == "__main__":
    main()